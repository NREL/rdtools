"""
Functions for calculating soiling metrics from photovoltaic system data.

The soiling module is currently experimental. The API, results,
and default behaviors may change in future releases (including MINOR
and PATCH releases) as the code matures.
"""

import bisect
import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.optimize import curve_fit
from scipy.stats.mstats import theilslopes
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from rdtools import degradation as RdToolsDeg
from rdtools.bootstrap import _make_time_series_bootstrap_samples

lowess = sm.nonparametric.lowess  # Used in CODSAnalysis/Matt

warnings.warn(
    "The soiling module is currently experimental. The API, results, "
    "and default behaviors may change in future releases (including MINOR "
    "and PATCH releases) as the code matures."
)


# Custom exception
class NoValidIntervalError(Exception):
    """raised when no valid rows appear in the result dataframe"""

    pass


class SRRAnalysis:
    """
    Class for running the stochastic rate and recovery (SRR) photovoltaic
    soiling loss analysis presented in Deceglie et al. JPV 8(2) p547 2018

    Parameters
    ----------
    energy_normalized_daily : pandas.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Alternatively, the soiling ratio output of a soiling sensor (e.g. the
        photocurrent ratio between matched dirty and clean PV reference cells).
        In either case, data should be insolation-weighted daily aggregates.
    insolation_daily : pandas.Series
        Daily plane-of-array insolation corresponding to
        `energy_normalized_daily`. Arbitrary units.
    precipitation_daily : pandas.Series, default None
        Daily total precipitation. (Ignored if ``clean_criterion='shift'`` in
        subsequent calculations.)
    """

    def __init__(self, energy_normalized_daily, insolation_daily, precipitation_daily=None):
        self.pm = energy_normalized_daily  # daily performance metric
        self.insolation_daily = insolation_daily
        self.precipitation_daily = precipitation_daily  # daily precipitation
        self.random_profiles = []  # random soiling profiles in _calc_monte
        # insolation-weighted soiling ratios in _calc_monte:
        self.monte_losses = []

        if pd.infer_freq(self.pm.index) != "D":
            raise ValueError("Daily performance metric series must have " "daily frequency")

        if pd.infer_freq(self.insolation_daily.index) != "D":
            raise ValueError("Daily insolation series must have " "daily frequency")

        if self.precipitation_daily is not None:
            if pd.infer_freq(self.precipitation_daily.index) != "D":
                raise ValueError("Precipitation series must have " "daily frequency")

    ###############################################################################
    # add neg_shift and piecewise into parameters/Matt
    def _calc_daily_df(
        self,
        day_scale=13,
        clean_threshold="infer",
        recenter=True,
        clean_criterion="shift",
        precip_threshold=0.01,
        outlier_factor=1.5,
        neg_shift=False,
        piecewise=False,
    ):
        """
        Calculates self.daily_df, a pandas dataframe prepared for SRR analysis,
        and self.renorm_factor, the renormalization factor for the daily
        performance

        Parameters
        ----------
        day_scale : int, default 13
            The number of days to use in rolling median for cleaning detection.
            An odd value is recommended.
        clean_threshold : float or 'infer', default 'infer'
            If float: the fractional positive shift in rolling median for
            cleaning detection.
            If 'infer': automatically use outliers in the shift as the
            threshold
        recenter : bool, default True
            Whether to recenter (renormalize) the daily performance to the
            median of the first year
        clean_criterion : str, {'shift', 'precip_and_shift', 'precip_or_shift', 'precip'} \
                default 'shift'
            The method of partitioning the dataset into soiling intervals.

            * 'precip_and_shift' - rolling median shifts must coincide
              with precipitation to be a valid cleaning event.
            * 'precip_or_shift' - rolling median shifts and precipitation
              events are each sufficient on their own to be a cleaning event.
            * 'shift', only rolling median shifts are treated as cleaning events.
            * 'precip', only precipitation events are treated as cleaning events.
        precip_threshold : float, default 0.01
            The daily precipitation threshold for defining precipitation
            cleaning events.
            Units must be consistent with ``self.precipitation_daily``.
        outlier_factor : float, default 1.5
            The factor used in the Tukey fence definition of outliers for flagging positive shifts
            in the rolling median used for cleaning detection. A smaller value will cause more and
            smaller shifts to be classified as cleaning events.
        neg_shift : bool, default True
            where True results in additional subdividing of soiling intervals
            when negative shifts are found in the rolling median of the performance
            metric. Inferred corrections in the soiling fit are made at these
            negative shifts. False results in no additional subdivides of the
            data where excessive negative shifts can invalidate a soiling interval.
        piecewise : bool, default True
            where True results in each soiling interval of sufficient length
            being tested for significant fit improvement with 2 piecewise linear
            fits. If the criteria of significance is met the soiling interval is
            subdivided into the 2 separate intervals. False results in no
            piecewise fit being tested.
        """
        if (day_scale % 2 == 0) and ("shift" in clean_criterion):
            warnings.warn(
                "An even value of day_scale was passed. An odd value is "
                "recommended, otherwise, consecutive days may be erroneously "
                "flagged as cleaning events. "
                "See https://github.com/NREL/rdtools/issues/189"
            )

        df = self.pm.to_frame()
        df.columns = ["pi"]
        df_insol = self.insolation_daily.to_frame()
        df_insol.columns = ["insol"]

        df = df.join(df_insol)
        precip = self.precipitation_daily

        if precip is not None:
            df_precip = precip.to_frame()
            df_precip.columns = ["precip"]
            df = df.join(df_precip)
        else:
            df["precip"] = 0

        # find first and last valid data point
        start = df[~df.pi.isnull()].index[0]
        end = df[~df.pi.isnull()].index[-1]
        df = df[start:end]

        # create a day count column
        df["day"] = range(len(df))

        # Recenter to median of first year, as in YoY degradation
        if recenter:
            oneyear = start + pd.Timedelta("364d")
            renorm = df.loc[start:oneyear, "pi"].median()
        else:
            renorm = 1

        df["pi_norm"] = df["pi"] / renorm

        # Find the beginning and ends of outages longer than dayscale
        # THIS CODE TRIGGERES DEPRECATION WARNING hance minor changes/Matt
        bfill = df["pi_norm"].bfill(limit=day_scale)
        ffill = df["pi_norm"].ffill(limit=day_scale)
        out_start = ~df["pi_norm"].isnull() & bfill.shift(-1).isnull()
        out_end = ~df["pi_norm"].isnull() & ffill.shift(1).isnull()

        # clean up the first and last elements
        out_start.iloc[-1] = False
        out_end.iloc[0] = False

        # Make a forward filled copy, just for use in
        # step, slope change detection
        # 1/6/24 Note several errors in soiling fit due to ffill for rolling
        # median change to day_scale/2 Matt
        df_ffill = df.copy()
        df_ffill = df.ffill(limit=int(round((day_scale / 2), 0)))

        # Calculate rolling median
        df["pi_roll_med"] = df_ffill.pi_norm.rolling(day_scale, center=True).median()

        # Detect steps in rolling median
        df["delta"] = df.pi_roll_med.diff()
        if clean_threshold == "infer":
            deltas = abs(df.delta)
            clean_threshold = deltas.quantile(0.75) + outlier_factor * (
                deltas.quantile(0.75) - deltas.quantile(0.25)
            )

        df["clean_event_detected"] = df.delta > clean_threshold

        ##########################################################################
        # Matt added these lines but the function "_collapse_cleaning_events"
        # was written by Asmund, it reduces multiple days of cleaning events
        # in a row to a single event

        reduced_cleaning_events = _collapse_cleaning_events(
            df.clean_event_detected, df.delta.values, 5
        )
        df["clean_event_detected"] = reduced_cleaning_events

        ##########################################################################
        precip_event = df["precip"] > precip_threshold

        if clean_criterion == "precip_and_shift":
            # Detect which cleaning events are associated with rain
            # within a 3 day window
            precip_event = (
                precip_event.rolling(3, center=True, min_periods=1).apply(any).astype(bool)
            )
            df["clean_event"] = df["clean_event_detected"] & precip_event
        elif clean_criterion == "precip_or_shift":
            df["clean_event"] = df["clean_event_detected"] | precip_event
        elif clean_criterion == "precip":
            df["clean_event"] = precip_event
        elif clean_criterion == "shift":
            df["clean_event"] = df["clean_event_detected"]
        else:
            raise ValueError(
                "clean_criterion must be one of "
                '{"precip_and_shift", "precip_or_shift", '
                '"precip", "shift"}'
            )

        df["clean_event"] = df.clean_event | out_start | out_end

        #######################################################################
        # add negative shifts which allows further segmentation of the soiling
        # intervals and handles correction for data outages/Matt
        df.delta = df.delta.fillna(0)  # to avoid NA corrupting calculation
        if neg_shift is True:
            df["drop_event"] = df.delta < -2.5 * clean_threshold
            df["break_event"] = df.clean_event | df.drop_event
        else:
            df["break_event"] = df.clean_event.copy()

        #######################################################################
        # This happens earlier than in the original code but is necessary
        # for adding piecewise breakpoints/Matt

        # Give an index to each soiling interval/run
        df["run"] = df.break_event.cumsum()
        df.index.name = "date"  # this gets used by name

        #######################################################################
        # df.fillna(0) /remove as the zeros introduced in pi_norm negatively
        # impact various fits in the code, I havent yet found the original purpose
        # or a failure due to removing/Matt

        #####################################################################
        # piecewise=True enables adding a single breakpoint per soiling intervals
        # if statistical criteria are met with the piecewise linear fit
        # compared to a single linear fit. Intervals <45 days reqire more
        # stringent statistical improvements/Matt
        if piecewise is True:
            warnings.warn(
                "Piecewise = True was passed, for both Piecewise=True"
                "and neg_shift=True cleaning_method choices should"
                "be perfect_clean_complex or inferred_clean_complex"
            )
            min_soil_length = 27  # min threshold of days necessary for piecewise fit
            piecewise_loop = sorted(list(set(df["run"])))
            cp_dates = []
            for r in piecewise_loop:
                run = df[df["run"] == r]
                pr = run.pi_norm.copy()
                pr = pr.ffill()  # linear fitting cant handle nans
                pr = pr.bfill()  # catch first position nan
                if len(run) > min_soil_length and run.pi_norm.sum() > 0:
                    sr, cp_date = segmented_soiling_period(pr, days_clean_vs_cp=13)
                    if cp_date is not None:
                        cp_dates.append(pr.index[cp_date])
            # save changes to df, note I would like to rename "clean_event" from
            # original code to something like "break_event
            df["slope_change_event"] = df.index.isin(cp_dates)
            df["break_event"] = df.break_event | df.slope_change_event
            df["run"] = df.break_event.cumsum()
        else:
            df["slope_change_event"] = False

        ######################################################################
        self.renorm_factor = renorm
        self.daily_df = df

    ######################################################################
    # added neg_shift into parameters in the following def/Matt
    def _calc_result_df(
        self,
        trim=False,
        max_relative_slope_error=500.0,
        max_negative_step=0.05,
        min_interval_length=7,
        neg_shift=False,
    ):
        """
        Calculates self.result_df, a pandas dataframe summarizing the soiling
        intervals identified and self.analyzed_daily_df, a version of
        self.daily_df with additional columns calculated during analysis.

        Parameters
        ----------
        trim : bool, default False
            whether to trim (remove) the first and last soiling intervals to
            avoid inclusion of partial intervals
        max_relative_slope_error : float, default 500
            the maximum relative size of the slope confidence interval for an
            interval to be considered valid (percentage).
        max_negative_step : float, default 0.05
            The maximum magnitude of negative discrete steps allowed in an
            interval for the interval to be considered valid (units of
            normalized performance metric).
        min_interval_length : int, default 7
            The minimum duration for an interval to be considered
            valid.  Cannot be less than 2 (days).
        neg_shift : bool, default True
            where True results in additional subdividing of soiling intervals
            when negative shifts are found in the rolling median of the performance
            metric. Inferred corrections in the soiling fit are made at these
            negative shifts. False results in no additional subdivides of the
            data where excessive negative shifts can invalidate a soiling interval.
        """

        daily_df = self.daily_df
        result_list = []
        if trim:
            # ignore first and last interval
            res_loop = sorted(list(set(daily_df["run"])))[1:-1]
        else:
            res_loop = sorted(list(set(daily_df["run"])))
        for r in res_loop:  # Matt added .iloc due to deprecation warning
            run = daily_df[daily_df["run"] == r]
            length = run.day.iloc[-1] - run.day.iloc[0]
            start_day = run.day.iloc[0]
            end_day = run.day.iloc[-1]
            start = run.index[0]
            end = run.index[-1]
            run_filtered = run[run.pi_norm > 0]
            # use the filtered version if it contains any points
            # otherwise use the unfiltered version to populate a
            # valid=False row
            if not run_filtered.empty:
                run = run_filtered
            ####################################################################
            # see commented changes
            result_dict = {
                "start": start,
                "end": end,
                "length": length,
                "run": r,
                "run_slope": 0,
                "run_slope_low": 0,
                "run_slope_high": 0,
                "max_neg_step": min(run.delta),
                "start_loss": 1,
                "inferred_start_loss": run.pi_norm.median(),  # changed from mean/Matt
                "inferred_end_loss": run.pi_norm.median(),  # changed from mean/Matt
                "slope_err": 10000,  # added high dummy start value for later logic/Matt
                "valid": False,
                "clean_event": run.clean_event.iloc[
                    0
                ],  # record of clean events to distiguisih from other breaks/Matt
                "run_loss_baseline": 0.0,  # loss from the polyfit over the soiling intercal/Matt
                ##############################################################
            }
            if len(run) > min_interval_length and run.pi_norm.sum() > 0:
                fit = theilslopes(run.pi_norm, run.day)
                fit_poly = np.poly1d(fit[0:2])
                result_dict["run_slope"] = fit[0]
                result_dict["run_slope_low"] = fit[2]
                result_dict["run_slope_high"] = min([0.0, fit[3]])
                result_dict["valid"] = True
                ########################################################
                # moved the following 2 line to the next section within conditional statement/Matt
                # result_dict['inferred_start_loss'] = fit_poly(start_day)
                # result_dict['inferred_end_loss'] = fit_poly(end_day)

                ####################################################
                # the following is moved here so median values are retained/Matt
                # for soiling inferrences when rejected fits occur
                result_dict["slope_err"] = (
                    result_dict["run_slope_high"] - result_dict["run_slope_low"]
                ) / abs(result_dict["run_slope"])

                if (result_dict["slope_err"] <= (max_relative_slope_error / 100.0)) & (
                    result_dict["run_slope"] < 0
                ):
                    result_dict["inferred_start_loss"] = fit_poly(start_day)
                    result_dict["inferred_end_loss"] = fit_poly(end_day)
                    #############################################
                    # calculate loss over soiling interval per polyfit/matt
                    result_dict["run_loss_baseline"] = (
                        result_dict["inferred_start_loss"] - result_dict["inferred_end_loss"]
                    )

                    ###############################################

            result_list.append(result_dict)

        results = pd.DataFrame(result_list)

        if results.empty:
            raise NoValidIntervalError("No valid soiling intervals were found")

        """
        # Filter results for each interval,
        # setting invalid interval to slope of 0
        #moved above to line 356/Matt
        results['slope_err'] = (
            results.run_slope_high - results.run_slope_low)\
            / abs(results.run_slope)
        """
        ###############################################################
        # negative shifts are now used as breaks for soiling intervals/Matt
        # so new criteria for final filter to modify dataframe
        if neg_shift is True:
            warnings.warn(
                "neg_shift = True was passed, for both Piecewise=True"
                "and neg_shift=True cleaning_method choices should"
                "be perfect_clean_complex or inferred_clean_complex"
            )
            filt = (
                (results.run_slope > 0)
                | (results.slope_err >= max_relative_slope_error / 100.0)
                # |(results.max_neg_step <= -1.0 * max_negative_step)
            )

            results.loc[filt, "run_slope"] = 0
            results.loc[filt, "run_slope_low"] = 0
            results.loc[filt, "run_slope_high"] = 0
            # only intervals that are now not valid are those that dont meet
            # the minimum inteval length or have no data
            # results.loc[filt, 'valid'] = False
        ##################################################################
        # original code below setting soiling intervals with extreme negative
        # shift to zero slopes, /Matt
        if neg_shift is False:
            filt = (
                (results.run_slope > 0)
                | (results.slope_err >= max_relative_slope_error / 100.0)
                | (results.max_neg_step <= -1.0 * max_negative_step)
                # remove line 389, want to store data for inferred values
                # for calculations below
                # |results.loc[filt, 'valid'] = False
            )
            results.loc[filt, "run_slope"] = 0
            results.loc[filt, "run_slope_low"] = 0
            results.loc[filt, "run_slope_high"] = 0

        # Calculate the next inferred start loss from next valid interval
        results["next_inferred_start_loss"] = np.clip(
            results[results.valid].inferred_start_loss.shift(-1), 0, 1
        )

        # Calculate the inferred recovery at the end of each interval
        ########################################################################
        # remove clipping on 'inferred_recovery' so absolute recovery can be
        # used in later step where clipping can be considered/Matt
        results["inferred_recovery"] = results.next_inferred_start_loss - results.inferred_end_loss

        ########################################################################
        # calculate beginning inferred shift (end of previous soiling period
        # to start of current period)/Matt
        results["prev_end"] = results[results.valid].inferred_end_loss.shift(1)
        # if the current interval starts with a clean event, the previous end
        # is a nan, and the current interval is valid then set prev_end=1
        results.loc[
            (results.clean_event is True) & (np.isnan(results.prev_end) & (results.valid is True)),
            "prev_end",
        ] = 1  # clean_event or clean_event_detected
        results["inferred_begin_shift"] = results.inferred_start_loss - results.prev_end
        # if orginal shift detection was positive the shift should not be
        # negative due to fitting results
        results.loc[results.clean_event, "inferred_begin_shift"] = np.clip(
            results.inferred_begin_shift, 0, 1
        )
        #######################################################################
        if neg_shift is False:
            results.loc[filt, "valid"] = False

        if len(results[results.valid]) == 0:
            raise NoValidIntervalError("No valid soiling intervals were found")
        new_start = results.start.iloc[0]
        new_end = results.end.iloc[-1]
        pm_frame_out = daily_df[new_start:new_end]
        pm_frame_out = (
            pm_frame_out.reset_index().merge(results, how="left", on="run").set_index("date")
        )

        pm_frame_out["loss_perfect_clean"] = np.nan
        pm_frame_out["loss_inferred_clean"] = np.nan
        pm_frame_out["days_since_clean"] = (pm_frame_out.index - pm_frame_out.start).dt.days

        #######################################################################
        # new code for perfect and inferred clean with handling of/Matt
        # negative shifts and changepoints within soiling intervals
        # goes to line 563
        #######################################################################
        pm_frame_out.inferred_begin_shift.bfill(inplace=True)
        pm_frame_out["forward_median"] = (
            pm_frame_out.pi.iloc[::-1].rolling(10, min_periods=5).median()
        )
        prev_shift = 1
        soil_inferred_clean = []
        soil_perfect_clean = []
        day_start = -1
        start_infer = 1
        start_perfect = 1
        soil_infer = 1
        soil_perfect = 1
        total_down = 0
        shift = 0
        shift_perfect = 0
        begin_perfect_shifts = [0]
        begin_infer_shifts = [0]

        for date, rs, d, start_shift, changepoint, forward_median in zip(
            pm_frame_out.index,
            pm_frame_out.run_slope,
            pm_frame_out.days_since_clean,
            pm_frame_out.inferred_begin_shift,
            pm_frame_out.slope_change_event,
            pm_frame_out.forward_median,
        ):
            new_soil = d - day_start
            day_start = d

            if new_soil <= 0:  # begin new soil period
                if (start_shift == prev_shift) | (changepoint):  # no shift at
                    # a slope changepoint
                    shift = 0
                    shift_perfect = 0
                else:
                    if (start_shift < 0) & (prev_shift < 0):  # (both negative) or
                        # downward shifts to start last 2 intervals
                        shift = 0
                        shift_perfect = 0
                        total_down = total_down + start_shift  # adding total downshifts
                        # to subtract from an eventual cleaning event
                    elif (start_shift > 0) & (prev_shift >= 0):  # (both positive) or
                        # cleanings start the last 2 intervals
                        shift = start_shift
                        shift_perfect = 1
                        total_down = 0
                    # add #####################3/27/24
                    elif (start_shift == 0) & (prev_shift >= 0):  # (
                        shift = start_shift
                        shift_perfect = start_shift
                        total_down = 0
                    #############################################################
                    elif (start_shift >= 0) & (prev_shift < 0):  # cleaning starts the current
                        # interval but there was a previous downshift
                        shift = start_shift + total_down  # correct for the negative shifts
                        shift_perfect = shift  # dont set to one 1 if correcting for a
                        # downshift (debateable alternative set to 1)
                        total_down = 0
                    elif (start_shift < 0) & (
                        prev_shift >= 0
                    ):  # negative shift starts the interval,
                        # previous shift was cleaning
                        shift = 0
                        shift_perfect = 0
                        total_down = start_shift
                # check that shifts results in being at or above the median of
                # the next 10 days of data
                # this catches places where start points of polyfits were
                # skewed below where data start
                if (soil_infer + shift) < forward_median:
                    shift = forward_median - soil_infer
                if (soil_perfect + shift_perfect) < forward_median:
                    shift_perfect = forward_median - soil_perfect

                # append the daily soiling ratio to each modeled fit
                begin_perfect_shifts.append(shift_perfect)
                begin_infer_shifts.append(shift)
                # clip to last value in case shift ends up negative
                soil_infer = np.clip((soil_infer + shift), soil_infer, 1)
                start_infer = soil_infer  # make next start value the last inferred value
                soil_inferred_clean.append(soil_infer)
                # clip to last value in case shift ends up negative
                soil_perfect = np.clip((soil_perfect + shift_perfect), soil_perfect, 1)
                start_perfect = soil_perfect
                soil_perfect_clean.append(soil_perfect)
                if changepoint is False:
                    prev_shift = start_shift  # assigned at new soil period

            elif new_soil > 0:  # within soiling period
                # append the daily soiling ratio to each modeled fit
                soil_infer = start_infer + rs * d
                soil_inferred_clean.append(soil_infer)

                soil_perfect = start_perfect + rs * d
                soil_perfect_clean.append(soil_perfect)

        pm_frame_out["loss_inferred_clean"] = pd.Series(
            soil_inferred_clean, index=pm_frame_out.index
        )
        pm_frame_out["loss_perfect_clean"] = pd.Series(
            soil_perfect_clean, index=pm_frame_out.index
        )

        results["begin_perfect_shift"] = pd.Series(begin_perfect_shifts)
        results["begin_infer_shift"] = pd.Series(begin_infer_shifts)
        #######################################################################
        self.result_df = results
        self.analyzed_daily_df = pm_frame_out

    def _calc_monte(self, monte, method="half_norm_clean"):
        """
        Runs the Monte Carlo step of the SRR method. Calculates
        self.random_profiles, a list of the random soiling profiles realized in
        the calculation, and self.monte_losses, a list of the
        insolation-weighted soiling ratios associated with the realizations.

        Parameters
        ----------
        monte : int
            number of Monte Carlo simulations to run
        method : str, {'half_norm_clean', 'random_clean', 'perfect_clean',
             perfect_clean_complex,inferred_clean_complex} \
            default 'half_norm_clean'

            How to treat the recovery of each cleaning event

            * 'random_clean' - a random recovery between 0-100%,
               pair with piecewise=False and neg_shift=False
            * 'perfect_clean' - each cleaning event returns the performance
              metric to 1,
              pair with piecewise=False and neg_shift=False
            * 'half_norm_clean' - The starting point of each interval is taken
              randomly from a half normal distribution with its
              mode (mu) at 1 and
              its sigma equal to 1/3 * (1-b) where b is the intercept of the fit to
              the interval,
              pair with piecewise=False and neg_shift=False
            *'perfect_clean_complex' - each detected clean event returns the
              performance metric to 1 while negative shifts in the data or
              piecewise linear fits result in no cleaning,
              pair with piecewise=True and neg_shift=True
            *'inferred_clean_complex' - at each detected clean event the
              performance metric increases based on fits to the data while
              negative shifts in the data or piecewise linear fits result in no
              cleaning,
              pair with piecewise=True and neg_shift=True
        """

        # Raise a warning if there is >20% invalid data
        if (
            (method == "half_norm_clean")
            or (method == "random_clean")
            or (method == "perfect_clean_complex")
            or (method == "inferred_clean_complex")
        ):
            valid_fraction = self.analyzed_daily_df["valid"].mean()
            if valid_fraction <= 0.8:
                warnings.warn('20% or more of the daily data is assigned to invalid soiling '
                    'intervals. This can be problematic with the "half_norm_clean" '
                    'and "random_clean" cleaning assumptions. Consider more permissive '
                    'validity criteria such as increasing "max_relative_slope_error" '
                    'and/or "max_negative_step" and/or decreasing "min_interval_length".'
                    ' Alternatively, consider using method="perfect_clean". For more'
                    ' info see https://github.com/NREL/rdtools/issues/272')
        monte_losses = []
        random_profiles = []
        for _ in range(monte):
            results_rand = self.result_df.copy()
            df_rand = self.analyzed_daily_df.copy()
            # only really need this column from the original frame:
            df_rand = df_rand[["insol", "run"]]
            results_rand["run_slope"] = np.random.uniform(
                results_rand.run_slope_low, results_rand.run_slope_high
            )
            results_rand["run_loss"] = results_rand.run_slope * results_rand.length

            results_rand["end_loss"] = np.nan
            results_rand["start_loss"] = np.nan

            # Make groups that start with a valid interval and contain
            # subsequent invalid intervals
            group_list = []
            group = 0
            for x in results_rand.valid:
                if x:
                    group += 1
                group_list.append(group)

            results_rand["group"] = group_list

            # randomize the extent of the cleaning
            inter_start = 1.0
            delta_previous_run_loss = 0
            start_list = []
            if (method == "half_norm_clean") or (method == "random_clean"):
                # Randomize recovery of valid intervals only
                valid_intervals = results_rand[results_rand.valid].copy()
                valid_intervals["inferred_recovery"] = np.clip(
                    valid_intervals.inferred_recovery, 0, 1
                )
                valid_intervals["inferred_recovery"] = valid_intervals.inferred_recovery.fillna(
                    1.0
                )

                end_list = []
                for i, row in valid_intervals.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    end_list.append(end)

                    if method == "half_norm_clean":
                        # Use a half normal with the inferred clean at the
                        # 3sigma point
                        x = np.clip(end + row.inferred_recovery, 0, 1)
                        inter_start = 1 - abs(np.random.normal(0.0, (1 - x) / 3))
                    elif method == "random_clean":
                        inter_start = np.random.uniform(end, 1)

                # Update the valid rows in results_rand
                valid_update = pd.DataFrame()
                valid_update["start_loss"] = start_list
                valid_update["end_loss"] = end_list
                valid_update.index = valid_intervals.index
                results_rand.update(valid_update)

                # forward and back fill to note the limits of random constant
                # derate for invalid intervals
                results_rand["previous_end"] = results_rand.end_loss.ffill()
                results_rand["next_start"] = results_rand.start_loss.bfill()

                # Randomly select random constant derate for invalid intervals
                # based on previous end and next beginning
                invalid_intervals = results_rand[~results_rand.valid].copy()
                # fill NaNs at beggining and end
                invalid_intervals.previous_end.fillna(1.0, inplace=True)
                invalid_intervals.next_start.fillna(1.0, inplace=True)
                groups = set(invalid_intervals.group)
                replace_levels = []

                if len(groups) > 0:
                    for g in groups:
                        rows = invalid_intervals[invalid_intervals.group == g]
                        n = len(rows)
                        low = rows.iloc[0].previous_end
                        high = rows.iloc[0].next_start
                        level = np.random.uniform(low, high)
                        replace_levels.append(np.full(n, level))

                    # Update results rand with the invalid rows
                    replace_levels = np.concatenate(replace_levels)
                    invalid_update = pd.DataFrame()
                    invalid_update["start_loss"] = replace_levels
                    invalid_update.index = invalid_intervals.index
                    results_rand.update(invalid_update)

            elif method == "perfect_clean":
                for i, row in results_rand.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    inter_start = 1
                results_rand["start_loss"] = start_list
            ##################################################################
            # matt additions

            elif method == "perfect_clean_complex":
                for i, row in results_rand.iterrows():
                    if row.begin_perfect_shift > 0:
                        inter_start = np.clip(
                            (inter_start + row.begin_perfect_shift + delta_previous_run_loss),
                            end,
                            1,
                        )
                        delta_previous_run_loss = -1 * row.run_loss - row.run_loss_baseline
                    else:
                        delta_previous_run_loss = (
                            delta_previous_run_loss - 1 * row.run_loss - row.run_loss_baseline
                        )
                        # inter_start=np.clip((inter_start+row.begin_shift+delta_previous_run_loss),0,1)
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss

                    inter_start = end
                results_rand["start_loss"] = start_list

            elif method == "inferred_clean_complex":
                for i, row in results_rand.iterrows():
                    if row.begin_infer_shift > 0:
                        inter_start = np.clip(
                            (inter_start + row.begin_infer_shift + delta_previous_run_loss),
                            end,
                            1,
                        )
                        delta_previous_run_loss = -1 * row.run_loss - row.run_loss_baseline
                    else:
                        delta_previous_run_loss = (
                            delta_previous_run_loss - 1 * row.run_loss - row.run_loss_baseline
                        )
                        # inter_start=np.clip((inter_start+row.begin_shift+delta_previous_run_loss),0,1)
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss

                    inter_start = end
                results_rand["start_loss"] = start_list
                """

                """
            ###############################################

            else:
                raise ValueError("Invalid method specification")

            df_rand = (
                df_rand.reset_index().merge(results_rand, how="left", on="run").set_index("date")
            )
            df_rand["loss"] = np.nan
            df_rand["days_since_clean"] = (df_rand.index - df_rand.start).dt.days
            df_rand["loss"] = df_rand.start_loss + df_rand.days_since_clean * df_rand.run_slope

            df_rand["soil_insol"] = df_rand.loss * df_rand.insol

            soiling_ratio = (
                df_rand.soil_insol.sum() / df_rand.insol[~df_rand.soil_insol.isnull()].sum()
            )
            monte_losses.append(soiling_ratio)
            random_profile = df_rand["loss"].copy()
            random_profile.name = "stochastic_soiling_profile"
            random_profiles.append(random_profile)

        self.random_profiles = random_profiles
        self.monte_losses = monte_losses

    #######################################################################
    # add neg_shift and piecewise to the following def/Matt
    def run(
        self,
        reps=1000,
        day_scale=13,
        clean_threshold="infer",
        trim=False,
        method="half_norm_clean",
        clean_criterion="shift",
        precip_threshold=0.01,
        min_interval_length=7,
        exceedance_prob=95.0,
        confidence_level=68.2,
        recenter=True,
        max_relative_slope_error=500.0,
        max_negative_step=0.05,
        outlier_factor=1.5,
        neg_shift=False,
        piecewise=False,
    ):
        """
        Run the SRR method from beginning to end.  Perform the stochastic rate
        and recovery soiling loss calculation. Based on the methods presented
        in Deceglie et al. "Quantifying Soiling Loss Directly From PV Yield"
        JPV 8(2) p547 2018.

        Parameters
        ----------
        reps : int, default 1000
            number of Monte Carlo realizations to calculate
        day_scale : int, default 13
            The number of days to use in rolling median for cleaning detection,
            and the maximum number of days of missing data to tolerate in a
            valid interval. An odd value is recommended.
        clean_threshold : float or 'infer', default 'infer'
            The fractional positive shift in rolling median for cleaning
            detection. Or specify 'infer' to automatically use outliers in the
            shift as the threshold.
        trim : bool, default False
            Whether to trim (remove) the first and last soiling intervals to
            avoid inclusion of partial intervals
        method : str, {'half_norm_clean', 'random_clean', 'perfect_clean',
             perfect_clean_complex,inferred_clean_complex} \
            default 'perfect_clean_complex'

            How to treat the recovery of each cleaning event

            * 'random_clean' - a random recovery between 0-100%,
               pair with piecewise=False and neg_shift=False
            * 'perfect_clean' - each cleaning event returns the performance
              metric to 1,
              pair with piecewise=False and neg_shift=False
            * 'half_norm_clean' - The starting point of each interval is taken
              randomly from a half normal distribution with its mode (mu) at 1 and
              its sigma equal to 1/3 * (1-b) where b is the intercept of the fit to
              the interval,
              pair with piecewise=False and neg_shift=False
            * 'perfect_clean_complex' - each detected clean event returns the
              performance metric to 1 while negative shifts in the data or
              piecewise linear fits result in no cleaning,
              pair with piecewise=True and neg_shift=True
            * 'inferred_clean_complex' - at each detected clean event the
              performance metric increases based on fits to the data while
              negative shifts in the data or piecewise linear fits result in no
              cleaning,
              pair with piecewise=True and neg_shift=True
        clean_criterion : str, {'shift', 'precip_and_shift', 'precip_or_shift', 'precip'} \
            default 'shift'
            The method of partitioning the dataset into soiling intervals

            * 'precip_and_shift' - rolling median shifts must coincide
              with precipitation to be a valid cleaning event.
            * 'precip_or_shift' - rolling median shifts and precipitation
              events are each sufficient on their own to be a cleaning event.
            * 'shift', only rolling median shifts are treated as cleaning events.
            * 'precip', only precipitation events are treated as cleaning events.
        precip_threshold : float, default 0.01
            The daily precipitation threshold for defining
            precipitation cleaning events.
            Units must be consistent with ``self.precipitation_daily``
        min_interval_length : int, default 7
            The minimum duration for an interval to be considered
            valid.  Cannot be less than 2 (days).
        exceedance_prob : float, default 95.0
            The probability level to use for exceedance value calculation in
            percent
        confidence_level : float, default 68.2
            The size of the confidence interval to return, in percent
        recenter : bool, default True
            Specify whether data is centered to normalized yield of 1 based on
            first year median
        max_relative_slope_error : float, default 500
            the maximum relative size of the slope confidence interval for an
            interval to be considered valid (percentage).
        max_negative_step : float, default 0.05
            The maximum magnitude of negative discrete steps allowed in an
            interval for the interval to be considered valid (units of
            normalized performance metric).
        outlier_factor : float, default 1.5
            The factor used in the Tukey fence definition of outliers for flagging positive shifts
            in the rolling median used for cleaning detection. A smaller value will cause more and
            smaller shifts to be classified as cleaning events.
        neg_shift : bool, default True
            where True results in additional subdividing of soiling intervals
            when negative shifts are found in the rolling median of the performance
            metric. Inferred corrections in the soiling fit are made at these
            negative shifts. False results in no additional subdivides of the
            data where excessive negative shifts can invalidate a soiling interval.
        piecewise : bool, default True
            where True results in each soiling interval of sufficient length
            being tested for significant fit improvement with 2 piecewise linear
            fits. If the criteria of significance is met the soiling interval is
            subdivided into the 2 separate intervals. False results in no
            piecewise fit being tested.

        Returns
        -------
        insolation_weighted_soiling_ratio : float
            P50 insolation-weighted soiling ratio based on stochastic rate and
            recovery analysis
        confidence_interval : numpy.array
            confidence interval (size specified by confidence_level) of
            insolation-weighted soiling ratio
        calc_info : dict
            * 'renormalizing_factor' - value used to recenter data
            * 'exceedance_level' - the insolation-weighted soiling ratio that
              was outperformed with probability of exceedance_prob
            * 'stochastic_soiling_profiles' - List of Pandas series
              corresponding to the Monte Carlo realizations of soiling ratio
              profiles
            * 'soiling_ratio_perfect_clean' - Pandas series of the soiling
              ratio during valid soiling intervals assuming perfect cleaning
              and P50 slopes
            * 'soiling_interval_summary' - Pandas dataframe summarizing the
              soiling intervals identified. The columns of the dataframe are
              as follows:

              +------------------------+----------------------------------------------+
              | Column Name            | Description                                  |
              +========================+==============================================+
              | 'start'                | Start timestamp of the soiling interval      |
              +------------------------+----------------------------------------------+
              | 'end'                  | End timestamp of the soiling interval        |
              +------------------------+----------------------------------------------+
              | 'soiling_rate'         | P50 Soiling rate for interval, in day^−1     |
              |                        | Negative value indicates soiling is          |
              |                        | occurring. E.g. a rate of −0.01 indicates 1% |
              |                        | soiling loss per day.                        |
              +------------------------+----------------------------------------------+
              | 'soiling_rate_low'     | Low edge of confidence interval for soiling  |
              |                        | rate for interval, in day^−1                 |
              +------------------------+----------------------------------------------+
              | 'soiling_rate_high'    | High edge of confidence interval for         |
              |                        | soiling rate for interval, in day^−1         |
              +------------------------+----------------------------------------------+
              | 'inferred_start_loss'  | Estimated performance metric at the start    |
              |                        | of the interval                              |
              +------------------------+----------------------------------------------+
              | 'inferred_end_loss'    | Estimated performance metric at the end      |
              |                        | of the interval                              |
              +------------------------+----------------------------------------------+
              | 'length'               | Number of days in the interval               |
              +------------------------+----------------------------------------------+
              | 'valid'                | Whether the interval meets the criteria to   |
              |                        | be treated as a valid soiling interval       |
              +------------------------+----------------------------------------------+

        """
        self._calc_daily_df(
            day_scale=day_scale,
            clean_threshold=clean_threshold,
            recenter=recenter,
            clean_criterion=clean_criterion,
            precip_threshold=precip_threshold,
            outlier_factor=outlier_factor,
            neg_shift=neg_shift,
            piecewise=piecewise,
        )
        self._calc_result_df(
            trim=trim,
            max_relative_slope_error=max_relative_slope_error,
            max_negative_step=max_negative_step,
            min_interval_length=min_interval_length,
            neg_shift=neg_shift,
        )
        self._calc_monte(reps, method=method)

        # Calculate the P50 and confidence interval
        half_ci = confidence_level / 2.0
        result = np.percentile(
            self.monte_losses,
            [50, 50.0 - half_ci, 50.0 + half_ci, 100 - exceedance_prob],
        )
        P_level = result[3]

        # Construct calc_info output
        ###############################################
        # add inferred_recovery, inferred_begin_shift /Matt
        ###############################################
        intervals_out = self.result_df[
            [
                "start",
                "end",
                "run_slope",
                "run_slope_low",
                "run_slope_high",
                "inferred_start_loss",
                "inferred_end_loss",
                "inferred_recovery",
                "inferred_begin_shift",
                "length",
                "valid",
            ]
        ].copy()
        intervals_out.rename(
            columns={
                "run_slope": "soiling_rate",
                "run_slope_high": "soiling_rate_high",
                "run_slope_low": "soiling_rate_low",
            },
            inplace=True,
        )

        df_d = self.analyzed_daily_df
        # sr_perfect = df_d[df_d['valid']]['loss_perfect_clean']
        sr_perfect = df_d.loss_perfect_clean

        calc_info = {
            "exceedance_level": P_level,
            "renormalizing_factor": self.renorm_factor,
            "stochastic_soiling_profiles": self.random_profiles,
            "soiling_interval_summary": intervals_out,
            "soiling_ratio_perfect_clean": sr_perfect,
        }

        return (result[0], result[1:3], calc_info)


# more updates are needed for documentation but added additional inputs
# that are in srr.run /Matt
def soiling_srr(
    energy_normalized_daily,
    insolation_daily,
    reps=1000,
    precipitation_daily=None,
    day_scale=13,
    clean_threshold="infer",
    trim=False,
    method="half_norm_clean",
    clean_criterion="shift",
    precip_threshold=0.01,
    min_interval_length=7,
    exceedance_prob=95.0,
    confidence_level=68.2,
    recenter=True,
    max_relative_slope_error=500.0,
    max_negative_step=0.05,
    outlier_factor=1.5,
    neg_shift=False,
    piecewise=False,
):
    """
    Functional wrapper for :py:class:`~rdtools.soiling.SRRAnalysis`. Perform
    the stochastic rate and recovery soiling loss calculation. Based on the
    methods presented in Deceglie et al. JPV 8(2) p547 2018.

    Parameters
    ----------
    energy_normalized_daily : pandas.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Alternatively, the soiling ratio output of a soiling sensor (e.g. the
        photocurrent ratio between matched dirty and clean PV reference cells).
        In either case, data should be insolation-weighted daily aggregates.
    insolation_daily : pandas.Series
        Daily plane-of-array insolation corresponding to
        `energy_normalized_daily`. Arbitrary units.
    reps : int, default 1000
        number of Monte Carlo realizations to calculate
    precipitation_daily : pandas.Series, default None
        Daily total precipitation. Units ambiguous but should be the same as
        precip_threshold. Note default behavior of precip_threshold. (Ignored
        if ``clean_criterion='shift'``.)
    day_scale : int, default 13
        The number of days to use in rolling median for cleaning detection,
        and the maximum number of days of missing data to tolerate in a valid
        interval. An odd value is recommended.
    clean_threshold : float or 'infer', default 'infer'
        The fractional positive shift in rolling median for cleaning detection.
        Or specify 'infer' to automatically use outliers in the shift as the
        threshold.
    trim : bool, default False
        Whether to trim (remove) the first and last soiling intervals to avoid
        inclusion of partial intervals
    method : str, {'half_norm_clean', 'random_clean', 'perfect_clean',
         perfect_clean_complex,inferred_clean_complex} \
        default 'half_norm_clean'

        How to treat the recovery of each cleaning event

        * 'random_clean' - a random recovery between 0-100%,
           pair with piecewise=False and neg_shift=False
        * 'perfect_clean' - each cleaning event returns the performance
          metric to 1,
          pair with piecewise=False and neg_shift=False
        * 'half_norm_clean' - The starting point of each interval is taken
          randomly from a half normal distribution with its mode (mu) at 1 and
          its sigma equal to 1/3 * (1-b) where b is the intercept of the fit to
          the interval,
          pair with piecewise=False and neg_shift=False
        *'perfect_clean_complex' - each detected clean event returns the
          performance metric to 1 while negative shifts in the data or
          piecewise linear fits result in no cleaning,
          pair with piecewise=True and neg_shift=True
        *'inferred_clean_complex' - at each detected clean event the
          performance metric increases based on fits to the data while
          negative shifts in the data or piecewise linear fits result in no
          cleaning,
          pair with piecewise=True and neg_shift=True
    clean_criterion : str, {'shift', 'precip_and_shift', 'precip_or_shift', 'precip'} \
        default 'shift'
        The method of partitioning the dataset into soiling intervals

        * 'precip_and_shift' - rolling median shifts must coincide
          with precipitation to be a valid cleaning event.
        * 'precip_or_shift' - rolling median shifts and precipitation
          events are each sufficient on their own to be a cleaning event.
        * 'shift', only rolling median shifts are treated as cleaning events.
        * 'precip', only precipitation events are treated as cleaning events.
    precip_threshold : float, default 0.01
        The daily precipitation threshold for defining precipitation
        cleaning events. Units must be consistent with precip.
    min_interval_length : int, default 7
        The minimum duration, in days, for an interval to be considered
        valid.  Cannot be less than 2 (days).
    exceedance_prob : float, default 95.0
        the probability level to use for exceedance value calculation in
        percent
    confidence_level : float, default 68.2
        the size of the confidence interval to return, in percent
    recenter : bool, default True
        specify whether data is centered to normalized yield of 1 based on
        first year median
    max_relative_slope_error : float, default 500.0
        the maximum relative size of the slope confidence interval for an
        interval to be considered valid (percentage).
    max_negative_step : float, default 0.05
        The maximum magnitude of negative discrete steps allowed in an interval
        for the interval to be considered valid (units of normalized
        performance metric).
    outlier_factor : float, default 1.5
        The factor used in the Tukey fence definition of outliers for flagging positive shifts
        in the rolling median used for cleaning detection. A smaller value will cause more and
        smaller shifts to be classified as cleaning events.
    neg_shift : bool, default True
        where True results in additional subdividing of soiling intervals
        when negative shifts are found in the rolling median of the performance
        metric. Inferred corrections in the soiling fit are made at these
        negative shifts. False results in no additional subdivides of the
        data where excessive negative shifts can invalidate a soiling interval.
    piecewise : bool, default True
        where True results in each soiling interval of sufficient length
        being tested for significant fit improvement with 2 piecewise linear
        fits. If the criteria of significance is met the soiling interval is
        subdivided into the 2 separate intervals. False results in no
        piecewise fit being tested.

    Returns
    -------
    insolation_weighted_soiling_ratio : float
        P50 insolation weighted soiling ratio based on stochastic rate and
        recovery analysis
    confidence_interval : numpy.array
        confidence interval (size specified by ``confidence_level``) of
        degradation rate estimate
    calc_info : dict
        * 'renormalizing_factor' - value used to recenter data
        * 'exceedance_level' - the insolation-weighted soiling ratio that
          was outperformed with probability of exceedance_prob
        * 'stochastic_soiling_profiles' - List of Pandas series
          corresponding to the Monte Carlo realizations of soiling ratio
          profiles
        * 'soiling_ratio_perfect_clean' - Pandas series of the soiling
          ratio during valid soiling intervals assuming perfect cleaning
          and P50 slopes
        * 'soiling_interval_summary' - Pandas dataframe summarizing the
          soiling intervals identified. The columns of the dataframe are
          as follows:

          +------------------------+----------------------------------------------+
          | Column Name            | Description                                  |
          +========================+==============================================+
          | 'start'                | Start timestamp of the soiling interval      |
          +------------------------+----------------------------------------------+
          | 'end'                  | End timestamp of the soiling interval        |
          +------------------------+----------------------------------------------+
          | 'soiling_rate'         | P50 Soiling rate for interval, in day^−1     |
          |                        | Negative value indicates soiling is          |
          |                        | occurring. E.g. a rate of −0.01 indicates 1% |
          |                        | soiling loss per day.                        |
          +------------------------+----------------------------------------------+
          | 'soiling_rate_low'     | Low edge of confidence interval for soiling  |
          |                        | rate for interval, in day^−1                 |
          +------------------------+----------------------------------------------+
          | 'soiling_rate_high'    | High edge of confidence interval for         |
          |                        | soiling rate for interval, in day^−1         |
          +------------------------+----------------------------------------------+
          | 'inferred_start_loss'  | Estimated performance metric at the start    |
          |                        | of the interval                              |
          +------------------------+----------------------------------------------+
          | 'inferred_end_loss'    | Estimated performance metric at the end      |
          |                        | of the interval                              |
          +------------------------+----------------------------------------------+
          | 'length'               | Number of days in the interval               |
          +------------------------+----------------------------------------------+
          | 'valid'                | Whether the interval meets the criteria to   |
          |                        | be treated as a valid soiling interval       |
          +------------------------+----------------------------------------------+
    """

    srr = SRRAnalysis(
        energy_normalized_daily,
        insolation_daily,
        precipitation_daily=precipitation_daily,
    )

    sr, sr_ci, soiling_info = srr.run(
        reps=reps,
        day_scale=day_scale,
        clean_threshold=clean_threshold,
        trim=trim,
        method=method,
        clean_criterion=clean_criterion,
        precip_threshold=precip_threshold,
        min_interval_length=min_interval_length,
        exceedance_prob=exceedance_prob,
        confidence_level=confidence_level,
        recenter=recenter,
        max_relative_slope_error=max_relative_slope_error,
        max_negative_step=max_negative_step,
        outlier_factor=outlier_factor,
        neg_shift=neg_shift,
        piecewise=piecewise,
    )

    return sr, sr_ci, soiling_info


def _count_month_days(start, end):
    """Return a dict of number of days between start and end
    (inclusive) in each month"""
    days = pd.date_range(start, end)
    months = [x.month for x in days]
    out_dict = {}
    for month in range(1, 13):
        out_dict[month] = months.count(month)

    return out_dict


def annual_soiling_ratios(stochastic_soiling_profiles, insolation_daily, confidence_level=68.2):
    """
    Return annualized soiling ratios and associated confidence intervals based
    on stochastic soiling profiles from SRR. Note that each year
    may be affected by previous years' profiles for all SRR cleaning
    assumptions (i.e. method) except perfect_clean.

    Parameters
    ----------
    stochastic_soiling_profiles : list
        List of pd.Series representing profile realizations from the SRR monte carlo.
        Typically ``soiling_interval_summary['stochastic_soiling_profiles']`` obtained with
        :py:func:`rdtools.soiling.soiling_srr` or :py:meth:`rdtools.soiling.SRRAnalysis.run`
    insolation_daily : pandas.Series
        Daily plane-of-array insolation with DatetimeIndex. Arbitrary units.
    confidence_level : float, default 68.2
        The size of the confidence interval to use in determining the
        upper and lower quantiles reported in the returned DataFrame.
        (The median is always included in the result.)

    Returns
    -------
    pandas.DataFrame
        DataFrame describing annual soiling rates.

        +------------------------+-------------------------------------------+
        | Column Name            | Description                               |
        +========================+===========================================+
        | 'year'                 | Calendar year                             |
        +------------------------+-------------------------------------------+
        | 'soiling_ratio_median' | The median insolation-weighted soiling    |
        |                        | ratio for the year                        |
        +------------------------+-------------------------------------------+
        | 'soiling_ratio_low'    | The lower edge of the confidence interval |
        |                        | for insolation-weighted soiling ratio for |
        |                        | the year                                  |
        +------------------------+-------------------------------------------+
        | 'soiling_ratio_high'   | The upper edge of the confidence interval |
        |                        | for insolation-weighted soiling ratio for |
        |                        | the year                                  |
        +------------------------+-------------------------------------------+
    """

    # Create a df with each realization as a column
    all_profiles = pd.concat(stochastic_soiling_profiles, axis=1)
    all_profiles = all_profiles.dropna()

    if not all_profiles.index.isin(insolation_daily.index).all():
        warnings.warn(
            "The indexes of stochastic_soiling_profiles are not entirely "
            "contained within the index of insolation_daily. Every day in "
            "stochastic_soiling_profiles should be represented in "
            "insolation_daily. This may cause erroneous results."
        )

    insolation_daily = insolation_daily.reindex(all_profiles.index)

    # Weight each day by insolation
    all_profiles_weighted = all_profiles.multiply(insolation_daily, axis=0)

    # Compute the insolation-weighted soiling ratio (IWSR) for each realization
    annual_insolation = insolation_daily.groupby(insolation_daily.index.year).sum()
    all_annual_weighted_sums = all_profiles_weighted.groupby(
        all_profiles_weighted.index.year
    ).sum()
    all_annual_iwsr = all_annual_weighted_sums.multiply(1 / annual_insolation, axis=0)

    annual_soiling = pd.DataFrame(
        {
            "soiling_ratio_median": all_annual_iwsr.quantile(0.5, axis=1),
            "soiling_ratio_low": all_annual_iwsr.quantile(
                0.5 - confidence_level / 2 / 100, axis=1
            ),
            "soiling_ratio_high": all_annual_iwsr.quantile(
                0.5 + confidence_level / 2 / 100, axis=1
            ),
        }
    )
    annual_soiling.index.name = "year"
    annual_soiling = annual_soiling.reset_index()

    return annual_soiling


def monthly_soiling_rates(
    soiling_interval_summary,
    min_interval_length=14,
    max_relative_slope_error=500.0,
    reps=100000,
    confidence_level=68.2,
):
    """
    Use Monte Carlo to calculate typical monthly soiling rates.
    Samples possible soiling rates from soiling rate confidence
    intervals associated with soiling intervals assuming a uniform
    distribution. Soiling intervals get samples proportionally
    to their overlap with each calendar month.

    Parameters
    ----------
    soiling_interval_summary : pandas.DataFrame
        DataFrame describing soiling intervals. Typically from
        ``soiling_info['soiling_interval_summary']`` obtained with
        :py:func:`rdtools.soiling.soiling_srr` or
        :py:meth:`rdtools.soiling.SRRAnalysis.run` Must have columns
        ``soiling_rate_high``, ``soiling_rate_low``, ``soiling_rate``,
        ``length``, ``valid``,``start``, and ``end``.

    min_interval_length : int, default 14
        The minimum number of days a soiling interval must contain to be
        included in the calculation. Similar to the same parameter in
        :py:func:`soiling_srr` and :py:meth:`SRRAnalysis.run` but with a
        more conservative default value as a
        starting point for monthly soiling rate analyses.

    max_relative_slope_error : float, default 500.0
        The maximum relative size of the slope confidence interval for an
        interval to be included in the calculation (percentage).

    reps : int, default 100000
        The number of Monte Carlo samples to take for each month.

    confidence_level : float, default 68.2
        The size of the confidence interval, as a percentage, to use
        in determining the upper and lower quantiles reported in the
        returned DataFrame. (The median is always included in the result.)

    Returns
    -------
    pandas.DataFrame
        DataFrame describing monthly soiling rates.

        +-----------------------+--------------------------------------------------+
        | Column Name           | Description                                      |
        +=======================+==================================================+
        | 'month'               | Integer month, January (1) to December (12)      |
        +-----------------------+--------------------------------------------------+
        | 'soiling_rate_median' | The median soiling rate for the month over       |
        |                       | the entire dataset, in units of day^−1.          |
        |                       | Negative value indicates soiling is occurring.   |
        |                       | E.g. a rate of −0.01 indicates 1% soiling loss   |
        |                       | per day.                                         |
        +-----------------------+--------------------------------------------------+
        | 'soiling_rate_low'    | The lower edge of the confidence interval        |
        |                       | for the monthly soiling rate in units of         |
        |                       | day^−1                                           |
        +-----------------------+--------------------------------------------------+
        | 'soiling_rate_high'   | The upper edge of the confidence interval        |
        |                       | for the monthly soiling rate in units of         |
        |                       | day^−1                                           |
        +-----------------------+--------------------------------------------------+
        | 'interval_count'      | The number of soiling intervals contributing     |
        |                       | to the monthly calculation. If only a few        |
        |                       | intervals contribute, the confidence interval    |
        |                       | is likely to underestimate the true uncertainty. |
        +-----------------------+--------------------------------------------------+
    """

    # filter to intervals of interest
    high = soiling_interval_summary["soiling_rate_high"]
    low = soiling_interval_summary["soiling_rate_low"]
    rate = soiling_interval_summary["soiling_rate"]
    rel_error = 100 * abs((high - low) / rate)
    intervals = soiling_interval_summary[
        (soiling_interval_summary["length"] >= min_interval_length)
        & (soiling_interval_summary["valid"])
        & (rel_error <= max_relative_slope_error)
    ].copy()

    # count the overlap of each interval with each month
    month_counts = []
    for _, row in intervals.iterrows():
        month_counts.append(_count_month_days(row["start"], row["end"]))

    # divy up the monte carlo reps based on overlap
    for month in range(1, 13):
        days_in_month = np.array([x[month] for x in month_counts])
        sample_col = f"samples_for_month_{month}"
        if days_in_month.sum() > 0:
            intervals[sample_col] = np.ceil(days_in_month / days_in_month.sum() * reps)
        else:
            intervals[sample_col] = 0
        intervals[sample_col] = intervals[sample_col].astype(int)

    # perform the monte carlo month by month
    ci_quantiles = [0.5 - confidence_level / 2 / 100, 0.5 + confidence_level / 2 / 100]
    monthly_rate_data = []
    relevant_interval_count = []
    for month in range(1, 13):
        rates = []
        sample_col = f"samples_for_month_{month}"

        relevant_intervals = intervals[intervals[sample_col] > 0]
        for _, row in relevant_intervals.iterrows():
            rates.append(
                np.random.uniform(
                    row["soiling_rate_low"], row["soiling_rate_high"], row[sample_col]
                )
            )

        rates = [x for sublist in rates for x in sublist]

        if rates:
            monthly_rate_data.append(np.quantile(rates, [0.5, ci_quantiles[0], ci_quantiles[1]]))
        else:
            monthly_rate_data.append(np.array([np.nan] * 3))

        relevant_interval_count.append(len(relevant_intervals))
    monthly_rate_data = np.array(monthly_rate_data)

    # make a dataframe out of the results
    monthly_soiling_df = pd.DataFrame(
        data=monthly_rate_data,
        columns=["soiling_rate_median", "soiling_rate_low", "soiling_rate_high"],
    )
    monthly_soiling_df.insert(0, "month", range(1, 13))
    monthly_soiling_df["interval_count"] = relevant_interval_count

    return monthly_soiling_df


class CODSAnalysis:
    """
    Container for the Combined Degradation and Soiling (CODS) algorithm
    for degradation and soiling loss analysis. Based on the
    method presented in [1]_.

    Parameters
    ----------
    energy_normalized_daily : pandas.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Index must be DatetimeIndex with daily frequency

    Attributes
    ----------
    pm : pandas.Series
        Equals `energy_normalized_daily`
    result_df : pandas.DataFrame with pandas datetimeindex
        Contains the columns/keys:

        +------------------------+----------------------------------------------+
        | Column Name            | Description                                  |
        +========================+==============================================+
        | 'soiling_ratio'        | soiling ratio (SR) (-)                       |
        +------------------------+----------------------------------------------+
        | 'soiling_rates'        | soiling rates (1/day)                        |
        +------------------------+----------------------------------------------+
        | 'cleaning_events'      | True at cleaning events                      |
        +------------------------+----------------------------------------------+
        | 'seasonal_component'   | seasonal component (SC)                      |
        +------------------------+----------------------------------------------+
        | 'degradation_trend'    | degradation trend (Rd)                       |
        +------------------------+----------------------------------------------+
        | 'total_model'          | the total model fit, i.e. SR * SC * Rd * rs, |
        |                        | where SR is the soiling ratio, SC is the     |
        |                        | seasonal component, Rd is the degradation    |
        |                        | trend, and rs is the residual shift, i.e.    |
        |                        | the mean of the residuals (adjusting the     |
        |                        | position of the model fit to the position of |
        |                        | the input data)                              |
        +------------------------+----------------------------------------------+
        | 'residuals'            | The residuals of the model fit, i.e.         |
        |                        | PI / (SR * SC * Rd)                          |
        +------------------------+----------------------------------------------+
        | 'SR_low'               | lower bound of 95 % conf. interval of SR     |
        +------------------------+----------------------------------------------+
        | 'SR_high'              | upper bound of 95 % conf. interval of SR     |
        +------------------------+----------------------------------------------+
        | 'rates_low'            | lower bound of 95 % conf. interval of        |
        |                        | soiling rates                                |
        +------------------------+----------------------------------------------+
        | 'rates_high'           | upper bound of 95 % conf. interval of        |
        |                        | soiling rates                                |
        +------------------------+----------------------------------------------+
        | 'bt_soiling_ratio'     | Bootstrapped estimate of soiling ratio (SR)  |
        +------------------------+----------------------------------------------+
        | 'bt_soiling_rates'     | Bootstrapped estimate of soiling rates       |
        +------------------------+----------------------------------------------+
        | 'seasonal_low'         | lower bound of 95 % conf. interval of        |
        |                        | seasonal component (SC)                      |
        +------------------------+----------------------------------------------+
        | 'seasonal_high'        | upper bound of 95 % conf. interval of        |
        |                        | seasonal component (SC)                      |
        +------------------------+----------------------------------------------+
        | 'model_high'           | upper bound of 95 % confidence interval of   |
        |                        | the model fit                                |
        +------------------------+----------------------------------------------+
        | 'model_low'            | lower bound of 95 % confidence interval of   |
        |                        | the model fit                                |
        +------------------------+----------------------------------------------+

    degradation : list
        List of linear degradation rate of system in %/year, lower and
        upper bound of 95% confidence interval
    soiling_loss : list
        List of average soiling losses over the time series in %, lower and
        upper bound of 95% confidence interval
    residual_shift : float
        Mean value of residuals. Multiply total model by this number for
        complete overlap with input pi
    RMSE : float
        Root Means Squared Error of total model vs input pi
    small_soiling_signal : bool
        Whether or not the signal is deemed too small to infer anything
        about it
    adf_res : list
        The results of an Augmented Dickey-Fuller test (telling whether the
        residuals are stationary or not)

    Raises
    ------
    ValueError
        If the performance metrix does not have daily index frequency

    References
    ----------
    .. [1] Skomedal, Å. and Deceglie, M. G., IEEE Journal of
       Photovoltaics, Sept. 2020. https://doi.org/10.1109/JPHOTOV.2020.3018219
    """

    def __init__(self, energy_normalized_daily):
        self.pm = energy_normalized_daily  # daily performance metric

        if np.isnan(self.pm.iloc[0]):
            first_keeper = self.pm.isna().idxmin()
            self.pm = self.pm.loc[first_keeper:]

        if self.pm.index.freq != "D":
            raise ValueError(
                "Daily performance metric series must have "
                "daily frequency (missing dates should be "
                "represented by NaNs)"
            )

    def iterative_signal_decomposition(
        self,
        order=("SR", "SC", "Rd"),
        degradation_method="YoY",
        max_iterations=18,
        cleaning_sensitivity=0.5,
        convergence_criterion=5e-3,
        pruning_iterations=1,
        clean_pruning_sensitivity=0.6,
        soiling_significance=0.75,
        process_noise=1e-4,
        renormalize_SR=None,
        ffill=True,
        clip_soiling=True,
        verbose=False,
    ):
        """
        Estimates the soiling losses and the degradation rate of a PV system
        based on its daily normalized energy, or daily Performance Index (PI).
        The underlying assumption is that the PI
        consists of a degradation trend, a seasonal component, and a soiling
        signal (defined as 1 if no soiling, decreasing with increasing soiling
        losses). I.e.: PI = degradation_trend * seasonal_component * soiling_ratio *
        residuals, or:

        .. math::

            PI = Rd * SC * SR * R

        The function has a heuristic for detecting whether the soiling signal is
        significant enough for soiling loss inference, which is based on the
        ratio between the spread in the soiling signal versus the spread in the
        residuals (defined by the 2.5th and 97.5th percentiles)

        * The degradation trend is obtained using the native RdTools Year-On-Year
          method [1]_.
        * The seasonal component is derived with statsmodels STL [2]_.
        * The soiling signal is derived with a Kalman Filter with a cleaning
          detection heuristic [3]_.

        Parameters
        ----------
        order : tuple, default ('SR', 'SC', 'Rd')
            Tuple containing 1 to 3 of the following strings 'SR' (soiling
            ratio), 'SC' (seasonal component), 'Rd' (degradation component),
            defining the order in which these components will be found during
            iterative decomposition
        degradation_method : string, default 'YoY'
            Either 'YoY' or 'STL'. If anything else, 'YoY' will be assumed.
            Decides whether to use the YoY method [3] for estimating the
            degradation trend (assumes linear trend), or the STL-method (does
            not assume linear trend). The latter is slower.
        max_iterations : int, default 18
            Max number of iterations to perform. Each iteration fits only one of the
            components, so three iterations are needed to fit all three components.
        cleaning_sensitivity : float, default .5
            Higher value gives lower cleaning event detection sensitivity.
            Should be between 0.1 and 2
        convergence_criterion : float, default 5e-3
            the relative change in the convergence metric required for
            convergence
        pruning_iterations : int, default 1
            Number of iterations when pruning (removing) cleaning events
        clean_pruning_sensitivity : float, default .6
            Sensitivity tuner that decides how easily a cleaning event is pruned
            (removed). Larger values means a smaller chance of pruning a given event.
            Should be between 0.1 and 2
        soiling_significance : float, default 0.75
            The minimum amplitude of the soiling signal relative to the amplitude of
            the residuals that is considered a significant soiling signal.
        process_noise : float, default 1e-4
            A Kalman Filter parameter that represents the expected amount of unmodeled
            variation in the process, the process being the variation in the
            performance index that is due to soiling, seasonality and degradation.
        renormalize_SR : float, default None
            If not none, defines the percentile for which the SR will be
            normalized to, based on the SR just after cleaning events
        ffill : bool, default True
            Whether to use forward fill (default) or backward fill before
            doing the rolling median for cleaning event detection
        clip_soiling : bool, default True
            Whether or not to clip the soiling ratio at max 1 and minimum 0.
        verbose : bool, default False
            If true, prints a progress report

        Returns
        -------
        df_out : pandas.DataFrame
            Dataframe that summarized the results of the iterative signal decomposition.
            Contains the followig columns:

            +------------------------+----------------------------------------------+
            | Column Name            | Description                                  |
            +========================+==============================================+
            | 'soiling_ratio'        | soiling ratio (SR) (-)                       |
            +------------------------+----------------------------------------------+
            | 'soiling_rates'        | soiling rates (1/day)                        |
            +------------------------+----------------------------------------------+
            | 'cleaning_events'      | True at cleaning events                      |
            +------------------------+----------------------------------------------+
            | 'seasonal_component'   | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'degradation_trend'    | degradation trend (Rd)                       |
            +------------------------+----------------------------------------------+
            | 'total_model'          | the total model fit, i.e. SR * SC * Rd * rs, |
            |                        | where SR is the soiling ratio, SC is the     |
            |                        | seasonal component, Rd is the degradation    |
            |                        | trend, and rs is the residual shift, i.e.    |
            |                        | the mean of the residuals (adjusting the     |
            |                        | position of the model fit to the position of |
            |                        | the input data)                              |
            +------------------------+----------------------------------------------+
            | 'residuals'            | The residuals of the model fit, i.e.         |
            |                        | PI / (SR * SC * Rd)                          |
            +------------------------+----------------------------------------------+

        results_dict: dict
            Dictionary with the following entries:

            +------------------------+----------------------------------------------+
            | Key                    | Description                                  |
            +========================+==============================================+
            | 'degradation'          | Linear degradation rate of system in %/year  |
            |                        | (float)                                      |
            +------------------------+----------------------------------------------+
            | 'soiling_loss'         | Average soiling losses over the time series  |
            |                        | in % (float)                                 |
            +------------------------+----------------------------------------------+
            | 'residual_shift'       | Mean value of residuals. Multiply total      |
            |                        | model by this number for complete overlap    |
            |                        | with input pi (float)                        |
            +------------------------+----------------------------------------------+
            | 'RMSE'                 | Root Means Squared Error of total model vs   |
            |                        | input pi (float)                             |
            +------------------------+----------------------------------------------+
            | 'small_soiling_signal' | Whether or not the signal is deemed too      |
            |                        | small to infer soiling ratio (bool)          |
            +------------------------+----------------------------------------------+
            | 'adf_res'              | The results of an Augmented Dickey-Fuller    |
            |                        | test (telling whether the residuals are      |
            |                        | stationary or not) (list)                    |
            +------------------------+----------------------------------------------+

        References
        ----------
        .. [1] Jordan, D.C., Deline, C., Kurtz, S.R., Kimball, G.M., Anderson, M.,
           2017. Robust PV Degradation Methodology and Application. IEEE J.
           Photovoltaics 1–7. https://doi.org/10.1109/JPHOTOV.2017.2779779

        .. [2] Deceglie, M.G., Micheli, L., Muller, M., 2018. Quantifying Soiling
           Loss Directly from PV Yield. IEEE J. Photovoltaics 8, 547–551.
           https://doi.org/10.1109/JPHOTOV.2017.2784682

        .. [3] Skomedal, Å. and Deceglie, M. G., IEEE Journal of
           Photovoltaics, Sept. 2020. https://doi.org/10.1109/JPHOTOV.2020.3018219
        """
        pi = self.pm.copy()
        if degradation_method == "STL" and "Rd" in order:
            order = tuple([c for c in order if c != "Rd"])

        if "SR" not in order:
            raise ValueError(
                "'SR' must be in argument 'order' " + "(e.g. order=['SR', 'SC', 'Rd']"
            )
        n_steps = len(order)
        day = np.arange(len(pi))
        degradation_trend = [1]
        seasonal_component = [1]
        soiling_ratio = [1]
        soiling_dfs = []
        yoy_save = [0]
        residuals = pi.copy()
        residual_shift = 1
        convergence_metric = [_RMSE(pi, np.ones((len(pi),)))]

        # Find possible cleaning events based on the performance index
        ce, rm9 = _rolling_median_ce_detection(
            pi.index, pi, ffill=ffill, tuner=cleaning_sensitivity
        )
        pce = _collapse_cleaning_events(ce, rm9.diff().values, 5)

        small_soiling_signal, perfect_cleaning = False, True
        ic = 0  # iteration counter

        if verbose:
            print("It. nr\tstep\tRMSE\ttimer")
        if verbose:
            print("{:}\t- \t{:.5f}".format(ic, convergence_metric[ic]))
        while ic < max_iterations:
            t0 = time.time()
            ic += 1

            # Find soiling component
            if order[(ic - 1) % n_steps] == "SR":
                if ic > 2:  # Add possible cleaning events found by considering
                    # the residuals
                    pce = soiling_dfs[-1].cleaning_events.copy()
                    cleaning_sensitivity *= 1.2  # decrease sensitivity
                    ce, rm9 = _rolling_median_ce_detection(
                        pi.index, residuals, ffill=ffill, tuner=cleaning_sensitivity
                    )
                    ce = _collapse_cleaning_events(ce, rm9.diff().values, 5)
                    pce[ce] = True
                    clean_pruning_sensitivity /= 1.1  # increase pruning sensitivity

                # Decompose input signal
                soiling_dummy = (
                    pi / degradation_trend[-1] / seasonal_component[-1] / residual_shift
                )

                # Run Kalman Filter for obtaining soiling component
                kdf, Ps = self._Kalman_filter_for_SR(
                    zs_series=soiling_dummy,
                    clip_soiling=clip_soiling,
                    prescient_cleaning_events=pce,
                    pruning_iterations=pruning_iterations,
                    clean_pruning_sensitivity=clean_pruning_sensitivity,
                    perfect_cleaning=perfect_cleaning,
                    process_noise=process_noise,
                    renormalize_SR=renormalize_SR,
                )
                soiling_ratio.append(kdf.soiling_ratio)
                soiling_dfs.append(kdf)

            # Find seasonal component
            if order[(ic - 1) % n_steps] == "SC":
                season_dummy = pi / soiling_ratio[-1]  # Decompose signal
                if season_dummy.isna().sum() > 0:
                    season_dummy.interpolate("linear", inplace=True)
                season_dummy = season_dummy.apply(np.log)  # Log transform
                # Run STL model
                STL_res = STL(
                    season_dummy,
                    period=365,
                    seasonal=999999,
                    seasonal_deg=0,
                    trend_deg=0,
                    robust=True,
                    low_pass_jump=30,
                    seasonal_jump=30,
                    trend_jump=365,
                ).fit()
                # Smooth result
                smooth_season = lowess(
                    STL_res.seasonal.apply(np.exp),
                    pi.index,
                    is_sorted=True,
                    delta=30,
                    frac=180 / len(pi),
                    return_sorted=False,
                )
                # Ensure periodic seaonal component
                seasonal_comp = _force_periodicity(smooth_season, season_dummy.index, pi.index)
                seasonal_component.append(seasonal_comp)
                if degradation_method == "STL":  # If not YoY
                    deg_trend = pd.Series(index=pi.index, data=STL_res.trend.apply(np.exp))
                    degradation_trend.append(deg_trend / deg_trend.iloc[0])
                    yoy_save.append(
                        RdToolsDeg.degradation_year_on_year(
                            degradation_trend[-1], uncertainty_method=None
                        )
                    )

            # Find degradation component
            if order[(ic - 1) % n_steps] == "Rd":
                # Decompose signal
                trend_dummy = pi / seasonal_component[-1] / soiling_ratio[-1]
                # Run YoY
                yoy = RdToolsDeg.degradation_year_on_year(trend_dummy, uncertainty_method=None)
                # Convert degradation rate to trend
                degradation_trend.append(
                    pd.Series(index=pi.index, data=(1 + day * yoy / 100 / 365.0))
                )
                yoy_save.append(yoy)

            # Combine and calculate residual flatness
            total_model = degradation_trend[-1] * seasonal_component[-1] * soiling_ratio[-1]
            residuals = pi / total_model
            residual_shift = residuals.mean()
            total_model *= residual_shift
            convergence_metric.append(_RMSE(pi, total_model))

            if verbose:
                print(
                    "{:}\t{:}\t{:.5f}\t\t\t{:.1f} s".format(
                        ic,
                        order[(ic - 1) % n_steps],
                        convergence_metric[-1],
                        time.time() - t0,
                    )
                )

            # Convergence happens if there is no improvement in RMSE from one
            # step to the next
            if ic >= n_steps:
                relative_improvement = (
                    convergence_metric[-n_steps - 1] - convergence_metric[-1]
                ) / convergence_metric[-n_steps - 1]
                if perfect_cleaning and (
                    ic >= max_iterations / 2 or relative_improvement < convergence_criterion
                ):
                    # From now on, do not assume perfect cleaning
                    perfect_cleaning = False
                    # Reorder to ensure SR first
                    order = tuple(
                        [
                            order[(i + n_steps - 1 - (ic - 1) % n_steps) % n_steps]
                            for i in range(n_steps)
                        ]
                    )
                    change_point = ic
                    if verbose:
                        print("Now not assuming perfect cleaning")
                elif not perfect_cleaning and (
                    ic >= max_iterations
                    or (
                        ic >= change_point + n_steps
                        and relative_improvement < convergence_criterion
                    )
                ):
                    if verbose:
                        if relative_improvement < convergence_criterion:
                            print("Convergence reached.")
                        else:
                            print("Max iterations reached.")
                    ic = max_iterations

        # Initialize output DataFrame
        df_out = pd.DataFrame(
            index=pi.index,
            columns=[
                "soiling_ratio",
                "soiling_rates",
                "cleaning_events",
                "seasonal_component",
                "degradation_trend",
                "total_model",
                "residuals",
            ],
        )

        # Save values
        df_out.seasonal_component = seasonal_component[-1]
        df_out.degradation_trend = degradation_trend[-1]
        degradation = yoy_save[-1]
        final_kdf = soiling_dfs[-1]
        df_out.soiling_ratio = final_kdf.soiling_ratio
        df_out.soiling_rates = final_kdf.soiling_rates
        df_out.cleaning_events = final_kdf.cleaning_events

        # Calculate soiling loss in %
        soiling_loss = (1 - df_out.soiling_ratio).mean() * 100

        # Total model
        df_out.total_model = (
            df_out.soiling_ratio * df_out.seasonal_component * df_out.degradation_trend
        )
        df_out.residuals = pi / df_out.total_model
        residual_shift = df_out.residuals.mean()
        df_out.total_model *= residual_shift
        RMSE = _RMSE(pi, df_out.total_model)
        adf_res = adfuller(df_out.residuals.dropna(), regression="ctt", autolag=None)
        if verbose:
            print(
                "p-value for the H0 that there is a unit root in the"
                + "residuals (using the Augmented Dickey-fuller test):"
                + "{:.3e}".format(adf_res[1])
            )

        # Check size of soiling signal vs residuals
        SR_amp = float(np.diff(df_out.soiling_ratio.quantile([0.1, 0.9])))
        residuals_amp = float(np.diff(df_out.residuals.quantile([0.1, 0.9])))
        soiling_signal_strength = SR_amp / residuals_amp
        if soiling_signal_strength < soiling_significance:
            if verbose:
                print("Soiling signal is small relative to the noise")
            small_soiling_signal = True
            df_out.SR_high = 1.0
            df_out.SR_low = 1.0 - SR_amp

        # Set up results dictionary
        results_dict = dict(
            degradation=degradation,
            soiling_loss=soiling_loss,
            residual_shift=residual_shift,
            RMSE=RMSE,
            small_soiling_signal=small_soiling_signal,
            adf_res=adf_res,
        )

        return df_out, results_dict

    def run_bootstrap(
        self,
        reps=512,
        confidence_level=68.2,
        degradation_method="YoY",
        process_noise=1e-4,
        order_alternatives=(("SR", "SC", "Rd"), ("SC", "SR", "Rd")),
        cleaning_sensitivity_alternatives=(0.25, 0.75),
        clean_pruning_sensitivity_alternatives=(1 / 1.5, 1.5),
        forward_fill_alternatives=(True, False),
        verbose=False,
        **kwargs,
    ):
        """
        Bootstrapping of CODS algorithm for uncertainty analysis, inherently accounting
        for model and parameter choices.

        First, calls on :py:func:`iterative_signal_decomposition` to fit N different
        models. Next, bootstrap samples are generated based on these N initial model fits.
        Each bootstrap sample is generated by bootstrapping the residuals of
        the respective model fit, using circular block
        bootstrapping, then multiplying these new residuals back onto the
        model. Then, for each bootstrap sample, model parameters are randomly
        chosen and the CODS model is fit to the bootstrapped signal.
        The seasonal component is perturbed randomly and
        divided out, so as to capture its uncertainty. In the end,
        confidence intervals are calulated based on the percentile levels of the
        collection of bootstrapped model fits.
        The returned soiling ratio and rates are based on
        the best fit of the initial N models. See [1]_ for more details.

        Parameters
        ----------
        reps : int, default 512,
            Number of bootstrap realizations to be run
            minimum N, where N is the possible combinations of model
            alternatives defined by possible combination of order_alternatives,
            cleaning_sensitivity_alternatives, clean_pruning_sensitivity_alternatives and
            forward_fill_alternatives.
        confidence_level : float, default 68.2
            The size of the confidence intervals to return, in percent
        degradation_method : string, default 'YoY'
            Either 'YoY' or 'STL'. If anything else, 'YoY' will be assumed.
            Decides whether to use the YoY method [3] for estimating the
            degradation trend (assumes linear trend), or the STL-method (does
            not assume linear trend). The latter is slower.
        order_alternatives : tuple of tuples, default (('SR', 'SC', 'Rd'), ('SC', 'SR', 'Rd'))
            Component estimation orders that will be tested during initial
            model fitting.
        cleaning_sensitivity_alternatives : tuple, default (.25, .75)
            Detection tuner values that will be tested during initial fitting.
            Length must be >= 1. First and last values define limits of values
            that will be used during bootstrapping.
        clean_pruning_sensitivity_alternatives : tuple, default (1/1.5, 1.5)
            Pruning tuner values that will be tested during initial fitting.
            Length must be >= 1. First and last values define limits of values
            that will be used during bootstrapping.
        forward_fill_alternatives : tuple, default (True, False)
            Forward fill values that will be tested during initial fitting.
        verbose : bool, default False
            Wheter or not to print information about progress
        **kwargs
            Keyword arguments that are passed on to
            :py:func:`iterative_signal_decomposition`

        Returns
        -------
        result_df : pandas.DataFrame with pandas datetimeindex
            Contains the columns/keys:

            +------------------------+----------------------------------------------+
            | Column Name            | Description                                  |
            +========================+==============================================+
            | 'soiling_ratio'        | soiling ratio (SR) (-)                       |
            +------------------------+----------------------------------------------+
            | 'soiling_rates'        | soiling rates (1/day)                        |
            +------------------------+----------------------------------------------+
            | 'cleaning_events'      | True at cleaning events                      |
            +------------------------+----------------------------------------------+
            | 'seasonal_component'   | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'degradation_trend'    | degradation trend (Rd)                       |
            +------------------------+----------------------------------------------+
            | 'total_model'          | the total model fit, i.e. SR * SC * Rd * rs, |
            |                        | where SR is the soiling ratio, SC is the     |
            |                        | seasonal component, Rd is the degradation    |
            |                        | trend, and rs is the residual shift, i.e.    |
            |                        | the mean of the residuals (adjusting the     |
            |                        | position of the model fit to the position of |
            |                        | the input data)                              |
            +------------------------+----------------------------------------------+
            | 'residuals'            | The residuals of the model fit, i.e.         |
            |                        | PI / (SR * SC * Rd)                          |
            +------------------------+----------------------------------------------+
            | 'SR_low'               | lower bound of 95 % conf. interval of SR     |
            +------------------------+----------------------------------------------+
            | 'SR_high'              | upper bound of 95 % conf. interval of SR     |
            +------------------------+----------------------------------------------+
            | 'rates_low'            | lower bound of 95 % conf. interval of        |
            |                        | soiling rates                                |
            +------------------------+----------------------------------------------+
            | 'rates_high'           | upper bound of 95 % conf. interval of        |
            |                        | soiling rates                                |
            +------------------------+----------------------------------------------+
            | 'bt_soiling_ratio'     | Bootstrapped estimate of soiling ratio (SR)  |
            +------------------------+----------------------------------------------+
            | 'bt_soiling_rates'     | Bootstrapped estimate of soiling rates       |
            +------------------------+----------------------------------------------+
            | 'seasonal_low'         | lower bound of 95 % conf. interval of        |
            |                        | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'seasonal_high'        | upper bound of 95 % conf. interval of        |
            |                        | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'model_high'           | upper bound of 95 % confidence interval of   |
            |                        | the model fit                                |
            +------------------------+----------------------------------------------+
            | 'model_low'            | lower bound of 95 % confidence interval of   |
            |                        | the model fit                                |
            +------------------------+----------------------------------------------+

        degradation : list
            List of linear degradation rate of system in %/year, lower and
            upper bound of confidence interval.
        soiling_loss : list
            List of average soiling losses over the time series in %, lower and
            upper bound of confidence interval.

        References
        ----------
        .. [1] Skomedal, Å. and Deceglie, M. G., IEEE Journal of
           Photovoltaics, Sept. 2020. https://doi.org/10.1109/JPHOTOV.2020.3018219
        """
        pi = self.pm.copy()

        # ###################### #
        # ###### STAGE 1 ####### #
        # ###################### #

        # Generate combinations of model parameter alternatives
        parameter_alternatives = [
            order_alternatives,
            cleaning_sensitivity_alternatives,
            clean_pruning_sensitivity_alternatives,
            forward_fill_alternatives,
        ]
        index_list = list(itertools.product([0, 1], repeat=len(parameter_alternatives)))
        combination_of_parameters = [
            [parameter_alternatives[j][indexes[j]] for j in range(len(parameter_alternatives))]
            for indexes in index_list
        ]
        nr_models = len(index_list)
        bootstrap_samples_list, list_of_df_out, results = [], [], []

        # Check boostrap number
        if reps % nr_models != 0:
            reps += nr_models - reps % nr_models

        if verbose:
            print("Initially fitting {:} models".format(nr_models))
        t00 = time.time()
        # For each combination of model parameter alternatives, fit one model:
        for c, (order, dt, pt, ff) in enumerate(combination_of_parameters):
            try:
                df_out, result_dict = self.iterative_signal_decomposition(
                    max_iterations=18,
                    order=order,
                    clip_soiling=True,
                    cleaning_sensitivity=dt,
                    pruning_iterations=1,
                    clean_pruning_sensitivity=pt,
                    process_noise=process_noise,
                    ffill=ff,
                    degradation_method=degradation_method,
                    **kwargs,
                )

                # Save results
                list_of_df_out.append(df_out)
                results.append(result_dict)
                adf = result_dict["adf_res"]
                # If we can reject the null-hypothesis that there is a unit
                # root in the residuals:
                if adf[1] < 0.05:
                    # ... generate bootstrap samples based on the fit:
                    bootstrap_samples_list.append(
                        _make_time_series_bootstrap_samples(
                            pi, df_out.total_model, sample_nr=int(reps / nr_models)
                        )
                    )

                # Print progress
                if verbose:
                    _progressBarWithETA(c + 1, nr_models, time.time() - t00, bar_length=30)
            except ValueError as ex:
                print(ex)

        # Revive results
        adfs = np.array([(r["adf_res"][0] if r["adf_res"][1] < 0.05 else 0) for r in results])
        RMSEs = np.array([r["RMSE"] for r in results])
        SR_is_one_fraction = np.array([(df.soiling_ratio == 1).mean() for df in list_of_df_out])
        small_soiling_signal = [r["small_soiling_signal"] for r in results]

        # Calculate weights
        weights = 1 / RMSEs / (1 + SR_is_one_fraction)
        weights /= np.sum(weights)

        # Save sensitivities and weights for initial model fits
        _parameters_n_weights = pd.concat(
            [
                pd.DataFrame(combination_of_parameters),
                pd.Series(RMSEs),
                pd.Series(SR_is_one_fraction),
                pd.Series(weights),
                pd.Series(small_soiling_signal),
            ],
            axis=1,
            ignore_index=True,
        )

        if verbose:  # Print summary
            _parameters_n_weights.columns = [
                "order",
                "dt",
                "pt",
                "ff",
                "RMSE",
                "SR==1",
                "weights",
                "small_soiling_signal",
            ]
            if verbose:
                print("\n", _parameters_n_weights)

        # Check if data is decomposable
        if np.sum(adfs == 0) > nr_models / 2:
            raise RuntimeError(
                "Test for stationary residuals (Augmented Dickey-Fuller"
                + " test) not passed in half  of the instances:\nData not"
                + " decomposable."
            )

        # Save best model
        self.initial_fits = [df for df in list_of_df_out]
        result_df = list_of_df_out[np.argmax(weights)]

        # If more than half of the model fits indicate small soiling signal,
        # don't do bootstrapping
        if np.sum(small_soiling_signal) > nr_models / 2:
            self.result_df = result_df
            self.residual_shift = results[np.argmax(weights)]["residual_shift"]
            YOY = RdToolsDeg.degradation_year_on_year(pi)
            self.degradation = [YOY[0], YOY[1][0], YOY[1][1]]
            self.soiling_loss = [0, 0, (1 - result_df.soiling_ratio).mean()]
            self.small_soiling_signal = True
            self.errors = (
                "Soiling signal is small relative to the noise. "
                "Iterative decomposition not possible. "
                "Degradation found by RdTools YoY."
            )
            warnings.warn(self.errors)
            return self.result_df, self.degradation, self.soiling_loss
        self.small_soiling_signal = False

        # Aggregate all bootstrap samples
        all_bootstrap_samples = pd.concat(bootstrap_samples_list, axis=1, ignore_index=True)

        # Seasonal samples are generated from previously fitted seasonal
        # components, by perturbing amplitude and phase shift
        # Number of samples per fit:
        sample_nr = int(reps / nr_models)
        list_of_SCs = [
            list_of_df_out[m].seasonal_component for m in range(nr_models) if weights[m] > 0
        ]
        seasonal_samples = _make_seasonal_samples(
            list_of_SCs,
            sample_nr=sample_nr,
            min_multiplier=0.8,
            max_multiplier=1.75,
            max_shift=30,
        )

        # ###################### #
        # ###### STAGE 2 ####### #
        # ###################### #

        if verbose and reps > 0:
            print(
                "\nBootstrapping for uncertainty analysis",
                "({:} realizations):".format(reps),
            )
        order = ("SR", "SC" if degradation_method == "STL" else "Rd")
        t0 = time.time()
        bt_kdfs, bt_SL, bt_deg, parameters, adfs, RMSEs, SR_is_1, rss, errors = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ["Bootstrapping errors"],
        )
        for b in range(reps):
            try:
                # randomly choose model sensitivities
                dt = np.random.uniform(parameter_alternatives[1][0], parameter_alternatives[1][-1])
                pt = np.random.uniform(parameter_alternatives[2][0], parameter_alternatives[2][-1])
                pn = np.random.uniform(process_noise / 1.5, process_noise * 1.5)
                renormalize_SR = np.random.choice([None, np.random.uniform(0.5, 0.95)])
                ffill = np.random.choice([True, False])
                parameters.append([dt, pt, pn, renormalize_SR, ffill])

                # Sample to infer soiling from
                bootstrap_sample = all_bootstrap_samples[b] / seasonal_samples[b]

                # Set up a temprary instance of the CODSAnalysis object
                temporary_cods_instance = CODSAnalysis(bootstrap_sample)

                # Do Signal decomposition for soiling and degradation component
                kdf, results_dict = temporary_cods_instance.iterative_signal_decomposition(
                    max_iterations=4,
                    order=order,
                    clip_soiling=True,
                    cleaning_sensitivity=dt,
                    pruning_iterations=1,
                    clean_pruning_sensitivity=pt,
                    process_noise=pn,
                    renormalize_SR=renormalize_SR,
                    ffill=ffill,
                    degradation_method=degradation_method,
                    **kwargs,
                )

                # If we can reject the null-hypothesis that there is a unit
                # root in the residuals:
                if results_dict["adf_res"][1] < 0.05:  # Save the results
                    bt_kdfs.append(kdf)
                    adfs.append(results_dict["adf_res"][0])
                    RMSEs.append(results_dict["RMSE"])
                    bt_deg.append(results_dict["degradation"])
                    bt_SL.append(results_dict["soiling_loss"])
                    rss.append(results_dict["residual_shift"])
                    SR_is_1.append((kdf.soiling_ratio == 1).mean())
                else:
                    seasonal_samples.drop(columns=[b], inplace=True)

            except ValueError as ve:
                seasonal_samples.drop(columns=[b], inplace=True)
                errors.append([b, ve])

            # Print progress
            if verbose:
                _progressBarWithETA(b + 1, reps, time.time() - t0, bar_length=30)

        # Reweight and save weights
        weights = 1 / np.array(RMSEs) / (1 + np.array(SR_is_1))
        weights /= np.sum(weights)
        self._parameters_n_weights = pd.concat(
            [
                pd.DataFrame(parameters),
                pd.Series(RMSEs),
                pd.Series(adfs),
                pd.Series(SR_is_1),
                pd.Series(weights),
            ],
            axis=1,
            ignore_index=True,
        )
        self._parameters_n_weights.columns = [
            "dt",
            "pt",
            "pn",
            "RSR",
            "ffill",
            "RMSE",
            "ADF",
            "SR==1",
            "weights",
        ]

        # ###################### #
        # ###### STAGE 3 ####### #
        # ###################### #

        # Set confidence interval edge quantile levels
        ci_low_edge = (50 - confidence_level / 2) / 100
        ci_high_edge = (50 + confidence_level / 2) / 100

        # Concatenate boostrap model fits
        concat_tot_mod = pd.concat([kdf.total_model for kdf in bt_kdfs], axis=1)
        concat_SR = pd.concat([kdf.soiling_ratio for kdf in bt_kdfs], axis=1)
        concat_r_s = pd.concat([kdf.soiling_rates for kdf in bt_kdfs], axis=1)
        concat_ce = pd.concat([kdf.cleaning_events for kdf in bt_kdfs], axis=1)

        # Find confidence intervals for SR and soiling rates
        df_out["SR_low"] = concat_SR.quantile(ci_low_edge, 1)
        df_out["SR_high"] = concat_SR.quantile(ci_high_edge, 1)
        df_out["rates_low"] = concat_r_s.quantile(ci_low_edge, 1)
        df_out["rates_high"] = concat_r_s.quantile(ci_high_edge, 1)

        # Save best estimate and bootstrapped estimates of SR and soiling rates
        df_out.soiling_ratio = df_out.soiling_ratio.clip(lower=0, upper=1)
        df_out.loc[df_out.soiling_ratio.diff() == 0, "soiling_rates"] = 0
        df_out["bt_soiling_ratio"] = (concat_SR * weights).sum(1)
        df_out["bt_soiling_rates"] = (concat_r_s * weights).sum(1)

        # Set probability of cleaning events
        df_out.cleaning_events = (concat_ce * weights).sum(1)

        # Find degradation rates
        self.degradation = [
            np.dot(bt_deg, weights),
            np.quantile(bt_deg, ci_low_edge),
            np.quantile(bt_deg, ci_high_edge),
        ]
        df_out.degradation_trend = 1 + np.arange(len(pi)) * self.degradation[0] / 100 / 365.0

        # Soiling losses
        self.soiling_loss = [
            np.dot(bt_SL, weights),
            np.quantile(bt_SL, ci_low_edge),
            np.quantile(bt_SL, ci_high_edge),
        ]

        # Save "confidence intervals" for seasonal component
        df_out.seasonal_component = (seasonal_samples * weights).sum(1)
        df_out["seasonal_low"] = seasonal_samples.quantile(ci_low_edge, 1)
        df_out["seasonal_high"] = seasonal_samples.quantile(ci_high_edge, 1)

        # Total model with confidence intervals
        df_out.total_model = (
            df_out.degradation_trend * df_out.seasonal_component * df_out.soiling_ratio
        )
        df_out["model_low"] = concat_tot_mod.quantile(ci_low_edge, 1)
        df_out["model_high"] = concat_tot_mod.quantile(ci_high_edge, 1)

        # Residuals and residual shift
        df_out.residuals = pi / df_out.total_model
        self.residual_shift = df_out.residuals.mean()
        df_out.total_model *= self.residual_shift
        self.RMSE = _RMSE(pi, df_out.total_model)
        self.adf_results = adfuller(df_out.residuals.dropna(), regression="ctt", autolag=None)
        self.result_df = df_out
        self.errors = errors

        if verbose:
            print("\nFinal RMSE: {:.5f}".format(self.RMSE))
            if len(self.errors) > 1:
                print(self.errors)

        return self.result_df, self.degradation, self.soiling_loss

    def _Kalman_filter_for_SR(
        self,
        zs_series,
        process_noise=1e-4,
        zs_std=0.05,
        rate_std=0.005,
        max_soiling_rates=0.0005,
        pruning_iterations=1,
        clean_pruning_sensitivity=0.6,
        renormalize_SR=None,
        perfect_cleaning=False,
        prescient_cleaning_events=None,
        clip_soiling=True,
        ffill=True,
    ):
        """
        A function for estimating the underlying Soiling Ratio (SR) and the
        rate of change of the SR (the soiling rate), based on a noisy time series
        of daily (corrected) normalized energy using a Kalman Filter (KF). See
        [1]_ for more details on Kalman Filters.

        Parameters
        ----------
        zs_series : pandas.Series
            Time series of daily normalized energy. Ideally corrected for degradation
            and seasonality
        process_noise : float, default 1e-4
            Represents the expected amount of unmodeled variation in the process itself
        zs_std : float, default 0.05
            Represents the expected variation in the zs_series
        rate_std : float, default 0.005
            Represents the expected variation in the rate of change of the zs_series
        max_soiling_rates : float, default 0.0005
            Represents the maximum allowed positive soiling rate (when soiling is removed)
        pruning_iterations : int, default 1
            Number of iterations when pruning (removing) cleaning events
        clean_pruning_sensitivity : float, default 0.6
            Sensitivity tuner that decides how easily a cleaning event is pruned
            (removed). Larger values means a smaller chance of pruning a given event.
        renormalize_SR : float or None, default None
            Quantile (of subsequent zs_series-values after cleaning events) for which
            to normalize SR against.
        perfect_cleaning : bool, default False
            Whether or not to assume perfect cleaning, i.e. SR = 1 after every
            cleaning event
        prescient_cleaning_events : list, pandas.Series, or None, default None
            List of "known" cleaning events that is passed on to the algorithm
        clip_soiling : bool, default True
            Whether or not to clip SR at a maximum value of 1
        ffill : bool, default True
            Whether to forward fill missing values when detecting cleaning events.

        Returns
        -------
        dfk : pandas.DataFrame
            Results of the Kalman Filter process. Contains the followig columns:

            +------------------------+----------------------------------------------+
            | Column Name            | Description                                  |
            +========================+==============================================+
            | 'raw_pi'               | Raw state estimate after Kalman Filter pass  |
            +------------------------+----------------------------------------------+
            | 'raw_rates'            | Raw rate estimate after Kalman Filter pass   |
            +------------------------+----------------------------------------------+
            | 'smooth_pi'            | Smoothed state estimate after running the    |
            |                        | smoother function                            |
            +------------------------+----------------------------------------------+
            | 'smooth_rates'         | Smoothed rate estimate after running the     |
            |                        | smoother function                            |
            +------------------------+----------------------------------------------+
            | 'soiling_ratio'        | soiling ratio (SR) estimate (-)              |
            +------------------------+----------------------------------------------+
            | 'soiling_rates'        | soiling rate estimate (1/day)                |
            +------------------------+----------------------------------------------+
            | 'cleaning_events'      | True at cleaning events                      |
            +------------------------+----------------------------------------------+
            | 'days_since_ce'        | Number of days since previous cleaning event |
            +------------------------+----------------------------------------------+

        Ps : numpy.array
            Array of covariance matrices for the states of each iteration of the Kalman
            Filter (one iteration per entry in zs_series).

        References
        ----------
        .. [1] R. R. Labbe, Kalman and Bayesian Filters in Python. 2016.
        """

        # Ensure numeric index
        zs_series = zs_series.copy()  # Make copy, so as not to change input
        original_index = zs_series.index.copy()
        if original_index.dtype not in [int, "int64"]:
            zs_series.index = range(len(zs_series))

        # Check prescient_cleaning_events. If not present, find cleaning events
        if isinstance(prescient_cleaning_events, list):
            cleaning_events = prescient_cleaning_events
        else:
            if isinstance(prescient_cleaning_events, type(zs_series)) and (
                prescient_cleaning_events.sum() > 4
            ):
                if len(prescient_cleaning_events) == len(zs_series):
                    prescient_cleaning_events = prescient_cleaning_events.copy()
                    prescient_cleaning_events.index = zs_series.index
                else:
                    raise ValueError(
                        "The indices of prescient_cleaning_events must correspond to the"
                        + " indices of zs_series; they must be of the same length"
                    )
            else:  # If no prescient cleaning events, detect cleaning events
                ce, rm9 = _rolling_median_ce_detection(zs_series.index, zs_series, tuner=0.5)
                prescient_cleaning_events = _collapse_cleaning_events(ce, rm9.diff().values, 5)

            cleaning_events = prescient_cleaning_events[prescient_cleaning_events].index.tolist()

        # Find soiling events (e.g. dust storms)
        soiling_events = _soiling_event_detection(zs_series.index, zs_series, ffill=ffill, tuner=5)
        soiling_events = soiling_events[soiling_events].index.tolist()

        # Initialize various parameters
        if ffill:
            rolling_median_13 = zs_series.ffill().rolling(13, center=True).median().ffill().bfill()
            rolling_median_7 = zs_series.ffill().rolling(7, center=True).median().ffill().bfill()
        else:
            rolling_median_13 = zs_series.bfill().rolling(13, center=True).median().ffill().bfill()
            rolling_median_7 = zs_series.bfill().rolling(7, center=True).median().ffill().bfill()
        # A rough estimate of the measurement noise
        measurement_noise = (rolling_median_13 - zs_series).var()
        # An initial guess of the slope
        initial_slope = np.array(theilslopes(zs_series.bfill().iloc[:14]))
        dt = 1  # All time stemps are one day

        # Initialize Kalman filter
        f = self._initialize_univariate_model(
            zs_series,
            dt,
            process_noise,
            measurement_noise,
            rate_std,
            zs_std,
            initial_slope,
        )

        # Initialize miscallenous variables
        dfk = pd.DataFrame(
            index=zs_series.index,
            dtype=float,
            columns=[
                "raw_pi",
                "raw_rates",
                "smooth_pi",
                "smooth_rates",
                "soiling_ratio",
                "soiling_rates",
                "cleaning_events",
                "days_since_ce",
            ],
        )
        dfk["cleaning_events"] = False

        # Kalman Filter part:
        #######################################################################
        # Call the forward pass function (the actual KF procedure)
        Xs, Ps, rate_std, zs_std = self._forward_pass(
            f, zs_series, rolling_median_7, cleaning_events, soiling_events
        )

        # Save results and smooth with rts smoother
        dfk, Xs, Ps = self._smooth_results(
            dfk, f, Xs, Ps, zs_series, cleaning_events, soiling_events, perfect_cleaning
        )
        #######################################################################

        # Some steps to clean up the soiling data:
        counter = 0
        while counter < pruning_iterations:
            counter += 1
            ce_0 = cleaning_events.copy()
            # 1: Remove false cleaning events by checking for outliers
            if len(ce_0) > 0:
                rm_smooth_pi = dfk.smooth_pi.rolling(7).median().shift(-6)
                pi_after_cleaning = rm_smooth_pi.loc[cleaning_events]
                # Detect outiers/false positives
                false_positives = _find_numeric_outliers(
                    pi_after_cleaning, clean_pruning_sensitivity, "lower"
                )
                cleaning_events = false_positives[~false_positives].index.tolist()

            # 2: Remove longer periods with positive (soiling) rates
            if (dfk.smooth_rates > max_soiling_rates).sum() > 1:
                exceeding_rates = dfk.smooth_rates > max_soiling_rates
                new_cleaning_events = _collapse_cleaning_events(
                    exceeding_rates, dfk.smooth_rates, 4
                )
                cleaning_events.extend(new_cleaning_events[new_cleaning_events].index)
                cleaning_events.sort()

            # 3: If the list of cleaning events has changed, run the Kalman
            #    Filter and smoother again
            if not ce_0 == cleaning_events:
                f = self._initialize_univariate_model(
                    zs_series,
                    dt,
                    process_noise,
                    measurement_noise,
                    rate_std,
                    zs_std,
                    initial_slope,
                )
                Xs, Ps, rate_std, zs_std = self._forward_pass(
                    f, zs_series, rolling_median_7, cleaning_events, soiling_events
                )
                dfk, Xs, Ps = self._smooth_results(
                    dfk,
                    f,
                    Xs,
                    Ps,
                    zs_series,
                    cleaning_events,
                    soiling_events,
                    perfect_cleaning,
                )

            else:
                counter = 100  # Make sure the while loop stops

            # 4: Estimate Soiling ratio from kalman estimate
            if perfect_cleaning:  # SR = 1 after cleaning events
                if len(cleaning_events) > 0:
                    pi_dummy = pd.Series(index=dfk.index, data=np.nan)
                    pi_dummy.loc[cleaning_events] = dfk.smooth_pi.loc[cleaning_events]
                    dfk.soiling_ratio = 1 / pi_dummy.ffill() * dfk.smooth_pi
                    # Set the SR in the first soiling period based on the mean
                    # ratio of the Kalman estimate (smooth_pi) and the SR
                    dfk.loc[: cleaning_events[0], "soiling_ratio"] = (
                        dfk.loc[: cleaning_events[0], "smooth_pi"]
                        * (dfk.soiling_ratio / dfk.smooth_pi).mean()
                    )
                else:  # If no cleaning events
                    dfk.soiling_ratio = 1
            else:  # Otherwise, if the inut signal has been decomposed, and
                # only contains a soiling component, the kalman estimate = SR
                dfk.soiling_ratio = dfk.smooth_pi
            # 5: Renormalize Soiling Ratio
            if renormalize_SR is not None:
                dfk.soiling_ratio /= dfk.loc[cleaning_events, "soiling_ratio"].quantile(
                    renormalize_SR
                )

            # 6: Force soiling ratio to not exceed 1:
            if clip_soiling:
                dfk.soiling_ratio.clip(upper=1, inplace=True)
                dfk.soiling_rates = dfk.smooth_rates
                dfk.loc[dfk.soiling_ratio.diff() == 0, "soiling_rates"] = 0

        # Set number of days since cleaning event
        nr_days_dummy = pd.Series(index=dfk.index, data=np.nan)
        nr_days_dummy.loc[cleaning_events] = [int(date - dfk.index[0]) for date in cleaning_events]
        nr_days_dummy.iloc[0] = 0
        dfk.days_since_ce = range(len(zs_series)) - nr_days_dummy.ffill()

        # Save cleaning events and soiling events
        dfk.loc[cleaning_events, "cleaning_events"] = True
        dfk.index = original_index  # Set index back to orignial index

        return dfk, Ps

    def _forward_pass(self, f, zs_series, rolling_median_7, cleaning_events, soiling_events):
        """Run the forward pass of the Kalman Filter algortihm"""
        zs = zs_series.values
        N = len(zs)
        Xs, Ps = np.zeros((N, 2)), np.zeros((N, 2, 2))
        # Enter forward pass of filtering algorithm
        for i, z in enumerate(zs):
            if 7 < i < N - 7 and (i in cleaning_events or i in soiling_events):
                rolling_median_local = rolling_median_7.loc[i - 5 : i + 5].values
                u = self._set_control_input(f, rolling_median_local, i, cleaning_events)
                f.predict(u=u)  # Predict wth control input u
            else:  # If no cleaning detection, predict without control input
                f.predict()
            if not np.isnan(z):
                f.update(z)  # Update

            Xs[i] = f.x
            Ps[i] = f.P
            rate_std, zs_std = Ps[-1, 1, 1], Ps[-1, 0, 0]
        return Xs, Ps, rate_std, zs_std  # Convert to numpy and return

    def _set_control_input(self, f, rolling_median_local, index, cleaning_events):
        """
        For each cleaning event, sets control input u based on current
        Kalman Filter state estimate (f.x), and the median value for the
        following week. If the cleaning event seems to be misplaced, moves
        the cleaning event to a more sensible location. If the cleaning
        event seems to be correct, removes other cleaning events in the 10
        days surrounding this day
        """
        u = np.zeros(f.x.shape)  # u is the control input
        window_size = 11  # len of rolling_median_local
        HW = 5  # Half window
        moving_diff = np.diff(rolling_median_local)
        # Index of maximum change in rolling median
        max_diff_index = moving_diff.argmax()
        if max_diff_index == HW - 1 or index not in cleaning_events:
            # The median zs of the week after the cleaning event
            z_med = rolling_median_local[HW + 3]
            # Set control input this future median
            u[0] = z_med - np.dot(f.H, np.dot(f.F, f.x))
            # If the change is bigger than the measurement noise:
            if np.abs(u[0]) > np.sqrt(f.R) / 2:
                index_dummy = [n + 3 for n in range(window_size - HW - 1) if n + 3 != HW]
                cleaning_events = [
                    ce for ce in cleaning_events if ce - index + HW not in index_dummy
                ]
            else:  # If the cleaning event is insignificant
                u[0] = 0
                if index in cleaning_events:
                    cleaning_events.remove(index)
        else:  # If the index with the maximum difference is not today...
            cleaning_events.remove(index)  # ...remove today from the list
            if (
                moving_diff[max_diff_index] > 0
                and index + max_diff_index - HW + 1 not in cleaning_events
            ):
                # ...and add the missing day
                bisect.insort(cleaning_events, index + max_diff_index - HW + 1)
        return u

    def _smooth_results(
        self,
        dfk,
        f,
        Xs,
        Ps,
        zs_series,
        cleaning_events,
        soiling_events,
        perfect_cleaning,
    ):
        """Smoother for Kalman Filter estimates. Smooths the Kalaman estimate
        between given cleaning events and saves all in DataFrame dfk"""
        # Save unsmoothed estimates
        dfk.raw_pi = Xs[:, 0]
        dfk.raw_rates = Xs[:, 1]

        # Set up cleaning events dummy list, inlcuding first and last day
        df_num_ind = pd.Series(index=dfk.index, data=range(len(dfk)))
        ce_dummy = cleaning_events.copy()
        ce_dummy.extend(dfk.index[[0, -1]])
        ce_dummy.extend(soiling_events)
        ce_dummy.sort()

        # Smooth between cleaning events
        for start, end in zip(ce_dummy[:-1], ce_dummy[1:]):
            num_ind = df_num_ind.loc[start:end].iloc[:-1]
            Xs[num_ind], Ps[num_ind], _, _ = f.rts_smoother(Xs[num_ind], Ps[num_ind])

        # Save smoothed estimates
        dfk.smooth_pi = Xs[:, 0]
        dfk.smooth_rates = Xs[:, 1]

        return dfk, Xs, Ps

    def _initialize_univariate_model(
        self,
        zs_series,
        dt,
        process_noise,
        measurement_noise,
        rate_std,
        zs_std,
        initial_slope,
    ):
        """Initializes the univariate Kalman Filter model, using the filterpy
        package"""
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.F = np.array([[1.0, dt], [0.0, 1.0]])
        f.H = np.array([[1.0, 0.0]])
        f.P = np.array([[zs_std**2, 0], [0, rate_std**2]])
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise**2)
        f.x = np.array([initial_slope[1], initial_slope[0]])
        f.B = np.zeros(f.F.shape)
        f.B[0] = 1
        f.R = measurement_noise
        return f


def soiling_cods(
    energy_normalized_daily,
    reps=512,
    confidence_level=68.2,
    degradation_method="YoY",
    process_noise=1e-4,
    order_alternatives=(("SR", "SC", "Rd"), ("SC", "SR", "Rd")),
    cleaning_sensitivity_alternatives=(0.25, 0.75),
    clean_pruning_sensitivity_alternatives=(1 / 1.5, 1.5),
    forward_fill_alternatives=(True, False),
    verbose=False,
    **kwargs,
):
    """
    Functional wrapper for :py:class:`~rdtools.soiling.CODSAnalysis` and its
    subroutine :py:func:`~rdtools.soiling.CODSAnalysis.run_bootstrap`. Runs
    the combined degradation and soiling (CODS) algorithm with bootstrapping.
    Based on the procedure presented in [1]_.

    Parameters
    ----------
    energy_normalized_daily : pandas.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Alternatively, the soiling ratio output of a soiling sensor (e.g. the
        photocurrent ratio between matched dirty and clean PV reference cells).
        In either case, data should be insolation-weighted daily aggregates.
    reps : int, default 512
        number of bootstrap realizations to calculate
    confidence_level : float, default 68.2
        The size of the confidence interval to return, in percent
    degradation_method : string, default 'YoY'
        Either 'YoY' or 'STL'. If anything else, 'YoY' will be assumed.
        Decides whether to use the YoY method [3] for estimating the
        degradation trend (assumes linear trend), or the STL-method (does
        not assume linear trend). The latter is slower.
    process_noise : float, default 1e-4
        A Kalman Filter parameter that represents the expected amount of unmodeled
        variation in the process, the process being the variation in the
        performance index that is due to soiling, seasonality and degradation.
    order_alternatives : tuple of tuples, default (('SR', 'SC', 'Rd'), ('SC', 'SR', 'Rd'))
        Component estimation orders that will be tested during initial
        model fitting.
    cleaning_sensitivity_alternatives : tuple, default (.25, .75)
        Detection tuner values that will be tested during initial fitting.
        Length must be >= 1. First and last values define limits of values
        that will be used during bootstrapping.
    clean_pruning_sensitivity_alternatives : tuple, default (1/1.5, 1.5)
        Pruning tuner values that will be tested during initial fitting.
        Length must be >= 1. First and last values define limits of values
        that will be used during bootstrapping.
    forward_fill_alternatives : tuple, default (True, False)
        Forward fill values that will be tested during initial fitting.
    verbose : bool, default False
        Wheter or not to print information about progress
    **kwargs
        keyword arguments that are passed on to :py:func:`iterative_signal_decomposition`

    Returns
    -------
    soiling_ratio : float
        Average soiling ratio based on CODS analysis (%)
    soiling_ratio_confidence_interval : numpy.array
        95 % confidence interval of soiling ratio estimate (%)
    degradation_rate : float
        Estimated degradation rate (%/year)
    degradation_rate_confidence_interval : numpy.array
        95 % confidence interval for degradation rate estimate (%/year)
    result_df : pandas dataframe
        Time series results from the CODS algorithm. Index is pandas.DatetimeIndex
        with daily frequency. Contains the following columns:

            +------------------------+----------------------------------------------+
            | Column Name            | Description                                  |
            +========================+==============================================+
            | 'soiling_ratio'        | soiling ratio (SR) (-)                       |
            +------------------------+----------------------------------------------+
            | 'soiling_rates'        | soiling rates (1/day)                        |
            +------------------------+----------------------------------------------+
            | 'cleaning_events'      | True at cleaning events                      |
            +------------------------+----------------------------------------------+
            | 'seasonal_component'   | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'degradation_trend'    | degradation trend (Rd)                       |
            +------------------------+----------------------------------------------+
            | 'total_model'          | the total model fit, i.e. SR * SC * Rd * rs, |
            |                        | where SR is the soiling ratio, SC is the     |
            |                        | seasonal component, Rd is the degradation    |
            |                        | trend, and rs is the residual shift, i.e.    |
            |                        | the mean of the residuals (adjusting the     |
            |                        | position of the model fit to the position of |
            |                        | the input data)                              |
            +------------------------+----------------------------------------------+
            | 'residuals'            | The residuals of the model fit, i.e.         |
            |                        | PI / (SR * SC * Rd)                          |
            +------------------------+----------------------------------------------+
            | 'SR_low'               | lower bound of 95 % conf. interval of SR     |
            +------------------------+----------------------------------------------+
            | 'SR_high'              | upper bound of 95 % conf. interval of SR     |
            +------------------------+----------------------------------------------+
            | 'rates_low'            | lower bound of 95 % conf. interval of        |
            |                        | soiling rates                                |
            +------------------------+----------------------------------------------+
            | 'rates_high'           | upper bound of 95 % conf. interval of        |
            |                        | soiling rates                                |
            +------------------------+----------------------------------------------+
            | 'bt_soiling_ratio'     | Bootstrapped estimate of soiling ratio (SR)  |
            +------------------------+----------------------------------------------+
            | 'bt_soiling_rates'     | Bootstrapped estimate of soiling rates       |
            +------------------------+----------------------------------------------+
            | 'seasonal_low'         | lower bound of 95 % conf. interval of        |
            |                        | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'seasonal_high'        | upper bound of 95 % conf. interval of        |
            |                        | seasonal component (SC)                      |
            +------------------------+----------------------------------------------+
            | 'model_high'           | upper bound of 95 % confidence interval of   |
            |                        | the model fit                                |
            +------------------------+----------------------------------------------+
            | 'model_low'            | lower bound of 95 % confidence interval of   |
            |                        | the model fit                                |
            +------------------------+----------------------------------------------+

    References
    ----------
    .. [1] Skomedal, Å. and Deceglie, M. G., IEEE Journal of Photovoltaics,
       Sept. 2020. https://doi.org/10.1109/JPHOTOV.2020.3018219
    """

    CODS = CODSAnalysis(energy_normalized_daily)

    CODS.run_bootstrap(
        reps=reps,
        confidence_level=confidence_level,
        verbose=verbose,
        degradation_method=degradation_method,
        process_noise=process_noise,
        order_alternatives=order_alternatives,
        cleaning_sensitivity_alternatives=cleaning_sensitivity_alternatives,
        clean_pruning_sensitivity_alternatives=clean_pruning_sensitivity_alternatives,
        forward_fill_alternatives=forward_fill_alternatives,
        **kwargs,
    )

    sr = 1 - CODS.soiling_loss[0] / 100
    sr_ci = 1 - np.array(CODS.soiling_loss[1:3]) / 100

    return (
        sr,
        sr_ci,
        CODS.degradation[0],
        np.array(CODS.degradation[1:3]),
        CODS.result_df,
    )


def _collapse_cleaning_events(inferred_ce_in, metric, f=4):
    """A function for replacing quick successive cleaning events with one
        (most probable) cleaning event.

    Parameters
    ----------
    inferred_ce_in : pandas.Series
        Contains daily booelan values for cleaning events
    metric : numpy.array/pandas.Series
        A metric which is large when probability of cleaning is large
        (eg. daily difference in rolling median of performance index)
    f : int, default 4
        Number of time stamps to collapse in each direction

    Returns
    -------
    inferred_ce : pandas.Series
        boolean values for cleaning events
    """
    # Ensure numeric index
    if isinstance(inferred_ce_in.index, pd.core.indexes.datetimes.DatetimeIndex):
        saveindex = inferred_ce_in.copy().index
        inferred_ce_in.index = range(len(saveindex))
    else:
        saveindex = inferred_ce_in.index

    # Make metric into series with same index
    metric = pd.Series(index=inferred_ce_in.index, data=np.array(metric))
    # Make a dummy, removing the f days at the beginning and end
    collapsed_ce_dummy = inferred_ce_in.iloc[f:-f].copy()
    # Make holder for collapes cleaning events
    collapsed_ce = pd.Series(index=inferred_ce_in.index, data=False)
    # Find the index of the first "island" of true values
    start_true_vals = collapsed_ce_dummy.idxmax()
    # Loop through data
    while start_true_vals > 0:
        # Find end of island of true values
        end_true_vals = collapsed_ce_dummy.loc[start_true_vals:].idxmin() - 1
        if end_true_vals >= start_true_vals:  # If the island ends
            # Find the day with mac probability of being a cleaning event
            max_diff_day = metric.loc[start_true_vals - f : end_true_vals + f].idxmax()
            # Set all days in this period as false
            collapsed_ce.loc[start_true_vals - f : end_true_vals + f] = False
            collapsed_ce_dummy.loc[start_true_vals - f : end_true_vals + f] = False
            # Set the max probability day as True (cleaning event)
            collapsed_ce.loc[max_diff_day] = True
            # Find the next island of true values
            start_true_vals = collapsed_ce_dummy.idxmax()
            if start_true_vals == f:
                start_true_vals = 0  # Stop iterations
        else:
            start_true_vals = 0  # Stop iterations
    # Return the series of collapsed cleaning events with the original index
    return pd.Series(index=saveindex, data=collapsed_ce.values)


def _rolling_median_ce_detection(x, y, ffill=True, rolling_window=9, tuner=1.5):
    """Finds cleaning events in a time series of performance index (y)"""
    y = pd.Series(index=x, data=y)
    if ffill:  # forward fill NaNs in y before running mean
        rm = y.ffill().rolling(rolling_window, center=True).median()
    else:  # ... or backfill instead
        rm = y.bfill().rolling(rolling_window, center=True).median()
    Q3 = rm.diff().abs().quantile(0.75)
    Q1 = rm.diff().abs().quantile(0.25)
    limit = Q3 + tuner * (Q3 - Q1)
    cleaning_events = rm.diff() > limit
    return cleaning_events, rm


def _soiling_event_detection(x, y, ffill=True, tuner=5):
    """Finds cleaning events in a time series of performance index (y)"""
    y = pd.Series(index=x, data=y)
    if ffill:  # forward fill NaNs in y before running mean
        rm = y.ffill().rolling(9, center=True).median()
    else:  # ... or backfill instead
        rm = y.bfill().rolling(9, center=True).median()
    Q3 = rm.diff().abs().quantile(0.99)
    Q1 = rm.diff().abs().quantile(0.01)
    limit = Q1 - tuner * (Q3 - Q1)
    soiling_events = rm.diff() < limit
    return soiling_events


def _make_seasonal_samples(
    list_of_SCs, sample_nr=10, min_multiplier=0.5, max_multiplier=2, max_shift=20
):
    """Generate seasonal samples by perturbing the amplitude and the phase of
    a seasonal components found with the fitted CODS model"""
    samples = pd.DataFrame(
        index=list_of_SCs[0].index,
        columns=range(int(sample_nr * len(list_of_SCs))),
        dtype=float,
    )
    # From each fitted signal, we will generate new seaonal components
    for i, signal in enumerate(list_of_SCs):
        # Remove beginning and end of signal
        signal_mean = signal.mean()
        # Make a signal matrix where each column is a year and each row a date
        year_matrix = (
            signal.rename("values")
            .to_frame()
            .assign(doy=signal.index.dayofyear, year=signal.index.year)
            .pivot(index="doy", columns="year", values="values")
        )
        # We will use the median signal through all the years...
        median_signal = year_matrix.median(1)
        for j in range(sample_nr):
            # Generate random multiplier and phase shift
            multiplier = np.random.uniform(min_multiplier, max_multiplier)
            shift = np.random.randint(-max_shift, max_shift)
            # Set up the signal by shifting the orginal signal index, and
            # constructing the new signal based on median_signal
            shifted_signal = pd.Series(
                index=signal.index,
                data=median_signal.reindex((signal.index.dayofyear - shift) % 365 + 1).values,
            )
            # Perturb amplitude by recentering to 0 multiplying by multiplier
            samples.loc[:, i * sample_nr + j] = multiplier * (shifted_signal - signal_mean) + 1
    return samples


def _force_periodicity(in_signal, signal_index, out_index):
    """Function for forcing periodicity in a seasonal component signal"""
    # Make sure the in_signal is a Series
    if isinstance(in_signal, np.ndarray):
        signal = pd.Series(index=pd.DatetimeIndex(signal_index.date), data=in_signal)
    elif isinstance(in_signal, pd.Series):
        signal = pd.Series(index=pd.DatetimeIndex(signal_index.date), data=in_signal.values)
    else:
        raise ValueError("in_signal must be numpy array or pandas Series")

    # Make sure that we don't remove too much of the data:
    remove_length = np.min([180, int((len(signal) - 365) / 2)])
    # Remove beginning and end of series
    signal.iloc[:remove_length] = np.nan
    signal.iloc[-remove_length:] = np.nan

    unique_years = signal.index.year.unique()  # Years involved in time series
    # Make a signal matrix where each column is a year and each row is a date
    year_matrix = pd.DataFrame(index=np.arange(0, 365), columns=unique_years)
    for year in unique_years:
        dates_in_year = pd.date_range(str(year) + "-01-01", str(year) + "-12-31")
        # We cut off the extra day(s) of leap years
        year_matrix[year] = signal.loc[str(year)].reindex(dates_in_year).values[:365]
    # We will use the median signal through all the years...
    median_signal = year_matrix.median(1)
    # The output is the median signal broadcasted to the whole time series
    output = pd.Series(index=out_index, data=median_signal.reindex(out_index.dayofyear - 1).values)
    return output


def _find_numeric_outliers(x, multiplier=1.5, where="both", verbose=False):
    """Function for finding numeric outliers"""
    try:  # Calulate third and first quartile
        Q3 = np.quantile(x, 0.75)
        Q1 = np.quantile(x, 0.25)
    except IndexError as ie:
        print(ie, x)
    except RuntimeWarning as rw:
        print(rw, x)
    IQR = Q3 - Q1  # Interquartile range
    if where == "upper":  # If detecting upper outliers
        if verbose:
            print("Upper limit", Q3 + multiplier * IQR)
        return x > Q3 + multiplier * IQR
    elif where == "lower":  # If detecting lower outliers
        if verbose:
            print("Lower limit", Q1 - multiplier * IQR)
        return x < Q1 - multiplier * IQR
    elif where == "both":  # If detecting both lower and upper outliers
        if verbose:
            print("Upper, lower limit", Q3 + multiplier * IQR, Q1 - multiplier * IQR)
        return (x > Q3 + multiplier * IQR), (x < Q1 - multiplier * IQR)


def _RMSE(y_true, y_pred):
    """Calculates the Root Mean Squared Error for y_true and y_pred, where
    y_pred is the "prediction", and y_true is the truth."""
    mask = ~pd.isnull(y_pred)
    return np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2))


def _MSD(y_true, y_pred):
    """Calculates the Mean Signed Deviation for y_true and y_pred, where y_pred
    is the "prediction", and y_true is the truth."""
    return np.mean(y_pred - y_true)


def _progressBarWithETA(value, endvalue, time, bar_length=20):
    """Prints a progressbar with an estimated time of "arrival" """
    percent = float(value) / endvalue * 100
    arrow = "-" * int(round(percent / 100 * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    used = time / 60  # Time Used
    left = used / percent * (100 - percent)  # Estimated time left
    sys.stdout.write(
        "\r# {:} | Used: {:.1f} min | Left: {:.1f}".format(value, used, left)
        + " min | Progress: [{:}] {:.0f} %".format(arrow + spaces, percent)
    )
    sys.stdout.flush()


###############################################################################
# all code below for new piecewise fitting in soiling intervals within srr/Matt
###############################################################################
def piecewise_linear(x, x0, b, k1, k2):
    cond_list = [x < x0, x >= x0]
    func_list = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0)]
    return np.piecewise(x, cond_list, func_list)


def segmented_soiling_period(
    pr,
    fill_method="bfill",
    days_clean_vs_cp=7,
    initial_guesses=[13, 1, 0, 0],
    bounds=None,
    min_r2=0.15,
):  # note min_r2 was 0.6 and it could be worth testing 10 day forward median as b guess
    """
    Applies segmented regression to a single deposition period
    (data points in between two cleaning events).
    Segmentation is neglected if change point occurs within a number of days
    (days_clean_vs_cp) of the cleanings.

    Parameters
    ----------
    pr :
        Series of daily performance ratios measured during the given deposition period.
    fill_method : str (default='bfill')
        Method to employ to fill any missing day.
    days_clean_vs_cp : numeric (default=7)
        Minimum number of days accepted between cleanings and change points.
    bounds : numeric (default=None)
        List of bounds for fitting function. If not specified, they are
        defined in the function.
    initial_guesses : numeric (default=0.1)
        List of initial guesses for fitting function
    min_r2 : numeric (default=0.1)
        Minimum R2 to consider valid the extracted soiling profile.

    Returns
    -------
    sr: numeric
        Series containing the daily soiling ratio values after segmentation.
        List of nan if segmentation was not possible.
    cp_date: datetime
        Datetime in which continuous change points occurred.
        None if segmentation was not possible.
    """

    # Check if PR dataframe has datetime index
    if not isinstance(pr.index, pd.DatetimeIndex):
        raise ValueError("The time series does not have DatetimeIndex")

    # Define bounds if not provided
    if bounds is None:
        # bounds are neg in first 4 and pos in second 4
        # ordered as x0,b,k1,k2 where x0 is the breakpoint k1 and k2 are slopes
        bounds = [(13, -5, -np.inf, -np.inf), ((len(pr) - 13), 5, +np.inf, +np.inf)]
    y = pr.values
    x = np.arange(0.0, len(y))

    try:
        # Fit soiling profile with segmentation
        p, e = curve_fit(piecewise_linear, x, y, p0=initial_guesses, bounds=bounds)

        # Ignore change point if too close to a cleaning
        # Change point p[0] converted to integer to extract a date.
        # None if no change point is found.
        if p[0] > days_clean_vs_cp and p[0] < len(y) - days_clean_vs_cp:
            z = piecewise_linear(x, *p)
            cp_date = int(p[0])
        else:
            z = [np.nan] * len(x)
            cp_date = None
        R2_original = st.linregress(y, x)[2] ** 2
        R2_piecewise = st.linregress(y, z)[2] ** 2

        R2_improve = R2_piecewise - R2_original
        R2_percent_improve = (R2_piecewise / R2_original) - 1
        R2_percent_of_possible_improve = R2_improve / (
            1 - R2_original
        )  # improvement relative to possible improvement

        if len(y) < 45:  # tighter requirements for shorter soiling periods
            if (R2_piecewise < min_r2) | (
                (R2_percent_of_possible_improve < 0.5) & (R2_percent_improve < 0.5)
            ):
                z = [np.nan] * len(x)
                cp_date = None
        else:
            if (R2_percent_improve < 0.01) | (R2_piecewise < 0.4):
                z = [np.nan] * len(x)
                cp_date = None
    except IndexError as x:
        z = [np.nan] * len(x)
        cp_date = None
    # Create Series from modelled profile
    sr = pd.Series(z, index=pr.index)

    return sr, cp_date
