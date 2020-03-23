'''Functions for calculating soiling metrics from photovoltaic system data.'''

from __future__ import division
from rdtools import degradation as RdToolsDeg
import pandas as pd
import numpy as np
from scipy.stats.mstats import theilslopes
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import itertools
import bisect
import time
import sys
from arch.bootstrap import CircularBlockBootstrap
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
lowess = sm.nonparametric.lowess


# Custom exception
class NoValidIntervalError(Exception):
    '''raised when no valid rows appear in the result dataframe'''
    pass


class srr_analysis():
    '''
    Class for running the stochastic rate and recovery (SRR) photovoltaic
    soiling loss analysis presented in Deceglie et al. JPV 8(2) p547 2018

    Parameters
    ----------
    daily_normalized_energy : pd.Series
        Daily performance metric (i.e. performance index, yield, etc.)
    daily_insolation : pd.Series
        Daily plane-of-array insolation corresponding to
        `daily_normalized_energy`
    precip : pd.Series, default None
        Daily total precipitation. (Only used if `precip_clean_only` is True in
        subsequent calculations)
    '''

    def __init__(self, daily_normalized_energy, daily_insolation, precip=None):
        self.pm = daily_normalized_energy  # daily performance metric
        self.insol = daily_insolation
        self.precip = precip  # daily precipitation
        self.random_profiles = []  # random soiling profiles in _calc_monte
        # insolation-weighted soiling ratios in _calc_monte:
        self.monte_losses = []

        if self.pm.index.freq != 'D':
            raise ValueError('Daily performance metric series must have '
                             'daily frequency')

        if self.insol.index.freq != 'D':
            raise ValueError('Daily insolation series must have '
                             'daily frequency')

        if self.precip is not None:
            if self.pm.index.freq != 'D':
                raise ValueError('Precipitation series must have '
                                 'daily frequency')

    def _calc_daily_df(self, day_scale=14, clean_threshold='infer',
                       recenter=True):
        '''
        Calculates self.daily_df, a pandas dataframe prepared for SRR analysis,
        and self.renorm_factor, the renormalization factor for the daily
        performance

        Parameters
        ----------
        day_scale : int, default 14
            The number of days to use in rolling median for cleaning detection

        clean_threshold : float or 'infer', default 'infer'
            If float: the fractional positive shift in rolling median for
            cleaning detection.
            If 'infer:' automatically use outliers in the shift as the
            threshold

        recenter : bool, default True
            Whether to recenter (renormalize) the daily performance to the
            median of the first year
        '''

        df = self.pm.to_frame()
        df.columns = ['pi']
        df_insol = self.insol.to_frame()
        df_insol.columns = ['insol']

        df = df.join(df_insol)
        precip = self.precip
        if precip is not None:
            df_precip = precip.to_frame()
            df_precip.columns = ['precip']
            df = df.join(df_precip)
        else:
            df['precip'] = 0

        # find first and last valid data point
        start = df[~df.pi.isnull()].index[0]
        end = df[~df.pi.isnull()].index[-1]
        df = df[start:end]

        # create a day count column
        df['day'] = range(len(df))

        # Recenter to median of first year, as in YoY degradation
        if recenter:
            oneyear = start + pd.Timedelta('364d')
            renorm = df.loc[start:oneyear, 'pi'].median()
        else:
            renorm = 1

        df['pi_norm'] = df['pi'] / renorm

        # Find the beginning and ends of outages longer than dayscale
        bfill = df['pi_norm'].fillna(method='bfill', limit=day_scale)
        ffill = df['pi_norm'].fillna(method='ffill', limit=day_scale)
        out_start = (~df['pi_norm'].isnull() & bfill.shift(-1).isnull())
        out_end = (~df['pi_norm'].isnull() & ffill.shift(1).isnull())

        # clean up the first and last elements
        out_start.iloc[-1] = False
        out_end.iloc[0] = False

        # Make a forward filled copy, just for use in
        # step, slope change detection
        df_ffill = df.fillna(method='ffill', limit=day_scale).copy()

        # Calculate rolling median
        df['pi_roll_med'] = \
            df_ffill.pi_norm.rolling(day_scale, center=True).median()

        # Detect steps in rolling median
        df['delta'] = df.pi_roll_med.diff()
        if clean_threshold == 'infer':
            deltas = abs(df.delta)
            clean_threshold = deltas.quantile(0.75) + \
                1.5 * (deltas.quantile(0.75) - deltas.quantile(0.25))

        df['clean_event'] = (df.delta > clean_threshold)
        df['clean_event'] = df.clean_event | out_start | out_end
        df['clean_event'] = (df.clean_event) & \
                            (~df.clean_event.shift(-1).fillna(False))

        # Detect which cleaning events are associated with rain
        rolling_precip = df.precip.rolling(3, center=True).sum()
        df['clean_wo_precip'] = ~(rolling_precip > 0.01) & (df.clean_event)

        df = df.fillna(0)

        # Give an index to each soiling interval/run
        run_list = []
        run = 0
        for x in df.clean_event:
            if x:
                run += 1
            run_list.append(run)

        df['run'] = run_list
        df.index.name = 'date'  # this gets used by name

        self.renorm_factor = renorm
        self.daily_df = df

    def _calc_result_df(self, trim=False, max_relative_slope_error=500.0,
                        max_negative_step=0.05):
        '''
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
        '''

        daily_df = self.daily_df
        result_list = []
        if trim:
            # ignore first and last interval
            res_loop = sorted(list(set(daily_df['run'])))[1:-1]
        else:
            res_loop = sorted(list(set(daily_df['run'])))

        for r in res_loop:
            run = daily_df[daily_df['run'] == r]
            length = (run.day[-1] - run.day[0])
            start_day = run.day[0]
            end_day = run.day[-1]
            run = run[run.pi_norm > 0]
            result_dict = {
                'start': run.index[0],
                'end': run.index[-1],
                'length': length,
                'run': r,
                'run_slope': 0,
                'run_slope_low': 0,
                'run_slope_high': 0,
                'max_neg_step': min(run.delta),
                'start_loss': 1,
                'clean_wo_precip': run.clean_wo_precip[0],
                'inferred_start_loss': run.pi_norm.mean(),
                'inferred_end_loss': run.pi_norm.mean(),
                'valid': False
            }
            if len(run) > 2 and run.pi_norm.sum() > 0:
                fit = theilslopes(run.pi_norm, run.day)
                fit_poly = np.poly1d(fit[0:2])
                result_dict['run_slope'] = fit[0]
                result_dict['run_slope_low'] = fit[2]
                result_dict['run_slope_high'] = min([0.0, fit[3]])
                result_dict['inferred_start_loss'] = fit_poly(start_day)
                result_dict['inferred_end_loss'] = fit_poly(end_day)
                result_dict['valid'] = True
            result_list.append(result_dict)

        results = pd.DataFrame(result_list)

        if results.empty:
            raise NoValidIntervalError('No valid soiling intervals were found')

        # Filter results for each interval,
        # setting invalid interval to slope of 0
        results['slope_err'] = (results.run_slope_high-results.run_slope_low) \
            / abs(results.run_slope)
        # critera for exclusions
        filt = (
            (results.run_slope > 0) |
            (results.slope_err >= max_relative_slope_error / 100.0) |
            (results.max_neg_step <= -1.0 * max_negative_step)
        )

        results.loc[filt, 'run_slope'] = 0
        results.loc[filt, 'run_slope_low'] = 0
        results.loc[filt, 'run_slope_high'] = 0
        results.loc[filt, 'valid'] = False

        # Calculate the next inferred start loss from next valid interval
        results['next_inferred_start_loss'] = np.clip(
                results[results.valid].inferred_start_loss.shift(-1),
                0, 1)
        # Calculate the inferred recovery at the end of each interval
        results['inferred_recovery'] = np.clip(
                results.next_inferred_start_loss - results.inferred_end_loss,
                0, 1)

        # Don't consider data outside of first and last valid interverals
        if len(results[results.valid]) == 0:
            raise NoValidIntervalError('No valid soiling intervals were found')
        new_start = results[results.valid].start.iloc[0]
        new_end = results[results.valid].end.iloc[-1]
        pm_frame_out = daily_df[new_start:new_end]
        pm_frame_out = pm_frame_out.reset_index() \
                                   .merge(results, how='left', on='run') \
                                   .set_index('date')

        pm_frame_out['loss_perfect_clean'] = np.nan
        pm_frame_out['loss_inferred_clean'] = np.nan
        pm_frame_out['days_since_clean'] = \
            (pm_frame_out.index - pm_frame_out.start).dt.days

        # Calculate the daily derate
        pm_frame_out['loss_perfect_clean'] = \
            pm_frame_out.start_loss + \
            pm_frame_out.days_since_clean * pm_frame_out.run_slope
        # filling the flat intervals may need to be recalculated
        # for different assumptions
        pm_frame_out.loss_perfect_clean = \
            pm_frame_out.loss_perfect_clean.fillna(1)

        pm_frame_out['loss_inferred_clean'] = \
            pm_frame_out.inferred_start_loss + \
            pm_frame_out.days_since_clean * pm_frame_out.run_slope
        # filling the flat intervals may need to be recalculated
        # for different assumptions
        pm_frame_out.loss_inferred_clean = \
            pm_frame_out.loss_inferred_clean.fillna(1)

        self.result_df = results
        self.analyzed_daily_df = pm_frame_out

    def _calc_monte(self, monte, method='half_norm_clean',
                    precip_clean_only=False, random_seed=None):
        '''
        Runs the Monte Carlo step of the SRR method. Calculates
        self.random_profiles, a list of the random soiling profiles realized in
        the calculation, and self.monte_losses, a list of the
        insolation-weighted soiling ratios associated with the realizations.

        Parameters
        ----------
        monte : int
            number of Monte Carlo simulations to run
        method : str, default 'half_norm_clean'
            how to treat the recovery of each cleaning event:
            * 'random_clean' - a random recovery between 0-100%
            * 'perfect_clean' - each cleaning event returns the performance
              metric to 1
            * 'half_norm_clean' - The three-sigma lower bound of recovery is
              inferred from the fit of the following interval, the upper bound
              is 1 with the magnitude drawn from a half normal centered at 1
        precip_clean_only : bool, default False
            If True, only consider cleaning events valid if they coincide with
            precipitation events
        random_seed : int, default None
            Seed for random number generation in the Monte Carlo simulation.
            Use to ensure identical results on subsequent runs. Not a
            substitute for doing a sufficient number of Mote Carlo repetitions.
        '''

        monte_losses = []
        random_profiles = []
        if random_seed is not None:
            np.random.seed(random_seed)
        for _ in range(monte):
            results_rand = self.result_df.copy()
            df_rand = self.analyzed_daily_df.copy()
            # only really need this column from the original frame:
            df_rand = df_rand[['insol', 'run']]
            results_rand['run_slope'] = \
                np.random.uniform(results_rand.run_slope_low,
                                  results_rand.run_slope_high)
            results_rand['run_loss'] = \
                results_rand.run_slope * results_rand.length

            results_rand['end_loss'] = np.nan
            results_rand['start_loss'] = np.nan

            # Make groups that start with a valid interval and contain
            # subsequent invalid intervals
            group_list = []
            group = 0
            for x in results_rand.valid:
                if x:
                    group += 1
                group_list.append(group)

            results_rand['group'] = group_list

            # randomize the extent of the cleaning
            inter_start = 1.0
            start_list = []
            if method == 'half_norm_clean':
                # Randomize recovery of valid intervals only
                valid_intervals = results_rand[results_rand.valid].copy()
                valid_intervals['inferred_recovery'] = \
                    valid_intervals.inferred_recovery.fillna(1.0)

                end_list = []
                for i, row in valid_intervals.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    end_list.append(end)

                    if row.clean_wo_precip and precip_clean_only:
                        # don't allow recovery if there was no precipitation
                        inter_start = end
                    else:
                        # Use a half normal with the infered clean at the
                        # 3sigma point
                        x = np.clip(end + row.inferred_recovery, 0, 1)
                        inter_start = 1 - abs(np.random.normal(0.0, (1 - x)/3))

                # Update the valid rows in results_rand
                valid_update = pd.DataFrame()
                valid_update['start_loss'] = start_list
                valid_update['end_loss'] = end_list
                valid_update.index = valid_intervals.index
                results_rand.update(valid_update)

                # forward and back fill to note the limits of random constant
                # derate for invalid intervals
                results_rand['previous_end'] = \
                    results_rand.end_loss.fillna(method='ffill')
                results_rand['next_start'] = \
                    results_rand.start_loss.fillna(method='bfill')

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
                    invalid_update['start_loss'] = replace_levels
                    invalid_update.index = invalid_intervals.index
                    results_rand.update(invalid_update)

            elif method == 'random_clean':
                for i, row in results_rand.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    if row.clean_wo_precip and precip_clean_only:
                        # don't allow recovery if there was no precipitation
                        inter_start = end
                    else:
                        inter_start = np.random.uniform(end, 1)
                results_rand['start_loss'] = start_list

            elif method == 'perfect_clean':
                for i, row in results_rand.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    if row.clean_wo_precip and precip_clean_only:
                        # don't allow recovery if there was no precipitation
                        inter_start = end
                    else:
                        inter_start = 1
                results_rand['start_loss'] = start_list

            else:
                raise ValueError("Invalid method specification")

            df_rand = df_rand.reset_index() \
                             .merge(results_rand, how='left', on='run') \
                             .set_index('date')
            df_rand['loss'] = np.nan
            df_rand['days_since_clean'] = \
                (df_rand.index - df_rand.start).dt.days
            df_rand['loss'] = df_rand.start_loss + \
                df_rand.days_since_clean * df_rand.run_slope

            df_rand['soil_insol'] = df_rand.loss * df_rand.insol

            monte_losses.append(df_rand.soil_insol.sum() / df_rand.insol.sum())
            random_profile = df_rand['loss'].copy()
            random_profile.name = 'stochastic_soiling_profile'
            random_profiles.append(random_profile)

        self.random_profiles = random_profiles
        self.monte_losses = monte_losses

    def run(self, reps=1000, day_scale=14, clean_threshold='infer',
            trim=False, method='half_norm_clean', precip_clean_only=False,
            exceedance_prob=95.0, confidence_level=68.2, recenter=True,
            max_relative_slope_error=500.0, max_negative_step=0.05,
            random_seed=None):
        '''
        Run the SRR method from beginning to end.  Perform the stochastic rate
        and recovery soiling loss calculation. Based on the methods presented
        in Deceglie et al. JPV 8(2) p547 2018.

        Parameters
        ----------
        reps : int, default 1000
            number of Monte Carlo realizations to calculate
        day_scale : int, default 14
            The number of days to use in rolling median for cleaning detection,
            and the maximum number of days of missing data to tolerate in a
            valid interval
        clean_threshold : float or 'infer', default 'infer'
            The fractional positive shift in rolling median for cleaning
            detection. Or specify 'infer' to automatically use outliers in the
            shift as the threshold.
        trim : bool, default False
            Whether to trim (remove) the first and last soiling intervals to
            avoid inclusion of partial intervals
        method : str, default 'half_norm_clean'
            How to treat the recovery of each cleaning event:

            * `random_clean` - a random recovery between 0-100%
            * `perfect_clean` - each cleaning event returns the performance
              metric to 1
            * `half_norm_clean` (default) - The three-sigma lower bound of
              recovery is inferred from the fit of the following interval, the
              upper bound is 1 with the magnitude drawn from a half normal
              centered at 1

        precip_clean_only : bool, default False
            If True, only consider cleaning events valid if they coincide with
            precipitation events
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
        random_seed : int, default None
            Seed for random number generation in the Monte Carlo simulation.
            Use to ensure identical results on subsequent runs. Not a
            substitute for doing a sufficient number of Mote Carlo repetitions.

        Returns
        -------
        insolation_weighted_soiling_ratio : float
            P50 insolation weighted soiling ratio based on stochastic rate and
            recovery analysis
        confidence_interval : np.array
            confidence interval (size specified by confidence_level) of
            degradation rate estimate
        calc_info : dict
            * `renormalizing_factor` - value used to recenter data
            * `exceedance_level` - the insolation-weighted soiling ratio that
              was outperformed with probability of exceedance_prob
            * `stochastic_soiling_profiles` - List of Pandas series
              corresponding to the Monte Carlo realizations of soiling ratio
              profiles
            * `soiling_interval_summary` - Pandas dataframe summarizing the
              soiling intervals identified
            * `soiling_ratio_perfect_clean` - Pandas series of the soiling
              ratio during valid soiling intervals assuming perfect cleaning
              and P50 slopes.
        '''
        self._calc_daily_df(day_scale=day_scale,
                            clean_threshold=clean_threshold,
                            recenter=recenter)
        self._calc_result_df(trim=trim,
                             max_relative_slope_error=max_relative_slope_error,
                             max_negative_step=max_negative_step)
        self._calc_monte(reps, method=method,
                         precip_clean_only=precip_clean_only,
                         random_seed=random_seed)

        # Calculate the P50 and confidence interval
        half_ci = confidence_level / 2.0
        result = np.percentile(self.monte_losses,
                               [50,
                                50.0 - half_ci,
                                50.0 + half_ci,
                                100 - exceedance_prob])
        P_level = result[3]

        # Construct calc_info output

        intervals_out = self.result_df[
                ['start', 'end', 'run_slope', 'run_slope_low',
                 'run_slope_high', 'inferred_start_loss', 'inferred_end_loss',
                 'length', 'valid']].copy()
        intervals_out.rename(columns={'run_slope': 'slope',
                                      'run_slope_high': 'slope_high',
                                      'run_slope_low': 'slope_low',
                                      }, inplace=True)

        df_d = self.analyzed_daily_df
        sr_perfect = df_d[df_d['valid']]['loss_perfect_clean']
        calc_info = {
            'exceedance_level': P_level,
            'renormalizing_factor': self.renorm_factor,
            'stochastic_soiling_profiles': self.random_profiles,
            'soiling_interval_summary': intervals_out,
            'soiling_ratio_perfect_clean': sr_perfect
        }

        return (result[0], result[1:3], calc_info)


def soiling_srr(daily_normalized_energy, daily_insolation, reps=1000,
                precip=None, day_scale=14, clean_threshold='infer',
                trim=False, method='half_norm_clean', precip_clean_only=False,
                exceedance_prob=95.0, confidence_level=68.2, recenter=True,
                max_relative_slope_error=500.0, max_negative_step=0.05,
                random_seed=None):
    '''
    Functional wrapper for srr_analysis(). Perform the stochastic rate and
    recovery soiling loss calculation. Based on the methods presented in
    Deceglie et al. JPV 8(2) p547 2018.

    Parameters
    ----------
    daily_normalized_energy : pd.Series
        Daily performance metric (i.e. performance index, yield, etc.)
    daily_insolation : pd.Series
        Daily plane-of-array insolation corresponding to d
        `daily_normalized_energy`
    reps : int, default 1000
        number of Monte Carlo realizations to calculate
    precip : pd.Series, default None
        Daily total precipitation. (Only used if precip_clean_only=True)
    day_scale : int, default 14
        The number of days to use in rolling median for cleaning detection,
        and the maximum number of days of missing data to tolerate in a valid
        interval
    clean_threshold : float or 'infer', default 'infer'
        The fractional positive shift in rolling median for cleaning detection.
        Or specify 'infer' to automatically use outliers in the shift as the
        threshold.
    trim : bool, default False
        Whether to trim (remove) the first and last soiling intervals to avoid
        inclusion of partial intervals
    method : str, default 'half_norm_clean'
        how to treat the recovery of each cleaning event

        * `random_clean` - a random recovery between 0-100%
        * `perfect_clean` - each cleaning event returns the performance metric
          to 1
        * `half_norm_clean` (default) - The three-sigma lower bound of recovery
          is inferred from the fit of the following interval, the upper bound
          is 1 with the magnitude drawn from a half normal centered at 1
    precip_clean_only : bool, default False
        If True, only consider cleaning events valid if they coincide with
        precipitation events
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
    random_seed : int, default None
        Seed for random number generation in the Monte Carlo simulation. Use to
        ensure identical results on subsequent runs. Not a substitute for doing
        a sufficient number of Mote Carlo repetitions.

    Returns
    -------
    insolation_weighted_soiling_ratio : float
        P50 insolation weighted soiling ratio based on stochastic rate and
        recovery analysis
    confidence_interval : np.array
        confidence interval (size specified by `confidence_level`) of
        degradation rate estimate
    calc_info : dict
        Calculation information from the SRR process.

        * `renormalizing_factor` - value used to recenter data
        * `exceedance_level` - the insolation-weighted soiling ratio that
          was outperformed with probability of exceedance_prob
        * `stochastic_soiling_profiles` - List of Pandas series
          corresponding to the Monte Carlo realizations of soiling
          ratio profiles
        * `soiling_interval_summary` - Pandas dataframe summarizing the
          soiling intervals identified
        * `soiling_ratio_perfect_clean` - Pandas series of the soiling
          ratio during valid soiling intervals assuming perfect cleaning
          and P50 slopes.
    '''

    srr = srr_analysis(daily_normalized_energy,
                       daily_insolation,
                       precip=precip)

    sr, sr_ci, soiling_info = srr.run(
            reps=reps,
            day_scale=day_scale,
            clean_threshold=clean_threshold,
            trim=trim,
            method=method,
            precip_clean_only=precip_clean_only,
            exceedance_prob=exceedance_prob,
            confidence_level=confidence_level,
            recenter=recenter,
            max_relative_slope_error=max_relative_slope_error,
            max_negative_step=max_negative_step,
            random_seed=random_seed)

    return sr, sr_ci, soiling_info


class cods_analysis():
    '''
    Class for running the Combined Degradation and Soling (CODS) algorithm
    for degradation and soiling loss analysis presented in 
    Skomedal and Deceglie. [JPV 8(2) p547] 2020
    The promary function to use is run_boostrap()

    Parameters
    ----------
    daily_normalized_energy : pd.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Index must be DatetimeIndex with daily frequency
    daily_insolation : pd.Series
        Daily plane-of-array insolation corresponding to
        `daily_normalized_energy`
    '''

    def __init__(self, daily_normalized_energy, daily_insolation):
        self.pm = daily_normalized_energy  # daily performance metric
        self.insol = daily_insolation  # daily insolation

        if self.pm.index.freq != 'D':
            raise ValueError('Daily performance metric series must have '
                             'daily frequency')

        if self.insol.index.freq != 'D':
            raise ValueError('Daily insolation series must have '
                             'daily frequency')


    def iterative_signal_decomposition(
        self, order=['SR', 'SC', 'Rd'], degradation_method='YoY',
        max_iterations=18, detection_tuner=.5, convergence_criterium=5e-3,
        pruning_iterations=1, pruning_tuner=.6, soiling_significance_knob=.75,
        process_noise=1e-4, renormalize_SR=None, perfect_cleaning=True,
        ffill=True, clip_soiling=True, verbose=False):
        '''
        Description
        -----------
        A function for doing iterative decomposition of Performance Index time
        series based on PV production data. The assumed underlying model
        consists of a degradation trend, a seasonal component, and a soiling
        signal (defined as 1 if no soiling, decreasing with increasing soiling
        losses).

            Model = degradation_trend * seasonal_component * soiling_ratio \
                    * residuals
              PI  ~         Rd        *         SC         *        SR     * R

        The function has a huristic for detecting whether the soiling signal is
        significant enough for soiling loss inference, which is based on the
        ratio between the spread in the soiling signal versus the spread in the
        residuals (defined by the 2.5th and 97.5th percentiles)

        The degradation trend is obtained using the native RdTools Year-On-Year
            method [1]
        The seasonal component is derived with statsmodels STL [2]
        The soiling signal is derived with a Kalman Filter with a cleaning
            detection heuristic [3]

        Parameters
        ----------
        order : list, defualt ['SR', 'SC', 'Rd']
            List containing 1 to 3 of the following strings 'SR' (soiling
            ratio), 'SC' (seasonal component), 'Rd' (degradation component),
            defining the order in which these components will be found during
            iterative decomposition
        degradation_method : string, default 'YoY'
            Either 'YoY' or 'STL'. If anything else, 'YoY' will be assumed.
            Decides whether to use the YoY method [3] for estimating the
            degradation trend (assumes linear trend), or the STL-method (does
            not assume linear trend). The latter is slower.
        max_iterations : int, default 18
            The number of iterations to perform (each iteration fits only 1
            component)
        detection_tuner : float, default .5
            Should be between 0.1 and 2
        convergence_criterium : float, default 1e-3
            the relative change in the convergence metric required for
            convergence
        pruning_iterations : int, default 1
        pruning_tuner : float, default .6
            Should be between 0.1 and 2
        soiling_significance_knob float, defualt 0.75
        process_noise : float, default 1e-4
        renormalize_SR : float, default None
            If not none, defines the percentile for which the SR will be
            normalized to, based on the SR just after cleaning events
        perfect_cleaning : bool, default False
            Defines the conversion mode for converting Kalman Filter estimates
            of the input signal to soiling ratio. See [3] for more details
        ffill : bool, default True
            Whether to use forward fill (default) or backward fill before
            doing the rolling median for cleaning event detection
        clip_soiling : bool, default True
            Whether or not to clip the soiling ratio at max 1 and minimum 0.
        verbose : bool, default False
            If true, prints a progress report
        ...

        Returns
        -------
        df_out : pandas.DataFrame
            Contains the estimated values of soiling ratio, soiling rates,
            seasonal component and degradation trend
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
            Whether or not the signal is deemed too small to infer soiling
            ratio
        adf_res : list
            The results of an Augmented Dickey-Fuller test (telling whether the
            residuals are stationary or not)
        ...

        References
        ----------
        [1] Jordan, D.C., Deline, C., Kurtz, S.R., Kimball, G.M., Anderson, M.,
            2017. Robust PV Degradation Methodology and Application. IEEE J.
            Photovoltaics 1–7. https://doi.org/10.1109/JPHOTOV.2017.2779779
        [2] Deceglie, M.G., Micheli, L., Muller, M., 2018. Quantifying Soiling
            Loss Directly from PV Yield. IEEE J. Photovoltaics 8, 547–551.
            https://doi.org/10.1109/JPHOTOV.2017.2784682
        [3] Skomedal, Å, Deceglie, M, 2020. ...
        '''
        pi = self.pm.copy()
        if degradation_method == 'STL' and 'Rd' in order:
            order.remove('Rd')

        if 'SR' not in order:
            raise ValueError('\'SR\' must be in argument \'order\' '
                             + '(e.g. order=[\'SR\', \'SC\', \'Rd\']')
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

        if not perfect_cleaning:
            change_point = 0

        # Find possible cleaning events based on the performance index
        ce, rm9 = rolling_median_ce_detection(pi.index, pi, ffill=ffill,
                                              tuner=detection_tuner)
        pce = collapse_cleaning_events(ce, rm9.diff().values, 5)

        small_soiling_signal = False
        ic = 0  # iteration counter

        if verbose:
            print('It. nr\tstep\tRMSE\ttimer')
        if verbose:
            print('{:}\t- \t{:.5f}'.format(ic, convergence_metric[ic]))
        while ic < max_iterations:
            t0 = time.time()
            ic += 1

            # Find soiling component
            if order[(ic-1) % n_steps] == 'SR':
                if ic > 2:  # Add possible cleaning events found by considering
                    # the residuals
                    pce = soiling_dfs[-1].cleaning_events.copy()
                    detection_tuner *= 1.2  # Increase value of detection tuner
                    ce, rm9 = rolling_median_ce_detection(
                        pi.index, residuals, ffill=ffill,
                        tuner=detection_tuner)
                    ce = collapse_cleaning_events(ce, rm9.diff().values, 5)
                    pce[ce] = True
                    pruning_tuner /= 1.1  # Decrease value of pruning tuner

                # Decompose input signal
                soiling_dummy = (pi
                                 / degradation_trend[-1]
                                 / seasonal_component[-1]
                                 / residual_shift)

                # Run Kalman Filter for obtaining soiling component
                kdf, Ps = self.Kalman_filter_for_SR(
                                zs_series=soiling_dummy,
                                clip_soiling=clip_soiling,
                                prescient_cleaning_events=pce,
                                pruning_iterations=pruning_iterations,
                                pruning_tuner=pruning_tuner,
                                perfect_cleaning=perfect_cleaning,
                                process_noise=process_noise,
                                renormalize_SR=renormalize_SR)
                soiling_ratio.append(kdf.soiling_ratio)
                soiling_dfs.append(kdf)

            # Find seasonal component
            if order[(ic-1) % n_steps] == 'SC':
                season_dummy = pi / soiling_ratio[-1]  # Decompose signal
                if season_dummy.isna().sum() > 0:
                    season_dummy.interpolate('linear', inplace=True)
                season_dummy = season_dummy.apply(np.log)  # Log transform
                # Run STL model
                STL_res = STL(season_dummy, period=365, seasonal=999999,
                              seasonal_deg=0, trend_deg=0,
                              robust=True, low_pass_jump=30, seasonal_jump=30,
                              trend_jump=365).fit()
                # Smooth result
                smooth_season = lowess(STL_res.seasonal.apply(np.exp),
                                       pi.index, is_sorted=True, delta=30,
                                       frac=180/len(pi), return_sorted=False)
                # Ensure periodic seaonal component
                seasonal_comp = force_periodicity(smooth_season,
                                                  season_dummy.index,
                                                  pi.index)
                seasonal_component.append(seasonal_comp)
                if degradation_method == 'STL':  # If not YoY
                    deg_trend = pd.Series(index=pi.index,
                                          data=STL_res.trend.apply(np.exp))
                    degradation_trend.append(deg_trend / deg_trend.iloc[0])
                    yoy_save.append(RdToolsDeg.naive_YOY(
                        degradation_trend[-1]))

            # Find degradation component
            if order[(ic-1) % n_steps] == 'Rd':
                # Decompose signal
                trend_dummy = (pi
                               / seasonal_component[-1]
                               / soiling_ratio[-1])
                yoy = RdToolsDeg.naive_YOY(trend_dummy)  # Run YoY
                # Convert degradation rate to trend
                degradation_trend.append(pd.Series(
                    index=pi.index, data=(1 + day * yoy / 100 / 365.24)))
                yoy_save.append(yoy)

            # Combine and calculate residual flatness
            total_model = (degradation_trend[-1]
                           * seasonal_component[-1]
                           * soiling_ratio[-1])
            residuals = pi / total_model
            residual_shift = residuals.mean()
            total_model *= residual_shift
            convergence_metric.append(_RMSE(pi, total_model))

            if verbose:
                print('{:}\t{:}\t{:.5f}\t\t\t{:.1f} s'.format(
                    ic, order[(ic-1) % n_steps], convergence_metric[-1],
                    time.time()-t0))

            # Convergence happens if there is no improvement in RMSE from one
            # step to the next
            if ic >= n_steps:
                relative_improvement = ((convergence_metric[-n_steps-1]
                                         - convergence_metric[-1])
                                        / convergence_metric[-n_steps-1])
                if perfect_cleaning and (
                        ic >= max_iterations / 2
                        or relative_improvement < convergence_criterium):
                    # From now on, do not assume perfect cleaning
                    perfect_cleaning = False
                    # Reorder to ensure SR first
                    order = [order[(i+n_steps-1-(ic-1) % n_steps) % n_steps]
                             for i in range(n_steps)]
                    change_point = ic
                    if verbose:
                        print('Now not assuming perfect cleaning')
                elif (not perfect_cleaning
                      and (ic >= max_iterations
                           or (ic >= change_point + n_steps
                               and relative_improvement
                               < convergence_criterium))):
                    if verbose:
                        if relative_improvement < convergence_criterium:
                            print('Convergence reached.')
                        else:
                            print('Max iterations reached.')
                    ic = max_iterations

        # Initialize output DataFrame
        df_out = pd.DataFrame(index=pi.index,
                              columns=['soiling_ratio', 'soiling_rates',
                                       'cleaning_events', 'seasonal_component',
                                       'degradation_trend', 'total_model',
                                       'residuals'])

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
        df_out.total_model = (df_out.soiling_ratio
                              * df_out.seasonal_component
                              * df_out.degradation_trend)
        df_out.residuals = pi / df_out.total_model
        residual_shift = df_out.residuals.mean()
        df_out.total_model *= residual_shift
        RMSE = _RMSE(pi, df_out.total_model)
        adf_res = adfuller(df_out.residuals.dropna(), regression='ctt')
        if verbose:
            print('p-value for the H0 that there is a unit root in the'
                  + 'residuals (using the Augmented Dickey-fuller test):'
                  + '{:.3e}'.format(adf_res[1]))

        # Check size of soiling signal vs residuals
        SR_amp = float(np.diff(df_out.soiling_ratio.quantile([.025, .975])))
        residuals_amp = float(np.diff(df_out.residuals.quantile([.025, .975])))
        soiling_signal_strength = SR_amp / residuals_amp
        if soiling_signal_strength < soiling_significance_knob:
            if verbose:
                print('Soiling signal is small relative to the noise')
            small_soiling_signal = True
            df_out.SR_high = 1.0
            df_out.SR_low = 1.0 - SR_amp

        return df_out, degradation, soiling_loss, residual_shift, RMSE, \
            small_soiling_signal, adf_res


    def run_bootstrap(self, bootstrap_nr=512, verbose=False,
                      degradation_method='YoY', process_noise=1e-4,
                      knob_alternatives=[[['SR', 'SC', 'Rd'],
                                          ['SC', 'SR', 'Rd']],
                                         [.4, .8],
                                         [.75, 1.25],
                                         [True, False]]):
        '''
        Boottrapping of iterative signal decomposition alforithm for
        uncertainty analysis.

        First, calls on iterative_signal_decomposition to fit N different
        models. Bootstrap samples are generated based on all of these models.
        Each bootstrap sample is generated by bootstrapping the residuals of
        the respective model (one of the N), using circular block
        bootstrapping, then multiplying these new residuals back onto the
        model. Then, for each bootstrap sample, one of the N models is randomly
        chosen and fit. The seasonal component is perturbed randomly and
        divided out, so as to capture its uncertainty. In the end, 95%
        confidence intervals are calulated based on the models fit to the
        bootrapped signals. The returned soiling ratio and rates are based on
        the best fit of the initial 16 models.

        Parameters
        ----------
        bootstrap_nr : int, default 512,
            Number of bootstrap realizations to be run
            minimum N, where N is the possible combinations of model
            knobs/parameters defined in knob_alternatives
        verbose : bool, default False
            Wheter or not to print information about progress
        degradation_method : string, default 'YoY'
            Either 'YoY' or 'STL'. If anything else, 'YoY' will be assumed.
            Decides whether to use the YoY method [3] for estimating the
            degradation trend (assumes linear trend), or the STL-method (does
            not assume linear trend). The latter is slower.
        knob_alternatives : list of lists, default [[['SR', 'SC', 'Rd'],
                                                     ['SC', 'SR', 'Rd']],
                                                    [.4, .8],
                                                    [.75, 1.25],
                                                    [True, False]]
            List of model knobs/parameters for the initial N model fits

        Returns
        -------
        df_out : pandas.DataFrame
            Contains the columns/keys
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
        knobs_n_weights : pandas.DataFrame
            Contains information about the knobs used in each bootstrap model
            fit, and the resultant weight
        '''
        pi = self.pm.copy()

        # Generate combinations of model knobs/parameters
        index_list = list(itertools.product(
                            [0, 1], repeat=len(knob_alternatives)))
        combination_of_knobs = [[knob_alternatives[j][indexes[j]]
                                 for j in range(len(knob_alternatives))]
                                for indexes in index_list]
        nr_models = len(index_list)
        bootstrap_samples_list, results = [], []

        # Check boostrap number
        if bootstrap_nr % nr_models != 0:
            bootstrap_nr += nr_models - bootstrap_nr % nr_models

        if verbose:
            print('Initially fitting {:} models'.format(nr_models))
        t00 = time.time()
        # For each combination of model knobs/parameters, fit one model:
        for c, (order, dt, pt, ff) in enumerate(combination_of_knobs):
            try:
                result = self.iterative_signal_decomposition(
                     max_iterations=18, order=order, clip_soiling=True,
                     detection_tuner=dt, pruning_iterations=1,
                     pruning_tuner=pt, process_noise=process_noise, ffill=ff,
                     degradation_method=degradation_method)

                # Save results
                results.append(result)
                adf = result[-1]
                # If we can reject the null-hypothesis that there is a unit
                # root in the residuals:
                if adf[1] < .05:
                    # ... generate bootstrap samples based on the fit:
                    bootstrap_samples_list.append(
                        make_bootstrap_samples(
                            pi, result[0].total_model,
                            sample_nr=int(bootstrap_nr / nr_models)))

                # Print progress
                if verbose:
                    progressBarWithETA(c+1, nr_models, time.time()-t00,
                                       bar_length=30)
            except ValueError as ex:
                print(ex)

        # Revive results
        adfs = np.array([(r[-1][0] if r[-1][1] < 0.05 else 0) for r in results])
        RMSEs = np.array([r[4] for r in results])
        SR_is_one_fraction = np.array(
            [(r[0].soiling_ratio == 1).mean() for r in results])
        sss = [r[5] for r in results]

        # Calculate weights
        weights = 1 / RMSEs / (.1 + SR_is_one_fraction)
        weights /= np.sum(weights)

        # Save knobs and weights for initial model fits
        knobs_n_weights = pd.concat([pd.DataFrame(combination_of_knobs),
                                     pd.Series(RMSEs),
                                     pd.Series(SR_is_one_fraction),
                                     pd.Series(weights),
                                     pd.Series(sss)],
                                    axis=1, ignore_index=True)

        if verbose:  # Print summary
            knobs_n_weights.columns = ['order', 'dt', 'pt', 'ff', 'RMSE',
                                       'SR==1', 'weights', 'sss']
            if verbose:
                print('\n', knobs_n_weights)

        # Check if data is decomposable
        if np.sum(adfs == 0) > nr_models / 2:
            self.errors = (
                'Test for stationary residuals (Augmented Dickey-Fuller'
                'test) not passed in half  of the instances:\nData not'
                ' decomposable.')
            print(self.errors)
            return

        # Save best model
        self.initial_fits = [r[0] for r in results]
        df_out = results[np.argmax(weights)][0]

        # If more than half of the model fits indicate small soiling signal,
        # don't do bootstrapping
        if np.sum(sss) > nr_models / 2:
            self.result_df = df_out
            self.residual_shift = results[np.argmax(weights)][3]
            YOY = RdToolsDeg.degradation_year_on_year(pi)
            self.degradation = [YOY[0], YOY[1][0], YOY[1][1]]
            self.soiling_loss = [0, 0, (1 - df_out.soiling_ratio).mean()]
            self.errors = (
                    'Soiling signal is small relative to the noise.'
                    'Iterative decomposition not possible.\n'
                    'Degradation found by RdTools YoY')
            print(self.errors)
            return

        # Aggregate all bootstrap samples
        all_bootstrap_samples = pd.concat(bootstrap_samples_list, axis=1,
                                          ignore_index=True)

        # Seasonal samples are generated from previously fitted seasonal
        # components, by perturbing amplitude and phase shift
        # Number of samples per fit:
        sample_nr = int(bootstrap_nr / nr_models)
        list_of_SCs = [results[m][0].seasonal_component
                       for m in range(nr_models) if weights[m] > 0]
        seasonal_samples = make_seasonal_samples(list_of_SCs,
                                                 sample_nr=sample_nr,
                                                 min_multiplier=.8,
                                                 max_multiplier=1.75,
                                                 max_shift=30)

        # Entering bootstrapping
        if verbose and bootstrap_nr > 0:
            print('\nBootstrapping for uncertainty analysis',
                  '({:} realizations):'.format(bootstrap_nr))
        order = ['SR', 'SC' if degradation_method == 'STL' else 'Rd']
        t0 = time.time()
        bt_kdfs, bt_SL, bt_deg, knobs, adfs, RMSEs, SR_is_1, rss, errors = \
            [], [], [], [], [], [], [], [], ['Bootstrapping errors']
        for b in range(bootstrap_nr):
            try:
                # randomly choose model knobs
                dt = np.random.uniform(knob_alternatives[1][0]*.75,
                                       knob_alternatives[1][1]*.75)
                pt = np.random.uniform(knob_alternatives[2][0]*1.5,
                                       knob_alternatives[2][1]*1.5)
                pn = np.random.uniform(process_noise / 1.5, process_noise * 1.5)
                renormalize_SR = np.random.choice([None,
                                                   np.random.uniform(.5, .95)])
                ffill = np.random.choice([True, False])
                knobs.append([dt, pt, pn, renormalize_SR, ffill])

                # Sample to infer soiling from
                bootstrap_sample = \
                    all_bootstrap_samples[b] / seasonal_samples[b]

                # Set up a temprary instance of the cods_analysis object
                temporary_cods_instance = cods_analysis(bootstrap_sample,
                                                        self.insol)
                # Do Signal decomposition for soiling and degradation component
                kdf, deg, SL, rs, RMSE, sss, adf = \
                    temporary_cods_instance.iterative_signal_decomposition(
                        max_iterations=4, order=order, clip_soiling=True,
                        detection_tuner=dt, pruning_iterations=1,
                        pruning_tuner=pt, process_noise=pn,
                        renormalize_SR=renormalize_SR, ffill=ffill,
                        degradation_method=degradation_method)

                # If we can reject the null-hypothesis that there is a unit
                # root in the residuals:
                if adf[1] < .05:  # Save the results
                    bt_kdfs.append(kdf)
                    adfs.append(adf[0])
                    RMSEs.append(RMSE)
                    bt_deg.append(deg)
                    bt_SL.append(SL)
                    rss.append(rs)
                    SR_is_1.append((kdf.soiling_ratio == 1).mean())
                else:
                    seasonal_samples.drop(columns=[b], inplace=True)

            except ValueError as ve:
                seasonal_samples.drop(columns=[b], inplace=True)
                errors.append(b, ve)

            # Print progress
            if verbose:
                progressBarWithETA(b+1, bootstrap_nr, time.time()-t0,
                                   bar_length=30)

        # Reweight and save weights
        weights = 1 / np.array(RMSEs) / (.1 + np.array(SR_is_1))
        weights /= np.sum(weights)
        self.knobs_n_weights = pd.concat(
            [pd.DataFrame(knobs),
             pd.Series(RMSEs),
             pd.Series(adfs),
             pd.Series(SR_is_1),
             pd.Series(weights)],
            axis=1, ignore_index=True)
        self.knobs_n_weights.columns = ['dt', 'pt', 'pn', 'RSR', 'ffill',
                                        'RMSE', 'ADF', 'SR==1', 'weights']

        # Concatenate boostrap model fits
        concat_tot_mod = pd.concat([kdf.total_model for kdf in bt_kdfs], 1)
        concat_SR = pd.concat([kdf.soiling_ratio for kdf in bt_kdfs], 1)
        concat_r_s = pd.concat([kdf.soiling_rates for kdf in bt_kdfs], 1)
        concat_ce = pd.concat([kdf.cleaning_events for kdf in bt_kdfs], 1)
        concat_deg = pd.concat([kdf.degradation_trend for kdf in bt_kdfs], 1)

        # Find confidence intervals for SR and soiling rates
        df_out['SR_low'] = concat_SR.quantile(.025, 1)
        df_out['SR_high'] = concat_SR.quantile(.975, 1)
        df_out['rates_low'] = concat_r_s.quantile(.025, 1)
        df_out['rates_high'] = concat_r_s.quantile(.975, 1)

        # Save best estimate and bootstrapped estimates of SR and soiling rates
        df_out.soiling_ratio = df_out.soiling_ratio.clip(lower=0, upper=1)
        df_out.loc[df_out.soiling_ratio.diff() == 0, 'soiling_rates'] = 0
        df_out['bt_soiling_ratio'] = (concat_SR * weights).sum(1)
        df_out['bt_soiling_rates'] = (concat_r_s * weights).sum(1)

        # Set probability of cleaning events
        df_out.cleaning_events = (concat_ce * weights).sum(1)

        # Find degradation rates
        self.degradation = [np.dot(bt_deg, weights),
                            np.quantile(bt_deg, .025),
                            np.quantile(bt_deg, .975)]
        df_out.degradation_trend = (concat_deg * weights).sum(1)
        df_out['degradation_low'] = concat_deg.quantile(.025, 1)
        df_out['degradation_high'] = concat_deg.quantile(.975, 1)

        # Soiling losses
        self.soiling_loss = [np.dot(bt_SL, weights),
                             np.quantile(bt_SL, .025),
                             np.quantile(bt_SL, .975)]

        # Save "confidence intervals" for seasonal component
        df_out.seasonal_component = (seasonal_samples * weights).sum(1)
        df_out['seasonal_low'] = seasonal_samples.quantile(.025, 1)
        df_out['seasonal_high'] = seasonal_samples.quantile(.975, 1)

        # Total model with confidence intervals
        df_out.total_model = (df_out.degradation_trend
                              * df_out.seasonal_component
                              * df_out.soiling_ratio)
        df_out['model_low'] = concat_tot_mod.quantile(.025, 1)
        df_out['model_high'] = concat_tot_mod.quantile(.975, 1)

        # Residuals and residual shift
        df_out.residuals = pi / df_out.total_model
        self.residual_shift = df_out.residuals.mean()
        df_out.total_model *= self.residual_shift
        self.RMSE = _RMSE(pi, df_out.total_model)
        self.adf_results = adfuller(df_out.residuals.dropna(),
                                    regression='ctt')
        self.result_df = df_out
        self.errors = errors

        if verbose:
            print('\nFinal RMSE: {:.5f}'.format(self.RMSE))
            if len(self.errors) > 1:
                print(self.errors)


    def Kalman_filter_for_SR(self, zs_series, process_noise=1e-4, zs_std=.05,
                             rate_std=.005, max_soiling_rates=.0005,
                             pruning_iterations=1, pruning_tuner=.6,
                             renormalize_SR=None, perfect_cleaning=False,
                             prescient_cleaning_events=None,
                             clip_soiling=True):
        '''
        A function for estimating the underlying Soiling Ratio (SR) and the
        rate of change of the SR (soiling rate), based on a noisy time series
        of SR using a Kalman Filter (KF).

        Parameters
        ----------
        zs_series: (pandas.Series) Time series of noisy SR-data
        window_size: (int)
        process_noise
        pi_std
        rate_std
        detection_tuner
        prescient_cleaning_events
        expected_max_soiling_period

        Returns
        -------
            - dfk: (pandas.DataFrame) results dataframe
            - Ps: (numpy.array) covariance matrix for the states of the KF
        '''

        # Ensure numeric index
        zs_series = zs_series.copy()  # Make copy, so as not to change input
        original_index = zs_series.index.copy()
        if (original_index.dtype not in [int, 'int64']):
            zs_series.index = range(len(zs_series))

        # Check prescient_cleaning_events. If not present, find cleaning events
        if type(prescient_cleaning_events) == list:
            cleaning_events = prescient_cleaning_events
        elif (isinstance(prescient_cleaning_events, type(zs_series))
              and np.sum(prescient_cleaning_events) > 4
              and (prescient_cleaning_events.index != zs_series.index).all()):
            prescient_cleaning_events = prescient_cleaning_events.copy()
            prescient_cleaning_events.index = zs_series.index
        else:  # If no prescient cleaning events, detect cleaning events
            ce, rm9 = rolling_median_ce_detection(zs_series.index,
                                                  zs_series,
                                                  tuner=0.5)
            prescient_cleaning_events = \
                collapse_cleaning_events(ce, rm9.diff().values, 5)
        cleaning_events = prescient_cleaning_events[prescient_cleaning_events
                                                    ].index.to_list()

        # Initialize various parameters
        rolling_median_13 = zs_series.ffill().rolling(13, center=True).median().ffill()
        # A rough estimate of the measurement noise
        measurement_noise = (rolling_median_13 - zs_series).var()
        # An initial guess of the slope
        initial_slope = np.array(theilslopes(zs_series.bfill().iloc[:14]))
        rolling_median_7 = zs_series.ffill().rolling(7, center=True).median().ffill()
        dt = 1  # All time stemps are one day

        # Initialize Kalman filter
        f = self.initialize_univariate_model(zs_series, dt, process_noise,
                                             measurement_noise, rate_std,
                                             zs_std, initial_slope)

        # Initialize miscallenous variables
        dfk = pd.DataFrame(index=zs_series.index, dtype=float,
                           columns=['raw_pi', 'raw_rates', 'smooth_pi',
                                    'smooth_rates', 'soiling_ratio',
                                    'soiling_rates', 'cleaning_events',
                                    'days_since_ce'])
        dfk['cleaning_events'] = False

        # Kalman Filter part:
        #######################################################################
        # Call the forward pass function (the actual KF procedure)
        Xs, Ps, rate_std, zs_std = self.forward_pass(f, zs_series,
                                                     rolling_median_7,
                                                     cleaning_events)

        # Save results and smooth with rts smoother
        dfk, Xs, Ps = self.smooth_results(dfk, f, Xs, Ps, zs_series,
                                          cleaning_events, perfect_cleaning)
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
                false_positives = find_numeric_outliers(pi_after_cleaning,
                                                        pruning_tuner, 'lower')
                cleaning_events = \
                    false_positives[~false_positives].index.to_list()

            # 2: Remove longer periods with positive (soiling) rates
            if (dfk.smooth_rates > max_soiling_rates).sum() > 1:
                exceeding_rates = dfk.smooth_rates > max_soiling_rates
                new_cleaning_events = collapse_cleaning_events(
                                        exceeding_rates, dfk.smooth_rates, 4)
                cleaning_events.extend(
                    new_cleaning_events[new_cleaning_events].index)
                cleaning_events.sort()

            # 3: If the list of cleaning events has changed, run the Kalman
            #    Filter and smoother again
            if not ce_0 == cleaning_events:
                f = self.initialize_univariate_model(zs_series, dt,
                                                     process_noise,
                                                     measurement_noise,
                                                     rate_std, zs_std,
                                                     initial_slope)
                Xs, Ps, rate_std, zs_std = self.forward_pass(f, zs_series,
                                                             rolling_median_7,
                                                             cleaning_events)
                dfk, Xs, Ps = self.smooth_results(dfk, f, Xs, Ps, zs_series,
                                                  cleaning_events,
                                                  perfect_cleaning)

            else:
                counter = 100  # Make sure the while loop stops

            # 4: Estimate Soiling ratio from kalman estimate
            if perfect_cleaning:  # SR = 1 after cleaning events
                if len(cleaning_events) > 0:
                    pi_dummy = pd.Series(index=dfk.index, data=np.nan)
                    pi_dummy.loc[cleaning_events] = \
                        dfk.smooth_pi.loc[cleaning_events]
                    dfk.soiling_ratio = 1 / pi_dummy.ffill() * dfk.smooth_pi
                    # Set the SR in the first soiling period based on the mean
                    # ratio of the Kalman estimate (smooth_pi) and the SR
                    dfk.loc[:cleaning_events[0], 'soiling_ratio'] = \
                        dfk.loc[:cleaning_events[0], 'smooth_pi'] \
                        * (dfk.soiling_ratio / dfk.smooth_pi).mean()
                else:  # If no cleaning events
                    dfk.soiling_ratio = 1
            else:  # Otherwise, if the inut signal has been decomposed, and
                # only contains a soiling component, the kalman estimate = SR
                dfk.soiling_ratio = dfk.smooth_pi
            # 5: Renormalize Soiling Ratio
            if renormalize_SR is not None:
                dfk.soiling_ratio /= dfk.loc[cleaning_events, 'soiling_ratio'
                                             ].quantile(renormalize_SR)

            # 6: Force soiling ratio to not exceed 1:
            if clip_soiling:
                dfk.soiling_ratio.clip(upper=1, inplace=True)
                dfk.soiling_rates = dfk.smooth_rates
                dfk.loc[dfk.soiling_ratio.diff() == 0, 'soiling_rates'] = 0

        # Set number of days since cleaning event
        nr_days_dummy = pd.Series(index=dfk.index, data=np.nan)
        nr_days_dummy.loc[cleaning_events] = [int(date-dfk.index[0])
                                              for date in cleaning_events]
        nr_days_dummy.iloc[0] = 0
        dfk.days_since_ce = range(len(zs_series)) - nr_days_dummy.ffill()

        # Save cleaning events and soiling events
        dfk.loc[cleaning_events, 'cleaning_events'] = True
        dfk.index = original_index  # Set index back to orignial index

        return dfk, Ps

    def forward_pass(self, f, zs_series, rolling_median_7, cleaning_events):
        ''' Run the forward pass of the Kalman Filter algortihm '''
        zs = zs_series.values
        N = len(zs)
        Xs, Ps = np.zeros((N, 2)), np.zeros((N, 2, 2))
        # Enter forward pass of filtering algorithm
        for i, z in enumerate(zs):
            if 7 < i < N-7 and i in cleaning_events:
                rolling_median_local = rolling_median_7.loc[i-5:i+5].values
                u = self.set_control_input(f, rolling_median_local, i,
                                           cleaning_events)
                f.predict(u=u)  # Predict wth control input u
            else:  # If no cleaning detection, predict without control input
                f.predict()
            if not np.isnan(z):
                f.update(z)  # Update

            Xs[i] = f.x
            Ps[i] = f.P
            rate_std, zs_std = Ps[-1, 1, 1], Ps[-1, 0, 0]
        return Xs, Ps, rate_std, zs_std  # Convert to numpy and return

    def set_control_input(self, f, rolling_median_local, index,
                          cleaning_events):
        '''
        For each cleaning event, sets control input u based on current
        Kalman Filter state estimate (f.x), and the median value for the
        following week. If the cleaning event seems to be misplaced, moves
        the cleaning event to a more sensible location. If the cleaning
        event seems to be correct, removes other cleaning events in the 10
        days surrounding this day
        '''
        u = np.zeros(f.x.shape)  # u is the control input
        window_size = 11  # len of rolling_median_local
        HW = 5  # Half window
        moving_diff = np.diff(rolling_median_local)
        # Index of maximum change in rolling median
        max_diff_index = moving_diff.argmax()
        if max_diff_index == HW-1:  # if the max difference is today
            # The median zs of the week after the cleaning event
            z_med = rolling_median_local[HW+3]
            # Set control input this future median
            u[0] = z_med - np.dot(f.H, np.dot(f.F, f.x))
            # If the change is bigger than the measurement noise:
            if u[0] > np.sqrt(f.R)/2:
                index_dummy = [n+3 for n in range(window_size-HW-1)
                               if n+3 != HW]
                cleaning_events = [ce for ce in cleaning_events
                                   if ce-index+HW not in index_dummy]
            else:  # If the cleaning event is insignificant
                u[0] = 0
                cleaning_events.remove(index)
        else:  # If the index with the maximum difference is not today...
            cleaning_events.remove(index)  # ...remove today from the list
            if moving_diff[max_diff_index] > 0 \
                    and index+max_diff_index-HW+1 not in cleaning_events:
                # ...and add the missing day
                bisect.insort(cleaning_events, index+max_diff_index-HW+1)
        return u

    def smooth_results(self, dfk, f, Xs, Ps, zs_series, cleaning_events,
                       perfect_cleaning):
        ''' Smoother for Kalman Filter estimates. Smooths the Kalaman estimate
            between given cleaning events and saves all in DataFrame dfk'''
        # Save unsmoothed estimates
        dfk.raw_pi = Xs[:, 0]
        dfk.raw_rates = Xs[:, 1]

        # Set up cleaning events dummy list, inlcuding first and last day
        df_num_ind = pd.Series(index=dfk.index, data=range(len(dfk)))
        ce_dummy = cleaning_events.copy()
        ce_dummy.extend(dfk.index[[0, -1]])
        ce_dummy.sort()

        # Smooth between cleaning events
        for start, end in zip(ce_dummy[:-1], ce_dummy[1:]):
            num_ind = df_num_ind.loc[start:end].iloc[:-1]
            Xs[num_ind], Ps[num_ind], _, _ = f.rts_smoother(Xs[num_ind],
                                                            Ps[num_ind])

        # Save smoothed estimates
        dfk.smooth_pi = Xs[:, 0]
        dfk.smooth_rates = Xs[:, 1]

        return dfk, Xs, Ps

    def initialize_univariate_model(self, zs_series, dt, process_noise,
                                    measurement_noise, rate_std, zs_std,
                                    initial_slope):
        ''' Initializes the univariate Kalman Filter model, using the filterpy
            package '''
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.F = np.array([[1., dt],
                        [0., 1.]])
        f.H = np.array([[1., 0.]])
        f.P = np.array([[zs_std**2, 0],
                        [0, rate_std**2]])
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise**2)
        f.x = np.array([initial_slope[1], initial_slope[0]])
        f.B = np.zeros(f.F.shape)
        f.B[0] = 1
        f.R = measurement_noise
        return f


def collapse_cleaning_events(inferred_ce_in, metric, f=4):
    ''' A function for replacing quick successive cleaning events with one
        (most probable) cleaning event.

    Parameters
    ----------
    inferred_ce_in : pandas.Series
        Contains daily booelan values for cleaning events
    metric : array/pandas.Series
        A metric which is large when probability of cleaning is large
        (eg. daily difference in rolling median of performance index)
    f : int, default 4
        Number of time stamps to collapse in each direction

    Returns
    -------
    inferred_ce : pandas.Series
        boolean values for cleaning events
    '''
    # Ensure numeric index
    if isinstance(inferred_ce_in.index,
                  pd.core.indexes.datetimes.DatetimeIndex):
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
            max_diff_day = \
                metric.loc[start_true_vals-f:end_true_vals+f].idxmax()
            # Set all days in this period as false
            collapsed_ce.loc[start_true_vals-f:end_true_vals+f] = False
            collapsed_ce_dummy.loc[start_true_vals-f:end_true_vals+f] = False
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


def rolling_median_ce_detection(x, y, ffill=True, rolling_window=9, tuner=1.5):
    ''' Finds cleaning events in a time series of performance index (y) '''
    y = pd.Series(index=x, data=y)
    if ffill:  # forward fill NaNs in y before running mean
        rm = y.ffill().rolling(rolling_window, center=True).median()
    else:  # ... or backfill instead
        rm = y.bfill().rolling(rolling_window, center=True).median()
    Q3 = rm.diff().abs().quantile(.75)
    Q1 = rm.diff().abs().quantile(.25)
    limit = Q3 + tuner*(Q3 - Q1)
    cleaning_events = rm.diff() > limit
    return cleaning_events, rm


def make_bootstrap_samples(pi, model, sample_nr=10):
    ''' Generate bootstrap samples based on a CODS model fit '''
    residuals = pi / model
    bs = CircularBlockBootstrap(180, residuals)
    bootstrap_samples = pd.DataFrame(index=model.index,
                                     columns=range(sample_nr))
    for b, bootstrapped_residuals in enumerate(bs.bootstrap(sample_nr)):
        bootstrap_samples.loc[:, b] = \
            model * bootstrapped_residuals[0][0].values
    return bootstrap_samples


def make_seasonal_samples(list_of_SCs, sample_nr=10, min_multiplier=0.5,
                          max_multiplier=2, max_shift=20):
    ''' Generate seasonal samples by perturbing the amplitude and the phase of
        a seasonal components found with the fitted CODS model '''
    samples = pd.DataFrame(index=list_of_SCs[0].index,
                           columns=range(int(sample_nr*len(list_of_SCs))))
    # From each fitted signal, we will generate new seaonal components
    for i, signal in enumerate(list_of_SCs):
        # Remove beginning and end of signal
        signal_mean = signal.mean()
        unique_years = signal.index.year.unique()  # Unique years
        # Make a signal matrix where each column is a year and each row a date
        year_matrix = pd.concat([pd.Series(signal.loc[str(year)].values)
                                 for year in unique_years],
                                axis=1, ignore_index=True)
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
                data=median_signal.reindex(
                    (signal.index.dayofyear-shift) % 365).values)
            # Perturb amplitude by recentering to 0 multiplying by multiplier
            samples.loc[:, i*sample_nr + j] = \
                multiplier * (shifted_signal - signal_mean) + 1
    return samples


def force_periodicity(in_signal, signal_index, out_index):
    ''' Function for forcing periodicity in a seasonal component signal '''
    # Make sure the in_signal is a Series
    if type(in_signal) == np.ndarray:
        signal = pd.Series(index=out_index, data=np.nan)
        signal.loc[signal_index] = in_signal
    else:
        signal = in_signal
    
    # Make sure that we don't remove too much of the data:
    remove_length = np.min([180, int((len(signal) - 365) / 2)])
    # Remove beginning and end of series
    signal.iloc[:remove_length] = np.nan
    signal.iloc[-remove_length:] = np.nan

    unique_years = signal.index.year.unique()  # Years involved in time series
    # Make a signal matrix where each column is a year and each row is a date
    year_matrix = pd.DataFrame(index=np.arange(0,365), columns=unique_years)
    for year in unique_years:
        dates_in_year = pd.date_range(str(year)+'-01-01', str(year)+'-12-31')
        # We cut off the extra day(s) of leap years
        year_matrix[year] = \
            signal.loc[str(year)].reindex(dates_in_year).values[:365]
    # We will use the median signal through all the years...
    median_signal = year_matrix.median(1)
    # The output is the median signal broadcasted to the whole time series
    output = pd.Series(
        index=signal.index,
        data=median_signal.reindex(signal.index.dayofyear-1).values)
    return output


def find_numeric_outliers(x, multiplier=1.5, where='both', verbose=False):
    ''' Function for finding numeric outliers '''
    try:  # Calulate third and first quartile
        Q3 = np.quantile(x, .75)
        Q1 = np.quantile(x, .25)
    except IndexError as ie:
        print(ie, x)
    except RuntimeWarning as rw:
        print(rw, x)
    IQR = Q3 - Q1  # Interquartile range
    if where == 'upper':  # If detecting upper outliers
        if verbose:
            print('Upper limit', Q3 + multiplier * IQR)
        return (x > Q3 + multiplier * IQR)
    elif where == 'lower':  # If detecting lower outliers
        if verbose:
            print('Lower limit', Q1 - multiplier * IQR)
        return (x < Q1 - multiplier * IQR)
    elif where == 'both':  # If detecting both lower and upper outliers
        if verbose:
            print('Upper, lower limit',
                  Q3 + multiplier * IQR,
                  Q1 - multiplier * IQR)
        return (x > Q3 + multiplier * IQR), (x < Q1 - multiplier * IQR)


def _RMSE(y_true, y_pred):
    '''Calculates the Root Mean Squared Error for y_true and y_pred, where
        y_pred is the "prediction", and y_true is the truth.'''
    mask = ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_pred[mask]-y_true[mask])**2))


def MSD(y_true, y_pred):
    '''Calculates the Mean Signed Deviation for y_true and y_pred, where y_pred
        is the "prediction", and y_true is the truth.'''
    return np.mean(y_pred - y_true)


def progressBarWithETA(value, endvalue, time, bar_length=20):
    ''' Prints a progressbar with an estimated time of "arrival" '''
    percent = float(value) / endvalue * 100
    arrow = '-' * int(round(percent/100 * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    used = time / 60  # Time Used
    left = used / percent*(100-percent)  # Estimated time left
    sys.stdout.write(
        "\r# {:} | Used: {:.1f} min | Left: {:.1f}".format(value, used, left) 
        + " min | Progress: [{:}] {:.0f} %".format(arrow + spaces, percent))
    sys.stdout.flush()
