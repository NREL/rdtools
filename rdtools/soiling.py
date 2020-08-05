'''Functions for calculating soiling metrics from photovoltaic system data.'''

import pandas as pd
import numpy as np
from scipy.stats.mstats import theilslopes


# Custom exception
class NoValidIntervalError(Exception):
    '''raised when no valid rows appear in the result dataframe'''
    pass


class SRRAnalysis():
    '''
    Class for running the stochastic rate and recovery (SRR) photovoltaic
    soiling loss analysis presented in Deceglie et al. JPV 8(2) p547 2018

    Parameters
    ----------
    energy_normalized_daily : pd.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Alternatively, the soiling ratio output of a soiling sensor (e.g. the
        photocurrent ratio between matched dirty and clean PV reference cells).
        In either case, data should be insolation-weighted daily aggregates.
    insolation_daily : pd.Series
        Daily plane-of-array insolation corresponding to
        `energy_normalized_daily`
    precipitation_daily : pd.Series, default None
        Daily total precipitation. (Ignored if ``clean_criterion='shift'`` in
        subsequent calculations.)
    '''

    def __init__(self, energy_normalized_daily, insolation_daily,
                 precipitation_daily=None):
        self.pm = energy_normalized_daily  # daily performance metric
        self.insolation_daily = insolation_daily
        self.precipitation_daily = precipitation_daily  # daily precipitation
        self.random_profiles = []  # random soiling profiles in _calc_monte
        # insolation-weighted soiling ratios in _calc_monte:
        self.monte_losses = []

        if self.pm.index.freq != 'D':
            raise ValueError('Daily performance metric series must have '
                             'daily frequency')

        if self.insolation_daily.index.freq != 'D':
            raise ValueError('Daily insolation series must have '
                             'daily frequency')

        if self.precipitation_daily is not None:
            if self.precipitation_daily.index.freq != 'D':
                raise ValueError('Precipitation series must have '
                                 'daily frequency')

    def _calc_daily_df(self, day_scale=14, clean_threshold='infer',
                       recenter=True, clean_criterion='shift', precip_threshold=0.01):
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
            If 'infer': automatically use outliers in the shift as the
            threshold
        recenter : bool, default True
            Whether to recenter (renormalize) the daily performance to the
            median of the first year
        clean_criterion : {'precip_and_shift', 'precip_or_shift', 'precip', 'shift'} \
                default 'shift'
            The method of partitioning the dataset into soiling intervals.
            If 'precip_and_shift', rolling median shifts must coincide
            with precipitation to be a valid cleaning event.
            If 'precip_or_shift', rolling median shifts and precipitation
            events are each sufficient on their own to be a cleaning event.
            If 'shift', only rolling median shifts are treated as cleaning events.
            If 'precip', only precipitation events are treated as cleaning events.
        precip_threshold : float, default 0.01
            The daily precipitation threshold for defining precipitation cleaning events.
            Units must be consistent with ``self.precipitation_daily``.
        '''

        df = self.pm.to_frame()
        df.columns = ['pi']
        df_insol = self.insolation_daily.to_frame()
        df_insol.columns = ['insol']

        df = df.join(df_insol)
        precip = self.precipitation_daily
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

        df['clean_event_detected'] = (df.delta > clean_threshold)
        precip_event = (df['precip'] > precip_threshold)

        if clean_criterion == 'precip_and_shift':
            # Detect which cleaning events are associated with rain within a 3 day window
            precip_event = precip_event.rolling(3, center=True, min_periods=1).apply(any).astype(bool)
            df['clean_event'] = (df['clean_event_detected'] & precip_event)
        elif clean_criterion == 'precip_or_shift':
            df['clean_event'] = (df['clean_event_detected'] | precip_event)
        elif clean_criterion == 'precip':
            df['clean_event'] = precip_event
        elif clean_criterion == 'shift':
            df['clean_event'] = df['clean_event_detected']
        else:
            raise ValueError('clean_criterion must be one of '
                             '{"precip_and_shift", "precip_or_shift", "precip", "shift"}')

        df['clean_event'] = df.clean_event | out_start | out_end
        df['clean_event'] = (df.clean_event) & (~df.clean_event.shift(-1).fillna(False))

        df = df.fillna(0)

        # Give an index to each soiling interval/run
        df['run'] = df.clean_event.cumsum()
        df.index.name = 'date'  # this gets used by name

        self.renorm_factor = renorm
        self.daily_df = df

    def _calc_result_df(self, trim=False, max_relative_slope_error=500.0,
                        max_negative_step=0.05, min_interval_length=2):
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
        min_interval_length : int, default 2
            The minimum duration for an interval to be considered
            valid.  Cannot be less than 2 (days).
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
            start = run.index[0]
            end = run.index[-1]
            run_filtered = run[run.pi_norm > 0]
            # use the filtered version if it contains any points
            # otherwise use the unfiltered version to populate a
            # valid=False row
            if not run_filtered.empty:
                run = run_filtered
            result_dict = {
                'start': start,
                'end': end,
                'length': length,
                'run': r,
                'run_slope': 0,
                'run_slope_low': 0,
                'run_slope_high': 0,
                'max_neg_step': min(run.delta),
                'start_loss': 1,
                'inferred_start_loss': run.pi_norm.mean(),
                'inferred_end_loss': run.pi_norm.mean(),
                'valid': False
            }
            if len(run) > min_interval_length and run.pi_norm.sum() > 0:
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
        results['slope_err'] = (results.run_slope_high-results.run_slope_low)/abs(results.run_slope)
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

    def _calc_monte(self, monte, method='half_norm_clean'):
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
        '''

        monte_losses = []
        random_profiles = []
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
                    inter_start = np.random.uniform(end, 1)
                results_rand['start_loss'] = start_list

            elif method == 'perfect_clean':
                for i, row in results_rand.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
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
            trim=False, method='half_norm_clean',
            clean_criterion='shift', precip_threshold=0.01, min_interval_length=2,
            exceedance_prob=95.0, confidence_level=68.2, recenter=True,
            max_relative_slope_error=500.0, max_negative_step=0.05):
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

        clean_criterion : {'precip_and_shift', 'precip_or_shift', 'precip', 'shift'} \
                default 'shift'
            The method of partitioning the dataset into soiling intervals.
            If 'precip_and_shift', rolling median shifts must coincide
            with precipitation to be a valid cleaning event.
            If 'precip_or_shift', rolling median shifts and precipitation
            events are each sufficient on their own to be a cleaning event.
            If 'shift', only rolling median shifts are treated as cleaning events.
            If 'precip', only precipitation events are treated as cleaning events.
        precip_threshold : float, default 0.01
            The daily precipitation threshold for defining precipitation cleaning events.
            Units must be consistent with ``self.precipitation_daily``
        min_interval_length : int, default 2
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
                            recenter=recenter,
                            clean_criterion=clean_criterion,
                            precip_threshold=precip_threshold)
        self._calc_result_df(trim=trim,
                             max_relative_slope_error=max_relative_slope_error,
                             max_negative_step=max_negative_step,
                             min_interval_length=min_interval_length)
        self._calc_monte(reps, method=method)

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


def soiling_srr(energy_normalized_daily, insolation_daily, reps=1000,
                precipitation_daily=None, day_scale=14, clean_threshold='infer',
                trim=False, method='half_norm_clean',
                clean_criterion='shift', precip_threshold=0.01, min_interval_length=2,
                exceedance_prob=95.0, confidence_level=68.2, recenter=True,
                max_relative_slope_error=500.0, max_negative_step=0.05):
    '''
    Functional wrapper for :py:class:`~rdtools.soiling.SRRAnalysis`. Perform
    the stochastic rate and recovery soiling loss calculation. Based on the
    methods presented in Deceglie et al. JPV 8(2) p547 2018.

    Parameters
    ----------
    energy_normalized_daily : pd.Series
        Daily performance metric (i.e. performance index, yield, etc.)
        Alternatively, the soiling ratio output of a soiling sensor (e.g. the
        photocurrent ratio between matched dirty and clean PV reference cells).
        In either case, data should be insolation-weighted daily aggregates.
    insolation_daily : pd.Series
        Daily plane-of-array insolation corresponding to
        `energy_normalized_daily`
    reps : int, default 1000
        number of Monte Carlo realizations to calculate
    precipitation_daily : pd.Series, default None
        Daily total precipitation. Units ambiguous but should be the same as
        precip_threshold. Note default behavior of precip_threshold. (Ignored
        if ``clean_criterion='shift'``.)
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
    clean_criterion : {'precip_and_shift', 'precip_or_shift', 'precip', 'shift'} \
                default 'shift'
            The method of partitioning the dataset into soiling intervals.
            If 'precip_and_shift', rolling median shifts must coincide
            with precipitation to be a valid cleaning event.
            If 'precip_or_shift', rolling median shifts and precipitation
            events are each sufficient on their own to be a cleaning event.
            If 'shift', only rolling median shifts are treated as cleaning events.
            If 'precip', only precipitation events are treated as cleaning events.
    precip_threshold : float, default 0.01
        The daily precipitation threshold for defining precipitation cleaning events.
        Units must be consistent with precip.
    min_interval_length : int, default 2
        The minimum duration for an interval to be considered
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

    srr = SRRAnalysis(energy_normalized_daily,
                      insolation_daily,
                      precipitation_daily=precipitation_daily)

    sr, sr_ci, soiling_info = srr.run(
        reps=reps,
        day_scale=day_scale,
        clean_threshold=clean_threshold,
        trim=trim,
        method=method,
        clean_criterion=clean_criterion,
        precip_threshold=precip_threshold,
        exceedance_prob=exceedance_prob,
        confidence_level=confidence_level,
        recenter=recenter,
        max_relative_slope_error=max_relative_slope_error,
        max_negative_step=max_negative_step)

    return sr, sr_ci, soiling_info
