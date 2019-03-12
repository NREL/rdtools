''' Soiling Module

This module contains functions to calculate soiling
metrics from photovoltaic system data.
'''

import pandas as pd
import numpy as np
from scipy.stats.mstats import theilslopes


# Custom exception
class NoValidIntervalError(Exception):
    '''raised when no valid rows appear in the result grame'''
    pass


class pm_frame(pd.DataFrame):
    '''Class consisting of dataframe for analysis constructed from system data, usually created with create_pm_frame'''

    def calc_result_frame(self, trim=True):
        '''Return a result_frame

        Returns a result_frame which contains the charecteristics of each soiling interval.soiling.
        An updated version of the pm_frame is stored as self.pm_frame.

        Parameters
        ----------
        trim (bolean): whether to trim (remove) the first and last soiling intervals to avoid inclusion of partial intervals

        '''

        # Estimate slope of each soiling interval, store results in a dataframe
        result_list = []
        if trim:
            res_loop = sorted(list(set(self['run'])))[1:-1]  # ignore first and last interval
        else:
            res_loop = sorted(list(set(self['run'])))

        for r in res_loop:
            run = self[self.run == r]
            length = (run.day[-1] - run.day[0])
            start_day = run.day[0]
            end_day = run.day[-1]
            run = run[run.pi_norm > 0]
            if len(run) > 2 and run.pi_norm.sum() > 0:
                fit = theilslopes(run.pi_norm, run.day)
                fit_poly = np.poly1d(fit[0:2])
                result_list.append({
                    'start': run.index[0],
                    'end': run.index[-1],
                    'length': length,
                    'run': r,
                    'run_slope': fit[0],
                    'run_slope_low': fit[2],
                    'run_slope_high': min([0.0, fit[3]]),
                    'max_neg_step': min(run.delta),
                    'start_loss': 1,
                    'clean_wo_precip': run.clean_wo_precip[0],
                    'inferred_start_loss': fit_poly(start_day),
                    'inferred_end_loss': fit_poly(end_day),
                    'valid': True
                })
            else:
                run = self[self.run == r]
                result_list.append({
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
                })
        results = pd.DataFrame(result_list)

        if results.empty:
            raise NoValidIntervalError('No valid soiling intervals were found')

        # Filter results for each interval setting invalid interval to slope of 0
        results['slope_err'] = (results.run_slope_high - results.run_slope_low) / abs(results.run_slope)
        # critera for exclusions
        filt = (
            (results.run_slope > 0) |
            (results.slope_err > 5) |
            (results.max_neg_step <= -0.05)
        )

        results.loc[filt, 'run_slope'] = 0
        results.loc[filt, 'run_slope_low'] = 0
        results.loc[filt, 'run_slope_high'] = 0
        results.loc[filt, 'valid'] = False

        # Calculate the next inferred start loss from next valid interval
        results['next_inferred_start_loss'] = np.clip(results[results.valid].inferred_start_loss.shift(-1), 0, 1)
        # Calculate the inferred recovery at the end of each interval
        results['inferred_recovery'] = np.clip(results.next_inferred_start_loss - results.inferred_end_loss, 0, 1)

        # Don't consider data outside of first and last valid interverals
        if len(results[results.valid]) == 0:
            raise NoValidIntervalError('No valid soiling intervals were found')
        new_start = results[results.valid].start.iloc[0]
        new_end = results[results.valid].end.iloc[-1]
        pm_frame_out = self[new_start:new_end]
        pm_frame_out = pm_frame_out.reset_index().merge(results, how='left', on='run').set_index('date')

        pm_frame_out['loss_perfect_clean'] = np.nan
        pm_frame_out['loss_inferred_clean'] = np.nan
        pm_frame_out['days_since_clean'] = (pm_frame_out.index - pm_frame_out.start).dt.days

        # Caluclate the daily derate
        pm_frame_out['loss_perfect_clean'] = pm_frame_out.start_loss + pm_frame_out.days_since_clean * pm_frame_out.run_slope
        pm_frame_out.loss_perfect_clean = pm_frame_out.loss_perfect_clean.fillna(1)  # filling the flat intervals may need to be recalculated for different assumptions
        pm_frame_out['loss_inferred_clean'] = pm_frame_out.inferred_start_loss + pm_frame_out.days_since_clean * pm_frame_out.run_slope
        pm_frame_out.loss_inferred_clean = pm_frame_out.loss_inferred_clean.fillna(1)  # filling the flat intervals may need to be recalculated for different assumptions

        out = result_frame(results)
        out.pm_frame = pm_frame_out

        return out


class result_frame(pd.DataFrame):
    '''Class consisting of dataframe for calculaitng losses, typically created from pm_frame.calc_result_frame()'''

    # Add  normal properties
    _metadata = ['pm_frame', 'randomized_loss']

    @property
    def _constructor(self):
        return result_frame

    def calc_monte(self, monte, method='infer_clean', precip_clean_only=False):
        '''Return monte carlo sample of losses

        Parameters
        ----------
        monte (int): number of monte carlo simulations to run

        method (str): how to treat the recovery of each cleaning event
                        'random_clean' - a random recovery between 0-100%
                        'perfect_clean' - each cleaning event returns the performance metric to 1
                        'infer_clean' (default) - The three-sigma lower bound of recovery is inferred from the fit
                        of the following interval, the upper bound is 1 with the magnitude drawn from a half normal centered at 1

        precip_clean_only(bool): If True, only consider cleaning events valid if they coincide with precipitation events

        Returns
        -------
        (list): Monte Carlo sample, of length monte, of expected irradiance-weighted soiling ratio
        '''

        monte_losses = []
        for _ in range(monte):
            results_rand = self.copy()
            df_rand = self.pm_frame.copy()
            df_rand = df_rand[['insol', 'run']]  # only really need this column from the original frame
            results_rand['run_slope'] = np.random.uniform(results_rand.run_slope_low, results_rand.run_slope_high)
            results_rand['run_loss'] = results_rand.run_slope * results_rand.length

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
            if method == 'infer_clean':
                # Randomize recovery of valid intervals only
                valid_intervals = results_rand[results_rand.valid].copy()
                valid_intervals['inferred_recovery'] = valid_intervals.inferred_recovery.fillna(1.0)
                inter_start = 1.0
                start_list = []
                end_list = []
                for i, row in valid_intervals.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    end_list.append(end)

                    if row.clean_wo_precip and precip_clean_only:
                        inter_start = end  # don't allow recovery if there was no precipitation
                    else:
                        # Use a half normal with the infered clean at the 3sigma point
                        x = np.clip(end + row.inferred_recovery, 0, 1)
                        inter_start = 1 - abs(np.random.normal(0.0, (1 - x) / 3))

                # Update the valid rows in results_rand
                valid_update = pd.DataFrame()
                valid_update['start_loss'] = start_list
                valid_update['end_loss'] = end_list
                valid_update.index = valid_intervals.index
                results_rand.update(valid_update)

                # forward and back fill to note the limits of random constant
                # derate for invalid intervals
                results_rand['previous_end'] = results_rand.end_loss.fillna(method='ffill')
                results_rand['next_start'] = results_rand.start_loss.fillna(method='bfill')

                # Randomly select random constant derate for invalid intervals
                # based on previous end and next beginning
                invalid_intervals = results_rand[~results_rand.valid].copy()
                # fill NaNs at beggining and end
                invalid_intervals.previous_end.fillna(1.0, inplace=True)
                invalid_intervals.next_start.fillna(1.0, inplace=True)
                groups = set(invalid_intervals.group)
                replace_levels = []

                results_rand.to_csv('temp.csv')

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

            else:
                inter_start = 1
                start_list = []
                for i, row in results_rand.iterrows():
                    start_list.append(inter_start)
                    end = inter_start + row.run_loss
                    if method == 'random_clean':
                        if row.clean_wo_precip and precip_clean_only:
                            inter_start = end  # don't allow recovery if there was no precipitation
                        else:
                            inter_start = np.random.uniform(end, 1)
                    elif method == 'perfect_clean':
                        if row.clean_wo_precip and precip_clean_only:
                            inter_start = end  # don't allow recovery if there was no precipitation
                        else:
                            inter_start = 1
                    elif method == 'infer_clean':
                        if row.clean_wo_precip and precip_clean_only:
                            inter_start = end  # don't allow recovery if there was no precipitation
                        else:
                            inter_start = np.random.uniform(np.clip(end + row.inferred_recovery, 0, 1), 1)
                    else:
                        raise(ValueError("Invalid method specification"))
                results_rand['start_loss'] = start_list

            df_rand = df_rand.reset_index().merge(results_rand, how='left', on='run').set_index('date')
            df_rand['loss'] = np.nan
            df_rand['days_since_clean'] = (df_rand.index - df_rand.start).dt.days
            df_rand['loss'] = df_rand.start_loss + df_rand.days_since_clean * df_rand.run_slope

            df_rand['soil_insol'] = df_rand.loss * df_rand.insol

            monte_losses.append(df_rand.soil_insol.sum() / df_rand.insol.sum())

        self.randomized_loss = df_rand  # Keep the last random loss frame
        return monte_losses


def create_pm_frame(pm, insol, precip=None, day_scale=14, clean_threshold='infer'):
    '''Return a pm_frame based on supplied perfromance metric and insolation

    Parameters
    ----------
    pm (pandas timeseries): Daily performance metric (i.e. performance index, yied, etc.)

    insol (pandas timeseries): Daily insolation

    precip (pandas timeseries): Daily total precipitation

    day_scale (int) : The number of days to use in rolling median for cleaning detection

    clean_threshold (float or str): The fractional positive shift in rolling median for cleaning detection.
                                    Or specify 'infer' to automatically use outliers in the shift as the threshold

    Returns
    -------
    (pm_frame)

    '''
    if pm.index.freq != 'D':
        raise ValueError('Daily performance metric series must have daily frequency')

    if insol.index.freq != 'D':
        raise ValueError('Daily insolation series must have daily frequency')

    if precip is not None:
        if pm.index.freq != 'D':
            raise ValueError('Precipitation series must have daily frequency')

    pm.name = 'pi'
    insol.name = 'insol'

    df = pm.to_frame()
    df_insol = insol.to_frame()

    df = df.join(df_insol)

    if precip is not None:
        precip.name = 'precip'
        df_precip = precip.to_frame()
        df = df.join(df_precip)
    else:
        df['precip'] = 0

    # find first and last valid data point
    start = df[~df.pi.isnull()].index[0]
    end = df[~df.pi.isnull()].index[-1]
    df = df[start:end]

    # create a day count column
    df['day'] = range(len(df))

    # Normalize pi to 95th percentile
    pi = df[df.pi > 0]['pi']
    df['pi_norm'] = df.pi / np.percentile(pi, 95)

    # Find the beginning and ends of outtages longer than dayscale
    out_start = (~df.pi_norm.isnull() & df.pi_norm.fillna(method='bfill', limit=day_scale).shift(-1).isnull())
    out_end = (~df.pi_norm.isnull() & df.pi_norm.fillna(method='ffill', limit=day_scale).shift(1).isnull())

    # clean up the first and last elements
    out_start.iloc[-1] = False
    out_end.iloc[0] = False

    # Make a forward filled copy, just for use in step, slope change detection
    df_ffill = df.fillna(method='ffill', limit=day_scale).copy()

    # Calculate rolling median
    df['pi_roll_med'] = df_ffill.pi_norm.rolling(day_scale, center=True).median()

    # Detect steps in rolling median
    df['delta'] = df.pi_roll_med.diff()
    if clean_threshold == 'infer':
        deltas = abs(df.delta)
        clean_threshold = deltas.quantile(0.75) + 1.5 * (deltas.quantile(0.75) - deltas.quantile(0.25))

    df['clean_event'] = (df.delta > clean_threshold)
    df['clean_event'] = df.clean_event | out_start | out_end
    df['clean_event'] = (df.clean_event) & (~df.clean_event.shift(-1).fillna(False))

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

    df.index.name = 'date'  # this gets used by name in calc_result_frame
    df.index = df.index.tz_localize(None)

    return pm_frame(df)


def soiling_srr(daily_normalized_energy, daily_insolation, reps=1000,
                precip=None, day_scale=14, clean_threshold='infer',
                trim=False, method='infer_clean', precip_clean_only=False):
    '''

    '''

    # create the performance metric dataframe
    pm_frame = create_pm_frame(daily_normalized_energy, daily_insolation, precip=precip,
                               day_scale=day_scale, clean_threshold=clean_threshold)

    # Then calculate a results frame summarizing the soiling intervals
    results = pm_frame.calc_result_frame(trim=trim)

    # perform the monte carlo simulations
    soiling_ratio_realizations = results.calc_monte(reps, method='infer_clean', precip_clean_only=False)

    # Calculate the P50 and confidence interval
    result = np.percentile(soiling_ratio_realizations, [50, 2.5, 97.5])

    return (result[0], result[1:3])
