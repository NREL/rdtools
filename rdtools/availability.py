"""
Functions for detecting and quantifying production loss from photovoltaic
system downtime events.

The availability module is currently experimental. The API, results,
and default behaviors may change in future releases (including MINOR
and PATCH releases) as the code matures.
"""

import rdtools

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings

warnings.warn(
    'The availability module is currently experimental. The API, results, '
    'and default behaviors may change in future releases (including MINOR '
    'and PATCH releases) as the code matures.'
)


class AvailabilityAnalysis:
    """
    A class to perform system availability and loss analysis.

    This class follows the analysis procedure described in [1]_, and
    implements two distinct algorithms. One for partial (subsystem) outages
    and one for system-wide outages. The :py:meth:`.AvailabilityAnalysis.run()`
    method executes both algorithms and combines their results.

    The input timeseries don't need to be in any particular set of units as
    long as all power and energy units are consistent, with energy units
    being the hourly-integrated power (e.g., kW and kWh). The units of the
    analysis outputs will match the inputs.

    Parameters
    ----------
    power_system : pd.Series
        Timeseries total system power. In the typical case, this is meter
        power data. Should be a right-labeled interval average (this is what
        is typically recorded in many DAS).

    power_subsystem : pd.DataFrame
        Timeseries power data, one column per subsystem. In the typical case,
        this is inverter AC power data. Each column is assumed to represent
        a subsystem, so no extra columns may be included. The index must
        match ``power_system``. Should be a right-labeled interval average.

    energy_cumulative : pd.Series
        Timeseries cumulative energy data for the entire system (e.g. meter).
        These values must be recorded at the device itself (rather than summed
        by a downstream device like a datalogger or DAS provider) to preserve
        its integrity across communication interruptions. Units must match
        ``power`` integrated to hourly energy (e.g. if ``power`` is in kW then
        ``energy`` must be in kWh).

    power_expected : pd.Series
        Expected system power data with the same index as the measured data.
        This can be modeled from on-site weather measurements if instruments
        are well calibrated and there is no risk of data gaps. However, because
        full system outages often cause weather data to be lost as well, it may
        be more useful to use data from an independent weather station or
        satellite-based weather provider. Should be a right-labeled interval
        average.

    Attributes
    ----------
    results : pd.DataFrame
        Rolled-up production, loss, and availability metrics. The index is
        a datetime index of the period passed to
        :py:meth:`AvailabilityAnalysis.run`. The columns of the dataframe are
        as follows:

        +----------------------+----------------------------------------------+
        | Column Name          | Description                                  |
        +======================+==============================================+
        | 'lost_production'    | Production loss from outages. Units match the|
        |                      | input power units (e.g. if power is given in |
        |                      | kW, 'lost_production' will be in kWh).       |
        +----------------------+----------------------------------------------+
        | 'actual_production'  | System energy production. Same units as      |
        |                      | 'lost_production'.                           |
        +----------------------+----------------------------------------------+
        | 'availability'       | Energy-weighted system availability as a     |
        |                      | fraction (0-1).                              |
        +----------------------+----------------------------------------------+

    loss_system : pd.Series
        Estimated timeseries lost power from system outages.

    loss_subsystem : pd.Series
        Estimated timeseries lost power from subsystem outages.

    loss_total : pd.Series
        Estimated total lost power from outages.

    reporting_mask : pd.DataFrame
        Boolean mask indicating whether subsystems appear online or not.

    power_expected_rescaled : pd.Series
        Expected power rescaled to better match system power during periods
        where the system is performing normally.

    energy_expected_rescaled : pd.Series
        Interval expected energy calculated from `power_expected_rescaled`.

    energy_cumulative_corrected : pd.Series
        Cumulative system production after filling in data gaps from outages
        with estimated production.

    error_info : pd.DataFrame
        Records about the error between expected power and actual power.

    interp_lower, interp_upper : function
        Functions to estimate the uncertainty interval bounds of expected
        production based on outage length.

    outage_info : pd.DataFrame
        Records about each detected system outage, one row per
        outage. The primary columns of interest are ``type``, which can be
        either ``'real'`` or ``'comms'`` and reports whether the outage
        was determined to be a real outage with lost production or just a
        communications interruption with no production impact; and ``loss``
        which reports the estimated production loss for the outage. The
        columns are as follows:

        +----------------------+----------------------------------------------+
        | Column Name          | Description                                  |
        +======================+==============================================+
        | 'start'              | Timestamp of the outage start.               |
        +----------------------+----------------------------------------------+
        | 'end'                | Timestamp of the outage end.                 |
        +----------------------+----------------------------------------------+
        | 'duration'           | Length of the outage (*i.e.*                 |
        |                      | ``outage_info['end'] - outage_info['start']``|
        |                      | ).                                           |
        +----------------------+----------------------------------------------+
        | 'intervals'          | Total count of data intervals contained in   |
        |                      | the outage.                                  |
        +----------------------+----------------------------------------------+
        | 'daylight_intervals' | Count of data intervals contained in the     |
        |                      | outage occurring during the day.             |
        +----------------------+----------------------------------------------+
        | 'error_lower'        | Lower error bound as a fraction of expected  |
        |                      | energy.                                      |
        +----------------------+----------------------------------------------+
        | 'error_upper'        | Upper error bound as a fraction of expected  |
        |                      | energy.                                      |
        +----------------------+----------------------------------------------+
        | 'energy_expected'    | Total expected production for the outage     |
        |                      | duration.                                    |
        +----------------------+----------------------------------------------+
        | 'energy_start'       | System cumulative production at the outage   |
        |                      | start.                                       |
        +----------------------+----------------------------------------------+
        | 'energy_end'         | System cumulative production at the outage   |
        |                      | end.                                         |
        +----------------------+----------------------------------------------+
        | 'energy_actual'      | System production during the outage (*i.e.*, |
        |                      | ``outage_info['energy_end'] -                |
        |                      | outage_info['energy_start']``).              |
        +----------------------+----------------------------------------------+
        | 'ci_lower'           | Lower bound for the expected energy          |
        |                      | confidence interval.                         |
        +----------------------+----------------------------------------------+
        | 'ci_upper'           | Lower bound for the expected energy          |
        |                      | confidence interval.                         |
        +----------------------+----------------------------------------------+
        | 'type'               | Type of the outage ('real or 'comms').       |
        +----------------------+----------------------------------------------+
        | 'loss'               | Estimated production loss.                   |
        +----------------------+----------------------------------------------+

    Notes
    -----
    This class's ability to detect short-duration outages is limited by
    the resolution of the system data. For instance, 15-minute averages
    would not be able to resolve the rapid power cycling of an intermittent
    inverter. Additionally, the loss at the edges of an outage may be
    underestimated because of masking by the interval averages.

    This class expects outages to be represented in the timeseries by NaN,
    zero, or very low values. If your DAS does not record data from outages
    (e.g., a three-hour outage results in three hours of omitted timestamps),
    you should insert those missing rows before using this analysis.

    References
    ----------
    .. [1] Anderson K. and Blumenthal R. "Overcoming communications outages in
       inverter downtime analysis", 2020 IEEE 47th Photovoltaic Specialists
       Conference (PVSC).
    """

    def __init__(self, power_system, power_subsystem, energy_cumulative,
                 power_expected):
        for series in [power_subsystem, energy_cumulative, power_expected]:
            if not power_system.index.equals(series.index):
                raise ValueError("Input timeseries indexes must match")

        self.power_system = power_system
        self.power_subsystem = power_subsystem
        self.energy_cumulative = energy_cumulative
        self.power_expected = power_expected

    def _calc_loss_subsystem(self, low_threshold, relative_sizes,
                             power_system_limit):
        """
        Estimate timeseries production loss from subsystem downtime events.

        This implements the "power comparison" method from [1]_ of comparing
        subsystem power data to total system power (e.g. inverter power to
        meter power).

        Because this method is based on peer-to-peer comparison at each
        timestamp, it is not suitable for full system outages (i.e., at least
        one inverter must be reporting along with the system meter).

        Sets the `reporting_mask` and `loss_subsystem` attributes.

        Parameters
        ----------
        low_threshold : float or pd.Series
            An optional threshold used to naively classify subsystems as
            online. If the threshold is a scalar, it will be used for all
            subsystems. For subsystems with different capacities, a pandas
            Series may be passed with index values matching the columns in
            ``power_subsystem``. Units must match ``power_subsystem`` and
            ``power_system``. If omitted, the limit is calculated for each
            subsystem independently as 0.001 times the 99th percentile of its
            power data.

        relative_sizes : dict or pd.Series
            The production capacity of each subsystem, normalized by the mean
            subsystem capacity. If not specified, it will be estimated from
            power data.

        power_system_limit : float or pd.Series, optional
            Maximum allowable system power. This parameter is used to account
            for cases where online subsystems can partially mitigate the loss
            of an offline subsystem, for example a system with a plant
            controller and dynamic inverter setpoints. This constraint is
            only applied to the subsystem loss calculation.
        """
        power_subsystem = self.power_subsystem
        power_system = self.power_system
        power_subsystem = power_subsystem.fillna(0)
        power_system = power_system.clip(lower=0)

        # Part A
        if low_threshold is None:
            # calculate the low-power threshold based on the upper edge of the
            # power distribution so that low-power strangeness (snow cover,
            # outages, shading etc) don't affect the estimate:
            low_threshold = power_subsystem.quantile(0.99) / 1000

        self.reporting_mask = looks_online = power_subsystem > low_threshold
        reporting = power_subsystem[looks_online]
        if relative_sizes is None:
            # normalize by mean power and take the median across the timeseries
            normalized = reporting.divide(reporting.mean(axis=1), axis=0)
            relative_sizes = normalized.median()
        else:
            # convert dict to Series (no effect if already Series)
            relative_sizes = pd.Series(relative_sizes)

        normalized_subsystem_powers = reporting.divide(relative_sizes, axis=1)
        mean_subsystem_power = normalized_subsystem_powers.mean(axis=1)

        virtual_full_power = mean_subsystem_power * power_subsystem.shape[1]

        system_delta = 1 - power_system / virtual_full_power

        subsystem_fraction = relative_sizes / relative_sizes.sum()
        smallest_delta = power_subsystem.le(low_threshold) \
                                        .replace(False, np.nan) \
                                        .multiply(subsystem_fraction) \
                                        .min(axis=1) \
                                        .fillna(1)  # use safe value of 100%
        is_downtime = system_delta > (0.75 * smallest_delta)
        is_downtime[looks_online.all(axis=1)] = False

        # Part B
        lowest_possible = looks_online.multiply(subsystem_fraction).sum(axis=1)
        f_online = power_system / virtual_full_power
        f_online = f_online.clip(lower=lowest_possible, upper=1)
        p_loss = (1 - f_online) / f_online * power_system
        p_loss[~is_downtime] = 0

        if power_system_limit is not None:
            limit_exceeded = p_loss + power_system > power_system_limit
            loss = power_system_limit - power_system[limit_exceeded]
            p_loss.loc[limit_exceeded] = loss.clip(lower=0)

        self.loss_subsystem = p_loss.fillna(0)

    def _calc_error_distributions(self, quantiles):
        """
        Calculate the error distributions of Section II-A in [1]_.

        Sets the `power_expected_rescaled`, `energy_expected_rescaled`,
        `error_info`, `interp_lower`, and `interp_upper` attributes.

        Parameters
        ----------
        quantiles : 2-element tuple, default (0.01, 0.99)
            The quantiles of the error distribution used for the expected
            energy confidence interval. The lower bound is used to classify
            outages as either (1) a simple communication interruption with
            no production loss or (2) a power outage with an associated
            production loss estimate.
        """
        df = pd.DataFrame({
            'Meter_kW': self.power_system,
            'Expected Power': self.power_expected,
            'Meter_kWh': self.energy_cumulative,
        })

        system_performing_normally = (
            (self.loss_subsystem == 0) & (self.power_system > 0)
        )
        # filter out nighttime as well, since night intervals shouldn't count
        subset = system_performing_normally & (df['Expected Power'] > 0)

        # rescale expected energy to better match actual production.
        # this shifts the error distributions so that as interval length
        # increases, error -> 0
        scaling_subset = df.loc[subset, ['Expected Power', 'Meter_kW']].sum()
        scaling_factor = (
            scaling_subset['Expected Power'] / scaling_subset['Meter_kW']
        )
        df['Expected Power'] /= scaling_factor
        self.power_expected_rescaled = df['Expected Power']
        df['Expected Energy'] = rdtools.energy_from_power(df['Expected Power'])
        self.energy_expected_rescaled = df['Expected Energy']
        df['Meter_kWh_interval'] = rdtools.energy_from_power(df['Meter_kW'])

        df_subset = df.loc[subset, :]

        # window length is "number of daytime intervals".
        # Note: these bounds are intended to provide good resolution
        # across many dataset lengths
        window_lengths = 2**np.arange(1, int(np.log2(len(df_subset))), 1)

        results_list = []
        for window_length in window_lengths:
            rolling = df_subset.rolling(window=window_length, center=True)
            window = rolling.sum()
            actual = window['Meter_kWh_interval']
            expected = window['Expected Energy']
            # remove the nans at beginning and end because of minimum window
            # length
            actual = actual[~np.isnan(actual)]
            expected = expected[~np.isnan(expected)]
            temp = pd.DataFrame({
                'actual': actual,
                'expected': expected,
                'window length': window_length
            })
            results_list.append(temp)

        df_error = pd.concat(results_list)
        df_error['error'] = df_error['actual'] / df_error['expected'] - 1

        self.error_info = df_error
        error = df_error.groupby('window length')['error']
        lower = error.quantile(quantiles[0])
        upper = error.quantile(quantiles[1])

        # functions to predict the confidence interval for a given outage
        # length. linear interp inside the range, nearest neighbor outside the
        # range.
        def interp(series):
            return interp1d(series.index, series.values,
                            fill_value=(series.values[0], series.values[-1]),
                            bounds_error=False)

        # functions mapping number of intervals (outage length) to error bounds
        def interp_lower(n_intervals):
            return float(interp(lower)(n_intervals))

        def interp_upper(n_intervals):
            return float(interp(upper)(n_intervals))

        self.interp_lower = interp_lower
        self.interp_upper = interp_upper

    def _calc_loss_system(self):
        """
        Estimate total production loss from system downtime events.

        See Section II-B in [1]_.

        This implements the "expected energy" method from [1]_ of comparing
        system production recovered from cumulative production data with
        expected production from an energy model.

        This function is useful for full system outages when no system data is
        available at all. However, it does require cumulative production data
        recorded at the device level and only reports estimated lost production
        for entire outages rather than timeseries lost power.

        Sets the `outage_info`, `energy_cumulative_corrected`, and
        `loss_system` attributes.
        """
        # Calculate boolean series to indicate full outages. Considerations:
        # - Multi-day outages need to span across nights
        # - Full outages don't always take out communications, so the
        #   cumulative meter can either drop out or stay constant depending on
        #   the case.
        # During a full outage, no inverters will report production:
        looks_offline = ~self.reporting_mask.any(axis=1)
        # Now span across nights:
        all_times = self.power_system.index
        masked = looks_offline[self.power_expected > 0].reindex(all_times)
        # Note: in Series, (nan | True) is False, but (True | nan) is True
        full_outage = (
            masked.ffill().fillna(False) | masked.bfill().fillna(False)
        )

        # Find expected production and associated uncertainty for each outage
        diff = full_outage.astype(int).diff()
        starts = all_times[diff == 1].tolist()
        ends = all_times[diff.shift(-1) == -1].tolist()
        steps = diff[~diff.isnull() & (diff != 0)]
        if not steps.empty:
            if steps[0] == -1:
                # data starts in an outage
                starts.insert(0, all_times[0])
            if steps[-1] == 1:
                # data ends in an outage
                ends.append(all_times[-1])

        outage_data = []
        for start, end in zip(starts, ends):
            outage_expected_power = self.power_expected_rescaled[start:end]
            daylight_intervals = (outage_expected_power > 0).sum()
            outage_expected_energy = self.energy_expected_rescaled[start:end]

            # self.cumulative_energy[start] is the first value in the outage.
            # so to get the starting energy, need to get previous value:
            start_minus_one = all_times[all_times.get_loc(start)-1]

            data = {
                'start': start,
                'end': end,
                'duration': end - start,
                'intervals': len(outage_expected_power),
                'daylight_intervals': daylight_intervals,
                'error_lower': self.interp_lower(daylight_intervals),
                'error_upper': self.interp_upper(daylight_intervals),
                'energy_expected': outage_expected_energy.sum(),
                'energy_start': self.energy_cumulative[start_minus_one],
                'energy_end': self.energy_cumulative[end],
            }
            outage_data.append(data)

        # specify columns in case no outages are found. Also specifies
        # the order for pandas < 0.25.0
        cols = ['start', 'end', 'duration', 'intervals', 'daylight_intervals',
                'error_lower', 'error_upper', 'energy_expected',
                'energy_start', 'energy_end']
        df_outages = pd.DataFrame(outage_data, columns=cols)

        df_outages['energy_actual'] = (
            df_outages['energy_end'] - df_outages['energy_start']
        )
        # poor-quality cumulative meter data can create "negative production"
        # outages. Set to nan so that negative value doesn't pollute other
        # calcs. However, if using a net meter (instead of delivered), system
        # consumption creates a legitimate decrease during some outages. Rule
        # of thumb is that system consumption is about 0.5% of system
        # production, but it'll be larger during winter. Choose 5% to be safer.
        lower_limit = -0.05 * df_outages['energy_expected']  # Note the sign
        below_limit = df_outages['energy_actual'] < lower_limit
        df_outages.loc[below_limit, 'energy_actual'] = np.nan

        df_outages['ci_lower'] = (
            (1 + df_outages['error_lower']) * df_outages['energy_expected']
        )
        df_outages['ci_upper'] = (
            (1 + df_outages['error_upper']) * df_outages['energy_expected']
        )
        df_outages['type'] = np.where(
            df_outages['energy_actual'] < df_outages['ci_lower'],
            'real',
            'comms')
        df_outages.loc[df_outages['energy_actual'].isnull(), 'type'] = 'unknown'
        df_outages['loss'] = np.where(
            df_outages['type'] == 'real',
            df_outages['energy_expected'] - df_outages['energy_actual'],
            0)
        df_outages.loc[df_outages['type'] == 'unknown', 'loss'] = np.nan

        self.outage_info = df_outages

        # generate a best-guess timeseries loss for the full outages by
        # scaling the expected power signal to match the actual
        lost_power_full = pd.Series(0, index=self.loss_subsystem.index)
        expected_power = self.power_expected
        corrected_cumulative_energy = self.energy_cumulative.copy()
        for i, row in self.outage_info.iterrows():
            start = row['start']
            end = row['end']
            subset = expected_power.loc[start:end].copy()
            subset_energy = rdtools.energy_from_power(subset)
            loss_fill = subset * row['loss'] / subset_energy.sum()
            lost_power_full.loc[subset.index] += loss_fill

            # fill in the cumulative meter during the outages, again using
            # the expected energy signal, but rescaled to match actual
            # production this time:
            production_fill = subset_energy.cumsum()
            production_fill *= row['energy_actual'] / subset_energy.sum()
            corrected_segment = row['energy_start'] + production_fill
            corrected_cumulative_energy.loc[start:end] = corrected_segment

        self.energy_cumulative_corrected = corrected_cumulative_energy
        self.loss_system = lost_power_full

    def _combine_losses(self, rollup_period='M'):
        """
        Combine subsystem and system losses.

        Sets the `loss_total` and `results` attributes.

        Parameters
        ----------
        rollup_period : pandas offset string, default 'M'
            The period on which to roll up losses and calculate availability.
        """

        if ((self.loss_system > 0) & (self.loss_subsystem > 0)).any():
            msg = (
                'Loss detected simultaneously at both system and subsystem '
                'levels. This is unexpected and could indicate a problem with '
                'the input time series data.'
            )
            warnings.warn(msg, UserWarning)

        self.loss_total = self.loss_system + self.loss_subsystem

        # calculate actual production based on corrected cumulative meter
        cumulative_energy = self.energy_cumulative_corrected
        resampled_cumulative = cumulative_energy.resample(rollup_period)
        actual_production = (
            resampled_cumulative.last() - resampled_cumulative.first()
        )

        lost_production = rdtools.energy_from_power(self.loss_total)
        df = pd.DataFrame({
            'lost_production': lost_production.resample(rollup_period).sum(),
            'actual_production': actual_production,
        })
        loss_plus_actual = df['lost_production'] + df['actual_production']
        df['availability'] = 1 - df['lost_production'] / loss_plus_actual
        self.results = df

    def run(self, low_threshold=None, relative_sizes=None,
            power_system_limit=None, quantiles=(0.01, 0.99),
            rollup_period='M'):
        """
        Run the availability analysis.

        Parameters
        ----------
        low_threshold : float or pd.Series, optional
            An optional threshold used to naively classify subsystems as
            online. If the threshold is a scalar, it will be used for all
            subsystems. For subsystems with different capacities, a pandas
            Series may be passed with index values matching the columns in
            ``power_subsystem``. Units must match ``power_subsystem`` and
            ``power_system``. If omitted, the limit is calculated for each
            subsystem independently as 0.001 times the 99th percentile of its
            power data.

        relative_sizes : dict or pd.Series, optional
            The production capacity of each subsystem, normalized by the mean
            subsystem capacity. If not specified, it will be estimated from
            power data.

        power_system_limit : float or pd.Series, optional
            Maximum allowable system power in the same units as the input
            power timeseries. This parameter is used to account
            for cases where online subsystems can partially mitigate the loss
            of an offline subsystem, for example a system with a plant
            controller and dynamic inverter setpoints. This constraint is
            only applied to the subsystem loss calculation.

        quantiles : 2-element tuple, default (0.01, 0.99)
            The quantiles of the error distribution used for the expected
            energy confidence interval. The lower bound is used to classify
            outages as either (1) a simple communication interruption with
            no production loss or (2) a power outage with an associated
            production loss estimate.

        rollup_period : pandas DateOffset or alias, default 'M'
            The period on which to roll up losses and calculate availability.
        """
        self._calc_loss_subsystem(low_threshold, relative_sizes,
                                  power_system_limit)
        self._calc_error_distributions(quantiles)
        self._calc_loss_system()
        self._combine_losses(rollup_period)

    def plot(self):
        """
        Create a figure summarizing the availability analysis results. The
        analysis must be run using the :py:meth:`.run` method before using
        this method.

        Returns
        -------
        fig : matplotlib Figure
        """
        try:
            self.loss_total
        except AttributeError:
            raise TypeError("No results to plot, use the `run` method first")

        return rdtools.plotting.availability_summary_plots(
            self.power_system, self.power_subsystem, self.loss_total,
            self.energy_cumulative, self.energy_expected_rescaled,
            self.outage_info)
