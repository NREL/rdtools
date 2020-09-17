"""
Functions for detecting and quantifying production loss from photovoltaic
system downtime events.
"""

import rdtools

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class AvailabilityAnalysis:
    """
    A class to perform system availability and loss analysis.

    This class follows the analysis procedure described in [1]_.

    Parameters
    ----------
    system_power : pd.Series
        Timeseries total system power. In the typical case, this is meter
        power data.

    subsystem_power : pd.DataFrame
        Timeseries power data, one column per subsystem. In the typical case,
        this is inverter AC power data. Each column is assumed to represent
        a subsystem, so no extra columns may be included. The index must
        match ``system_power``.

    cumulative_energy : pd.Series
        Timeseries cumulative energy data for the entire system (e.g. meter).
        These values must be recorded at the device itself (rather than summed
        by a downstream device like a datalogger or DAS provider) to preserve
        its integrity across communication interruptions. Units must match
        ``power`` integrated to hourly energy (e.g. if ``power`` is in kW then
        ``energy`` must be in kWh).

    expected_power : pd.Series
        Expected system power data with the same index as the measured data.
        This can be modeled from on-site weather measurements if there is no
        risk of instrument calibration or data gaps. However, because full
        system outages often cause weather data to be lost as well, it may
        be more useful to use data from an independent weather station or
        satellite-based weather provider.

    Attributes
    ----------
    results : pd.DataFrame
        Rolled-up production, loss, and availability metrics. TODO: col table

    system_loss : pd.Series
        Estimated timeseries lost power from system outages.

    subsystem_loss : pd.Series
        Estimated timeseries lost power from subsystem outages.

    reporting_mask : pd.DataFrame
        Boolean mask indicating whether subsystems appear online or not.

    rescaled_expected_power : pd.Series
        Expected power rescaled to better match system power during periods
        where the system is performing normally.

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
        | `start`              | Timestamp of the outage start.               |
        +----------------------+----------------------------------------------+
        | `end`                | Timestamp of the outage end.                 |
        +----------------------+----------------------------------------------+
        | `duration`           | Length of the outage (*i.e.*, `end - start`).|
        +----------------------+----------------------------------------------+
        | `intervals`          | Total count of data intervals contained in   |
        |                      | the outage.                                  |
        +----------------------+----------------------------------------------+
        | `daylight_intervals` | Count of data intervals contained in the     |
        |                      | outage occurring during the day.             |
        +----------------------+----------------------------------------------+
        | `error_lower`        | Lower error bound as a fraction of expected  |
        |                      | energy.                                      |
        +----------------------+----------------------------------------------+
        | `error_upper`        | Upper error bound as a fraction of expected  |
        |                      | energy.                                      |
        +----------------------+----------------------------------------------+
        | `expected_energy`    | Total expected production for the outage     |
        |                      | duration.                                    |
        +----------------------+----------------------------------------------+
        | `start_energy`       | System cumulative production at the outage   |
        |                      | start.                                       |
        +----------------------+----------------------------------------------+
        | `end_energy`         | System cumulative production at the outage   |
        |                      | end.                                         |
        +----------------------+----------------------------------------------+
        | `actual_energy`      | System production during the outage (*i.e.*, |
        |                      | `end_energy - start_energy`).                |
        +----------------------+----------------------------------------------+
        | `ci_lower`           | Lower bound for the expected energy          |
        |                      | confidence interval.                         |
        +----------------------+----------------------------------------------+
        | `ci_upper`           | Lower bound for the expected energy          |
        |                      | confidence interval.                         |
        +----------------------+----------------------------------------------+
        | `type`               | Type of the outage (`'real'` or `'comms'`).  |
        +----------------------+----------------------------------------------+
        | `loss`               | Estimated production loss.                   |
        +----------------------+----------------------------------------------+

    Notes
    -----
    This class's ability to detect short-duration outages is limited by
    the resolution of the system data. For instance, 15-minute averages
    would not be able to resolve the rapid power cycling of an intermittent
    inverter. Additionally, the loss at the edges of an outage may be
    underestimated because of masking by the interval averages.

    The outage detection routine assumes that the input timeseries are
    continuous, so any gaps in the timeseries should be filled with nan before
    using this function.  See :py:func:`rdtools.normalization.interpolate`.

    References
    ----------
    .. [1] Anderson K. and Blumenthal R. "Overcoming communications outages in
       inverter downtime analysis", 2020 IEEE 47th Photovoltaic Specialists
       Conference (PVSC).
    """

    def __init__(self, system_power, subsystem_power, cumulative_energy,
                 expected_power):
        self.system_power = system_power
        self.subsystem_power = subsystem_power
        self.cumulative_energy = cumulative_energy
        self.expected_power = expected_power
        # TODO: assert indexes are all aligned?

    def _loss_from_power(self, low_threshold, relative_sizes,
                         system_power_limit):
        """
        Estimate timeseries production loss from subsystem downtime events.

        This implements the "power comparison" method from [1]_ of comparing
        subsystem power data to total system power (e.g. inverter power to
        meter power).

        Because this method is based on peer-to-peer comparison at each
        timestamp, it is not suitable for full system outages (i.e., at least
        one inverter must be reporting along with the system meter).

        Sets the `reporting_mask` and `subsystem_loss` attributes.

        Parameters
        ----------
        low_threshold : float or pd.Series
            An optional threshold used to naively classify subsystems as
            online. If the threshold is a scalar, it will be used for all
            subsystems. For subsystems with different capacities, a pandas
            Series may be passed with index values matching the columns in
            ``subsystem_power``. Units must match ``subsystem_power`` and
            ``system_power``. If omitted, the limit is calculated for each
            subsystem independently as 0.001 times the 99th percentile of its
            power data.

        relative_sizes : dict or pd.Series
            The production capacity of each subsystem, normalized by the mean
            subsystem capacity. If not specified, it will be estimated from
            power data.

        system_power_limit : float or pd.Series
            An optional maximum system power used as an upper limit for
            (system_power + lost_power) so that the maximum system capacity or
            interconnection limit is not exceeded. If omitted, that check is
            skipped.
        """
        subsystem_power = self.subsystem_power
        system_power = self.system_power
        subsystem_power = subsystem_power.fillna(0)
        system_power = system_power.clip(lower=0)

        # Part A
        if low_threshold is None:
            low_threshold = subsystem_power.quantile(0.99) / 1000

        self.reporting_mask = looks_online = subsystem_power > low_threshold
        reporting = subsystem_power[looks_online]
        if relative_sizes is None:
            # normalize by mean power and take the median across the timeseries
            normalized = reporting.divide(reporting.mean(axis=1), axis=0)
            relative_sizes = normalized.median()

        normalized_subsystem_powers = reporting.divide(relative_sizes, axis=1)
        mean_subsystem_power = normalized_subsystem_powers.mean(axis=1)

        virtual_full_power = mean_subsystem_power * subsystem_power.shape[1]

        system_delta = 1 - system_power / virtual_full_power

        subsystem_fraction = relative_sizes / relative_sizes.sum()
        smallest_delta = subsystem_power.le(low_threshold) \
                                        .replace(False, np.nan) \
                                        .multiply(subsystem_fraction) \
                                        .min(axis=1) \
                                        .fillna(1)  # use safe value of 100%
        is_downtime = system_delta > (0.75 * smallest_delta)
        is_downtime[looks_online.all(axis=1)] = False

        # Part B
        lowest_possible = looks_online.multiply(subsystem_fraction).sum(axis=1)
        f_online = system_power / virtual_full_power
        f_online = f_online.clip(lower=lowest_possible, upper=1)
        p_loss = (1 - f_online) / f_online * system_power
        p_loss[~is_downtime] = 0

        if system_power_limit is not None:
            limit_exceeded = p_loss + system_power > system_power_limit
            loss = system_power_limit - system_power[limit_exceeded]
            p_loss.loc[limit_exceeded] = loss.clip(lower=0)

        self.subsystem_loss = p_loss.fillna(0)

    def _error_distributions(self, quantiles):
        """
        Calculate the error distributions of Section II-A in [1]_.

        Sets the `rescaled_expected_power`, `rescaled_expected_energy`,
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
            'Meter_kW': self.system_power,
            'Expected Power': self.expected_power,
            'Meter_kWh': self.cumulative_energy,
        })

        system_performing_normally = (
            (self.subsystem_loss == 0) & (self.system_power > 0)
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
        self.rescaled_expected_power = df['Expected Power']
        df['Expected Energy'] = rdtools.energy_from_power(df['Expected Power'])
        self.rescaled_expected_energy = df['Expected Energy']
        df['Meter_kWh_interval'] = rdtools.energy_from_power(df['Meter_kW'])

        df_subset = df.loc[subset, :]

        # window length is "number of daytime intervals".
        # Note: the logspace bounds are currently kind of arbitrary
        window_lengths = np.logspace(np.log10(12),
                                     np.log10(0.75*len(df_subset)),
                                     10).astype(int)
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

    def _loss_from_energy(self):
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

        Sets the `outage_info`, `corrected_cumulative_energy`, and
        `system_loss` attributes.
        """
        # Calculate boolean series to indicate full outages. Considerations:
        # - Multi-day outages need to span across nights
        # - Full outages don't always take out communications, so the
        #   cumulative meter can either drop out or stay constant depending on
        #   the case.
        # During a full outage, no inverters will report production:
        looks_offline = ~self.reporting_mask.any(axis=1)
        # Now span across nights:
        all_times = self.system_power.index
        masked = looks_offline[self.expected_power > 0].reindex(all_times)
        full_outage = masked.ffill() | masked.bfill()
        full_outage = full_outage.fillna(False)

        # Find expected production and associated uncertainty for each outage
        diff = full_outage.astype(int).diff()
        starts = all_times[diff == 1].tolist()
        ends = all_times[diff == -1].tolist()
        steps = diff[~diff.isnull() & (diff != 0)]
        if steps[0] == -1:
            # data starts in an outage
            starts.insert(0, all_times[0])
        if steps[-1] == 1:
            # data ends in an outage
            ends.append(all_times[-1])

        outage_data = []
        for start, end in zip(starts, ends):
            outage_expected_power = self.rescaled_expected_power[start:end]
            daylight_intervals = (outage_expected_power > 0).sum()
            outage_expected_energy = self.rescaled_expected_energy[start:end]

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
                'expected_energy': outage_expected_energy.sum(),
                'start_energy': self.cumulative_energy[start_minus_one],
                'end_energy': self.cumulative_energy[end],
            }
            outage_data.append(data)

        df_outages = pd.DataFrame(outage_data)
        # pandas < 0.25.0 sorts columns alphabetically.  revert to dict order:
        df_outages = df_outages[data.keys()]

        df_outages['actual_energy'] = (
            df_outages['end_energy'] - df_outages['start_energy']
        )
        # poor-quality cumulative meter data can create "negative production"
        # outages. Set to nan so that negative value doesn't pollute other
        # calcs. However, if using a net meter (instead of delivered), system
        # consumption creates a legitimate decrease during some outages. Rule
        # of thumb is that system consumption is about 0.5% of system
        # production, but it'll be larger during winter. Choose 5% to be safer.
        lower_limit = -0.05 * df_outages['expected_energy']  # Note the sign
        below_limit = df_outages['actual_energy'] < lower_limit
        df_outages.loc[below_limit, 'actual_energy'] = np.nan

        df_outages['ci_lower'] = (
            (1 + df_outages['error_lower']) * df_outages['expected_energy']
        )
        df_outages['ci_upper'] = (
            (1 + df_outages['error_upper']) * df_outages['expected_energy']
        )
        df_outages['type'] = np.where(
            df_outages['actual_energy'] < df_outages['ci_lower'],
            'real',
            'comms')
        df_outages.loc[df_outages['actual_energy'].isnull(), 'type'] = 'unknown'
        df_outages['loss'] = np.where(
            df_outages['type'] == 'real',
            df_outages['expected_energy'] - df_outages['actual_energy'],
            0)
        df_outages.loc[df_outages['type'] == 'unknown', 'loss'] = np.nan

        self.outage_info = df_outages

        # generate a best-guess timeseries loss for the full outages by
        # scaling the expected power signal to match the actual
        lost_power_full = pd.Series(0, index=self.subsystem_loss.index)
        expected_power = self.expected_power
        corrected_cumulative_energy = self.cumulative_energy.copy()
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
            production_fill *= row['actual_energy'] / subset_energy.sum()
            corrected_segment = row['start_energy'] + production_fill
            corrected_cumulative_energy.loc[start:end] = corrected_segment

        self.corrected_cumulative_energy = corrected_cumulative_energy
        self.system_loss = lost_power_full

    def _combine_losses(self, rollup_period='m'):
        """
        Combine subsystem and system losses.

        Sets the `total_loss` and `results` attributes.

        Parameters
        ----------
        rollup_period : pandas offset string, default 'm'
            The period on which to roll up losses and calculate availability.
        """
        self.total_loss = self.system_loss + self.subsystem_loss

        # calculate actual production based on corrected cumulative meter
        cumulative_energy = self.corrected_cumulative_energy
        resampled_cumulative = cumulative_energy.resample(rollup_period)
        actual_production = (
            resampled_cumulative.last() - resampled_cumulative.first()
        )

        lost_production = rdtools.energy_from_power(self.total_loss)
        df = pd.DataFrame({
            'lost_production': lost_production.resample(rollup_period).sum(),
            'actual_production': actual_production,
        })
        loss_plus_actual = df['lost_production'] + df['actual_production']
        df['availability'] = 1 - df['lost_production'] / loss_plus_actual
        self.results = df.tz_localize(None)

    def run(self, low_threshold=None, relative_sizes=None,
            system_power_limit=None, quantiles=(0.01, 0.99),
            rollup_period='m'):
        """
        Run the availability analysis.

        Parameters
        ----------
        low_threshold : float or pd.Series, optional
            An optional threshold used to naively classify subsystems as
            online. If the threshold is a scalar, it will be used for all
            subsystems. For subsystems with different capacities, a pandas
            Series may be passed with index values matching the columns in
            ``subsystem_power``. Units must match ``subsystem_power`` and
            ``system_power``. If omitted, the limit is calculated for each
            subsystem independently as 0.001 times the 99th percentile of its
            power data.

        relative_sizes : dict or pd.Series, optional
            The production capacity of each subsystem, normalized by the mean
            subsystem capacity. If not specified, it will be estimated from
            power data.

        system_power_limit : float or pd.Series, optional
            An optional maximum system power used as an upper limit for
            (system_power + lost_power) so that the maximum system capacity or
            interconnection limit is not exceeded. If omitted, that check is
            skipped.

        quantiles : 2-element tuple, default (0.01, 0.99)
            The quantiles of the error distribution used for the expected
            energy confidence interval. The lower bound is used to classify
            outages as either (1) a simple communication interruption with
            no production loss or (2) a power outage with an associated
            production loss estimate.

        rollup_period : pandas offset string, default 'm'
            The period on which to roll up losses and calculate availability.
        """
        self._loss_from_power(low_threshold, relative_sizes,
                              system_power_limit)
        self._error_distributions(quantiles)
        self._loss_from_energy()
        self._combine_losses(rollup_period)

    def plot(self):
        """
        pass.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        """
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[:, 1], sharex=ax1)

        # inverter power
        self.subsystem_power.plot(ax=ax1)
        ax1.set_ylabel('Inverter Power [kW]')
        # meter power
        self.system_power.plot(ax=ax2)
        ax2.set_ylabel('System power [kW]')
        # lost power
        self.total_loss.plot(ax=ax3)
        ax3.set_ylabel('Estimated lost power [kW]')

        # cumulative energy
        measured_artist = ax4.plot(self.cumulative_energy)
        for i, row in self.outage_info.iterrows():
            start, end = row[['start', 'end']]
            start_energy = row['start_energy']
            expected_energy = row['expected_energy']
            lo, hi = np.abs(expected_energy - row[['ci_lower', 'ci_upper']])
            expected_curve = self.rescaled_expected_energy[start:end].cumsum()
            expected_curve += start_energy
            expected_artist = ax4.plot(expected_curve, c='tab:orange')
            energy_end = expected_curve.iloc[-1]
            uncertainty_artist = ax4.errorbar([end], [energy_end],
                                              [[lo], [hi]], c='k')

        artists = [measured_artist[0], expected_artist[0], uncertainty_artist]
        labels = ['Reported Production', 'Expected Production', 'Uncertainty']
        ax4.legend(artists, labels, loc='upper left')
        ax4.set_ylabel('Cumulative Energy [kWh]')
        return fig
