'''
A high-level but customizable wrapper around the low-level RdTools functions.
'''

from rdtools import (
    normalization, clearsky_temperature, filtering, aggregation,
    soiling, degradation, plotting
)

import dask

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
import pvlib

import logging
log = logging.getLogger('rdtools')
log.addHandler(logging.NullHandler())


def transpose(tilt, azimuth, albedo, solar_position, irradiance, **kwargs):
    poa = pvlib.irradiance.get_total_irradiance(
        tilt,
        azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        irradiance['dni'],
        irradiance['ghi'],
        irradiance['dhi'],
        albedo=albedo,
        **kwargs)
    return poa['poa_global']


def power_to_energy(signal, max_timedelta):
    return normalization.energy_from_power(signal, max_timedelta=max_timedelta)


def irradiance_rescale(clearsky_poa_unscaled, poa, rescale_poa, method):
    if rescale_poa:
        clearsky_poa = normalization.irradiance_rescale(
            poa, clearsky_poa_unscaled, method=method
        )
    else:
        clearsky_poa = clearsky_poa_unscaled
    return clearsky_poa


def cell_temperature(poa, windspeed, ambient_temperature,
                     temperature_model):
    kwargs = dict(
        poa_global=poa,
        wind_speed=windspeed,
        temp_air=ambient_temperature
    )
    if temperature_model is not None:
        kwargs['model'] = temperature_model
    tcell = pvlib.pvsystem.sapm_celltemp(**kwargs)
    return tcell['temp_cell']


def normalize(pv_energy, poa, sensor_cell_temperature,
              gamma_pdc, g_ref, t_ref, system_size):
    if system_size is None:
        renorm = True
        system_size = 1.0
    else:
        renorm = False

    pvwatts_kws = {
        "poa_global": poa,
        "P_ref": system_size,
        "T_cell": sensor_cell_temperature,
        "G_ref": g_ref,
        "T_ref": t_ref,
        "gamma_pdc": gamma_pdc
    }

    normalized, insolation = normalization.normalize_with_pvwatts(
        pv_energy, pvwatts_kws
    )

    if renorm:
        # Normalize to the 95th percentile for convenience;
        # this is renormalized out in the calculations but is relevant
        # to normalized_filter()
        x = normalized[np.isfinite(normalized)]
        normalized = normalized / x.quantile(0.95)
    return normalized, insolation


def aggregate(normalized, insolation, overall_filter,
              aggregation_frequency):
    norm = normalized[overall_filter]
    insol = insolation[overall_filter]
    aggregated = aggregation.aggregation_insol(norm, insol,
                                               aggregation_frequency)
    aggregated_insolation = insol.resample(aggregation_frequency).sum()

    return aggregated, aggregated_insolation


def srr_soiling(aggregated, aggregated_insolation, **kwargs):
    if aggregated.index.freq != 'D' or \
       aggregated_insolation.index.freq != 'D':
        raise ValueError(
            'Soiling SRR analysis requires daily aggregation.'
        )

    sr, sr_ci, soiling_info = soiling.soiling_srr(
        aggregated, aggregated_insolation,
        **kwargs
    )
    srr_results = {
        'p50_sratio': sr,
        'sratio_confidence_interval': sr_ci,
        'calc_info': soiling_info
    }
    return srr_results


def yoy_degradation(aggregated, **kwargs):
    yoy_rd, yoy_ci, yoy_info = degradation.degradation_year_on_year(
        aggregated,
        **kwargs
    )

    yoy_results = {
        'p50_rd': yoy_rd,
        'rd_confidence_interval': yoy_ci,
        'calc_info': yoy_info
    }
    return yoy_results


class SystemAnalysis:
    """
    A class for lazy evaluation of complex computation sequences.
    The class is based on a dynamic model plugin architecture that
    automatically determines the required calculations for a desired result.

    Parameters
    ----------
    init : dict
        Base values to use as inputs for the analysis.
    """

    DEFAULTS = dict(
        irradiance_rescale_method='iterative',
        albedo=0.25,
        system_size=None,
        dc_model='pvwatts',
        g_ref=1000,
        t_ref=25,
        normalized_low_cutoff=0,
        normalized_high_cutoff=np.inf,
        poa_low_cutoff=200,
        poa_high_cutoff=1200,
        cell_temperature_low_cutoff=-50,
        cell_temperature_high_cutoff=110,
        clip_quantile=0.98,
        clearsky_index_threshold=0.15,
        aggregation_frequency='D',
        temperature_model=None,
        rescale_poa=True,
        clearsky_windspeed=0,
    )

    def __init__(self, init):

        interpolation_frequency = init.get('interpolation_frequency', None)
        if interpolation_frequency is not None:
            max_timedelta = init['max_timedelta']
            interpolation_keys = [
                'pv', 'poa', 'ghi', 'dni', 'dhi', 'ambient_temperature',
                'windspeed'
            ]
            for key in ['pv_tilt', 'pv_azimuth']:
                if isinstance(init[key], (pd.Series, pd.DataFrame)):
                    interpolation_keys.append(key)
            for key in interpolation_keys:
                if key not in init:
                    continue
                values = init[key]
                interpolated = normalization.interpolate(
                    values, interpolation_frequency, max_timedelta
                )
                init[key] = interpolated

        common_graph = {
            'pvlib_location': (
                    pvlib.location.Location, 'latitude', 'longitude'
            ),
            'times': (
                lambda series: series.index, 'pv'
            ),
            'pv_energy': (
                power_to_energy, 'pv', 'max_timedelta'
            ),
            'solar_position': (
                lambda loc, times: loc.get_solarposition(times),
                'pvlib_location',
                'times'
            ),
            'clip_filter': (
                filtering.clip_filter, 'pv', 'clip_quantile'
            )
        }

        # Clear-sky deg+soiling workflow
        clearsky_graph = {
            'clearsky_irradiance': (
                lambda loc, times, solpos:
                    loc.get_clearsky(times, solar_position=solpos),
                'pvlib_location',
                'times',
                'solar_position'
            ),
            'clearsky_poa_unscaled': (
                transpose,
                'pv_tilt',
                'pv_azimuth',
                'albedo',
                'solar_position',
                'clearsky_irradiance'
            ),
            'clearsky_poa': (
                irradiance_rescale,
                'clearsky_poa_unscaled',
                'poa',
                'rescale_poa',
                'irradiance_rescale_method'
            ),
            'clearsky_ambient_temperature': (
                lambda loc, times: clearsky_temperature.get_clearsky_tamb(
                    times,
                    loc.latitude,
                    loc.longitude),
                'pvlib_location',
                'times'
            ),
            'clearsky_cell_temperature': (
                cell_temperature,
                'clearsky_poa',
                'clearsky_windspeed',
                'clearsky_ambient_temperature',
                'temperature_model'
            ),
            'clearsky_normalized-clearsky_insolation': (
                normalize,
                'pv_energy',
                'clearsky_poa',
                'clearsky_cell_temperature',
                'gamma_pdc',
                'g_ref',
                't_ref',
                'system_size'
            ),
            'clearsky_normalized': (
                lambda tup: tup[0], 'clearsky_normalized-clearsky_insolation'
            ),
            'clearsky_insolation': (
                lambda tup: tup[1], 'clearsky_normalized-clearsky_insolation'
            ),
            'clearsky_normalized_filter': (
                filtering.normalized_filter,
                'clearsky_normalized',
                'normalized_low_cutoff',
                'normalized_high_cutoff'
            ),
            'clearsky_poa_filter': (
                filtering.poa_filter,
                'clearsky_poa',
                'poa_low_cutoff',
                'poa_high_cutoff'
            ),
            'clearsky_cell_temperature_filter': (
                filtering.tcell_filter,
                'clearsky_cell_temperature',
                'cell_temperature_low_cutoff',
                'cell_temperature_high_cutoff'
            ),
            'clearsky_csi_filter': (
                filtering.csi_filter,
                'poa',
                'clearsky_poa',
                'clearsky_index_threshold'
            ),
            'clearsky_overall_filter': (
                np.bitwise_and.reduce,
                [
                    'clearsky_normalized_filter',
                    'clearsky_poa_filter',
                    'clip_filter',
                    'clearsky_cell_temperature_filter',
                    'clearsky_csi_filter'
                ]
            ),
            'clearsky_aggregated-clearsky_aggregated_insolation': (
                aggregate,
                'clearsky_normalized',
                'clearsky_insolation',
                'clearsky_overall_filter',
                'aggregation_frequency'
            ),
            'clearsky_aggregated': (
                lambda tup: tup[0],
                'clearsky_aggregated-clearsky_aggregated_insolation'
            ),
            'clearsky_aggregated_insolation': (
                lambda tup: tup[1],
                'clearsky_aggregated-clearsky_aggregated_insolation'
            ),
            'clearsky_soiling_results': (
                srr_soiling,
                'clearsky_aggregated',
                'clearsky_aggregated_insolation'
            ),
            'clearsky_degradation_results': (
                yoy_degradation,
                'clearsky_aggregated'
            ),
        }

        # Sensor-based sensor+deg workflow
        sensor_graph = {
            'sensor_cell_temperature': (
                cell_temperature,
                'poa',
                'windspeed',
                'ambient_temperature',
                'temperature_model'
            ),
            'sensor_normalized-sensor_insolation': (
                normalize,
                'pv_energy',
                'poa',
                'sensor_cell_temperature',
                'gamma_pdc',
                'g_ref',
                't_ref',
                'system_size'
            ),
            'sensor_normalized': (
                lambda tup: tup[0], 'sensor_normalized-sensor_insolation'
            ),
            'sensor_insolation': (
                lambda tup: tup[1], 'sensor_normalized-sensor_insolation'
            ),
            'sensor_normalized_filter': (
                filtering.normalized_filter,
                'sensor_normalized',
                'normalized_low_cutoff',
                'normalized_high_cutoff'
            ),
            'sensor_poa_filter': (
                filtering.poa_filter,
                'poa',
                'poa_low_cutoff',
                'poa_high_cutoff'
            ),
            'sensor_cell_temperature_filter': (
                filtering.tcell_filter,
                'sensor_cell_temperature',
                'cell_temperature_low_cutoff',
                'cell_temperature_high_cutoff'
            ),
            'sensor_overall_filter': (
                np.bitwise_and.reduce,
                [
                    'sensor_normalized_filter',
                    'sensor_poa_filter',
                    'clip_filter',
                    'sensor_cell_temperature_filter'
                ]
            ),
            'sensor_aggregated-sensor_aggregated_insolation': (
                aggregate,
                'sensor_normalized',
                'sensor_insolation',
                'sensor_overall_filter',
                'aggregation_frequency'
            ),
            'sensor_aggregated': (
                lambda tup: tup[0],
                'sensor_aggregated-sensor_aggregated_insolation'
            ),
            'sensor_aggregated_insolation': (
                lambda tup: tup[1],
                'sensor_aggregated-sensor_aggregated_insolation'
            ),
            'sensor_soiling_results': (
                srr_soiling,
                'sensor_aggregated',
                'sensor_aggregated_insolation'
            ),
            'sensor_degradation_results': (
                yoy_degradation,
                'sensor_aggregated'
            )
        }

        overall_graph = {
            **SystemAnalysis.DEFAULTS,
            **common_graph,
            **clearsky_graph,
            **sensor_graph,
            **init  # add in the init vars last so that they take precedence
        }

        # self.graph is a Dask task graph.  Keys are task/output names,
        # values are either literals (floats, dataframes etc) or "computations"
        # that are a tuple like (function, input1, input2), where function is
        # an actual function handle and input1/input2 are output names.
        # Note:  Dask permits input1 and input2 to be literals, but we don't
        # to simplify the graph validation code.  All literals should be stored
        # under their own key.
        self.graph = {}

        for key, computation in overall_graph.items():
            self[key] = computation

    def __getitem__(self, key):
        computation = self.graph[key]
        if isinstance(computation, tuple):
            # unwrap the calculation function before returning
            computation = (
                computation[0].__wrapped__,
                *computation[1:]
            )
        return computation

    def __setitem__(self, key, computation):
        if isinstance(computation, tuple):
            # wrap the calculation function before inserting
            wrapper = self._loggify(key, computation[0])
            computation = (
                wrapper,
                *computation[1:]
            )
        self.graph[key] = computation

    def keys(self):
        """
        Get the names of computations in the internal task graph.

        Returns
        -------
        keys : dict_keys
        """
        return self.graph.keys()

    def _loggify(self, key, f):
        """
        Wrap function calls with logging debug info since dask tracebacks
        aren't as helpful as we'd like.
        """
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # debug message like:

            # "calculating pv_energy with
            # power_to_energy(pandas.core.series.Series, builtins.str)"
            argnames = self.graph[key][1:]
            argtypes = [f"{arg.__class__.__module__}."
                        f"{arg.__class__.__name__}"
                        for arg in args+tuple(kwargs.values())]
            argmsg = ", ".join([
                "{} [{}]".format(argname, argtype)
                for argname, argtype in zip(argnames, argtypes)
            ])
            msg = f"calculating {key} with {f.__name__}({argmsg})"
            log.debug(msg)

            try:
                value = f(*args, **kwargs)
            except Exception as e:
                # chained exceptions like:

                # IndexError: index 0 is out of bounds for axis 0 with size 0
                #
                # The above exception was the direct cause of the following
                # exception:
                #
                # RuntimeError: Could not evaluate 'pv_energy'. Check the above
                # traceback for details.  Note: some inputs are of type str;
                # this can be an indication that not all inputs are specified.

                # note:  ipython recently fixed a bug with printing exc chains
                # https://github.com/ipython/ipython/issues/11995
                err_msg = (f"Could not evaluate '{key}'.  Check the above "
                            "traceback for details.  ")
                if any(argtype == 'builtins.str' for argtype in argtypes):
                    err_msg += ("Note: some inputs are of type str; this "
                                "can be an indication that not all inputs "
                                "are specified. ")
                # TODO: probably better to raise a custom error here?
                raise RuntimeError(err_msg) from e
            return value

        return wrapper

    def calculate(self, key, scheduler=dask.get, **kwargs):
        """
        Calculate and return the value of ``key``, dynamically resolving any
        dependencies.

        Parameters
        ----------
        key : str or list of str
            The variable(s) to retrieve.
        scheduler : function, default dask.get
            A function capable of evaluating keys from task graphs.  The
            default scheduler ``dask.get`` is synchronous (single-threaded).
            Dask does provide asynchronous schedulers through
            ``dask.multiprocessing.get`` and ``dask.threaded.get``.  It may
            also be possible to use dasks's distributed functionality here.
        kwargs :
            Extra parameters passed to the scheduler function.

        Returns
        -------
        The value of ``key`` (or a tuple of values if passed a list) as
        calculated from the task graph.
        """
        args = {}
        if scheduler in [dask.get, dask.multiprocessing.get]:
            args['keys'] = key
        else:
            args['result'] = key  # dask.threaded.get has a different interface
        # trace the graph first to make sure it's complete
        if isinstance(key, list):
            for k in key:
                _ = self.trace(k)
        else:
            _ = self.trace(key)
        dsk, _ = dask.optimization.cull(self.graph, key)
        args['dsk'] = dsk
        val = scheduler(**args, **kwargs)
        return val

    def visualize(self):
        """
        Generate a visualization of the internal task graph.

        Requires graphviz.
        """
        return dask.visualize(self.graph)

    def trace(self, key, collapse_tree=False):
        """
        Recursively traverse the model graph and trace the dependencies
        required to calculate ``key``.

        Parameters
        ----------
        key : str
            The variable to trace.
        collapse_tree : bool, default False
            If True, flatten the depency tree into a set of dependencies.

        Returns
        -------
        A nested dict representing the dependency tree of ``key``.
        """
        # TODO: probably better to raise custom errors in here

        def recurse(tree, key, history):
            history_path = "/".join(history)
            if key in history:
                cycle = history[history.index(key):] + [key]
                cycle_msg = " -> ".join(cycle)
                raise ValueError(
                    f'{history_path}: Circular dependency: {cycle_msg}'
                )
            history = history + [key]
            if key not in tree:
                raise ValueError(f'{history_path:} {key} not specified')
            computation = tree[key]
            if isinstance(computation, tuple):
                args = computation[1:]
                # flatten nested lists
                list_of_lists = [
                    arg if isinstance(arg, list) else [arg] for arg in args
                ]
                args = sum(list_of_lists, [])
                # this assumes that all arguments are keys; ie no constants
                dependencies = {
                    arg: recurse(tree, arg, history) for arg in args
                }
            else:
                dependencies = {}
            return dependencies

        dependencies = recurse(self.graph, key, [])
        if collapse_tree:
            def collapse(tree):
                subtrees = [collapse(branch) for _, branch in tree.items()]
                collapsed_subtrees = sum(subtrees, [])
                return list(tree.keys()) + collapsed_subtrees
            dependencies = set(collapse(dependencies))

        return dependencies

    def plot_soiling_interval(self, result_to_plot, **kwargs):
        '''
        Return a figure visualizing the valid soiling intervals used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and
            'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_interval_plot()

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        rdtools.plotting.soiling_interval_plot
        '''

        keys = [f'{result_to_plot}_soiling_results',
                f'{result_to_plot}_aggregated']

        results_dict, aggregated = self.calculate(keys)
        fig = plotting.soiling_interval_plot(results_dict['calc_info'],
                                             aggregated, **kwargs)
        return fig

    def plot_soiling_rate_histogram(self, result_to_plot, **kwargs):
        '''
        Return a histogram of soiling rates found in the stochastic rate and
        recovery soiling analysis

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and
            'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_rate_histogram()

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        rdtools.plotting.soiling_rate_histogram
        '''

        key = f'{result_to_plot}_soiling_results'
        results_dict = self.calculate(key)

        fig = plotting.soiling_rate_histogram(results_dict['calc_info'],
                                              **kwargs)
        return fig

    def plot_soiling_monte_carlo(self, result_to_plot, **kwargs):
        '''
        Return a figure visualizing the Monte Carlo of soiling profiles used in
        stochastic rate and recovery soiling analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and
            'clearsky'
        kwargs :
            Extra parameters passed to plotting.soiling_monte_carlo_plot()

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        rdtools.plotting.soiling_monte_carlo_plot
        '''

        keys = [f'{result_to_plot}_soiling_results',
                f'{result_to_plot}_aggregated']
        results_dict, aggregated = self.calculate(keys)

        fig = plotting.soiling_monte_carlo_plot(results_dict['calc_info'],
                                                aggregated, **kwargs)
        return fig

    def plot_pv_vs_irradiance(self, poa_type, alpha=0.01, **kwargs):
        '''
        Plot PV energy vs irradiance, useful in diagnosing things like timezone
        problems or transposition errors.

        Parameters
        ----------
        poa_type: str
            The plane of array irradiance type to plot, allowed values are
            'sensor' and 'clearsky'
        alpha : numeric
            transparency of the scatter plot
        kwargs :
            Extra parameters passed to matplotlib.pyplot.axis.plot()

        Returns
        -------
        matplotlib.figure.Figure
        '''

        if poa_type == 'sensor':
            key = 'poa'
        elif poa_type == 'clearsky':
            key = 'clearsky_poa'

        poa, pv_energy = self.calculate([key, 'pv_energy'])
        to_plot = pd.merge(pd.DataFrame(poa),
                           pd.DataFrame(pv_energy),
                           left_index=True, right_index=True)

        fig, ax = plt.subplots()
        ax.plot(to_plot.iloc[:, 0], to_plot.iloc[:, 1], 'o', alpha=alpha,
                **kwargs)
        ax.set_xlim(0, 1500)
        ax.set_xlabel('Irradiance (W/m$^2$)')
        ax.set_ylabel('PV Energy (Wh/timestep)')

        return fig

    def plot_degradation_summary(self, result_to_plot, **kwargs):
        '''
        Return a figure of a scatter plot and a histogram summarizing
        degradation rate analysis.

        Parameters
        ----------
        result_to_plot : str
            The workflow result to plot, allowed values are 'sensor' and
            'clearsky'
        kwargs :
            Extra parameters passed to plotting.degradation_summary_plots()

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        rdtools.plotting.degradation_summary_plots
        '''

        keys = [f'{result_to_plot}_degradation_results',
                f'{result_to_plot}_aggregated']
        results_dict, aggregated = self.calculate(keys)

        fig = plotting.degradation_summary_plots(
                results_dict['p50_rd'],
                results_dict['rd_confidence_interval'],
                results_dict['calc_info'],
                aggregated,
                **kwargs)
        return fig
