'''
A high-level but customizable wrapper around the low-level RdTools functions.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
import pvlib
import warnings
import logging

import sys
sys.path.insert(0, r'C:\Users\KANDERSO\projects\rdtools')

from rdtools import (
    normalization, clearsky_temperature, filtering, aggregation,
    soiling, degradation, plotting
)

log = logging.getLogger('rdtools')
log.addHandler(logging.NullHandler())


def _debug(msg):
    log.debug(msg)


def _warn(msg):
    log.warning(msg)
    warnings.warn(msg)


class ModelChain:
    """
    A class for lazy evaluation of complex computation sequences.
    The class is based on a dynamic model plugin architecture that
    automatically determines the required calculations for a desired result.

    Parameters
    ----------
    **kwargs : optional
        Base values to use as inputs for models.
    """

    def __init__(self, **kwargs):

        self.dataset = kwargs
        self.primary_inputs = self.dataset.keys()  # used for diagram colors
        self.PROVIDES_REGISTRY = {}  # map keys to models -- "who provides X?"
        self.REQUIRES_REGISTRY = {}  # map models to keys -- "X requires what?"
        self.OPTIONAL_REGISTRY = {}  # map models to optional keys
        self.DEFERRED_REGISTRY = {}  # map models to deferred keys
        self.default_plugins()

    def default_plugins(self):
        """
        Initialize the default set of plugins by calling ``@self.plugin(...)``

        Abstract method intended to be overwritten by subclasses.
        """
        pass

    def trace(self, key, collapse_tree=False):
        """
        Recursively traverse the model graph and trace the dependencies
        required to calculate ``key``.

        Parameters
        ----------
        key : str
            The variable to trace.

        collapse_tree : bool, default False
            If True, flatten the depency tree into a list of dependencies.

        Returns
        -------
        A nested dict representing the dependency tree of ``key``.
        """
        provider = self.PROVIDES_REGISTRY.get(key, None)
        requires = self.REQUIRES_REGISTRY.get(provider, [])
        deferred = self.DEFERRED_REGISTRY.get(provider, [])
        optional = self.OPTIONAL_REGISTRY.get(provider, [])
        tree = {key: self.trace(key) for key in requires + deferred + optional}

        if collapse_tree:
            def collapse(tree):
                subtrees = [collapse(branch) for _, branch in tree.items()]
                collapsed_subtrees = sum(subtrees, [])
                return list(tree.keys()) + collapsed_subtrees
            tree = set(collapse(tree))
        return tree

    def model_inputs(self):
        """
        Return the list of variables that the current set of plugins requires.
        """
        return list(sorted(set(sum(self.REQUIRES_REGISTRY.values(), []))))

    def model_outputs(self):
        """
        Return the list of variables that the current set of plugins can
        calculate.
        """
        return list(sorted(self.PROVIDES_REGISTRY.keys()))

    def calculate(self, key, **kwargs):
        """
        Calculate and return the value of ``key``, dynamically resolving any
        dependencies.

        Parameters
        ----------
        key : str
            The variable to retrieve.

        kwargs :
            Extra parameters passed to the model plugin that provides ``key``.

        Returns
        -------
        The value of ``key`` returned from its provider model.
        """
        if key in self.dataset:
            return self.dataset[key]
        provider = self.PROVIDES_REGISTRY[key]
        provider(self.dataset, **kwargs)
        return self.dataset[key]

    def diagram(self, target=None):
        """
        Generate a diagram of a directed graph representing the current set of
        model plugins.  Edges represent model dependencies.

        Parameters
        ----------
        target : str, optional
            A model output to focus on.  Graph edges between this variable and
            its prerequisites will be drawn in bold.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        all_nodes = set(self.model_inputs() + self.model_outputs())
        G.add_nodes_from(all_nodes)

        edge_labels = {}
        for key in self.model_outputs():
            neighbors = self.trace(key).keys()
            provider = self.PROVIDES_REGISTRY[key]
            for neighbor in neighbors:
                G.add_edge(neighbor, key)
                edge_labels[(key, neighbor)] = provider.__name__

        # poached from https://stackoverflow.com/a/50048063/1641381
        df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
        for row, data in nx.shortest_path_length(G):
            for col, dist in data.items():
                df.loc[row, col] = dist
        df = df.fillna(df.max().max()/3)
        pos = nx.kamada_kawai_layout(G, dist=df.to_dict())

        colors = ['cyan' if key in self.primary_inputs else 'yellow'
                  for key in G.nodes]

        if target is not None:
            path_nodes = self.trace(target, collapse_tree=True)
            path_nodes.add(target)
            edges = G.edges()
            width = [3.0 if edge[1] in path_nodes else 1.0 for edge in edges]
            bold_labels1 = [edge[1] for edge in edges if edge[1] in path_nodes]
            bold_labels2 = [edge[0] for edge in edges if edge[1] in path_nodes]
            bold_labels = list(set(bold_labels1) | set(bold_labels2))
            normal_labels = [l for l in G.nodes() if l not in bold_labels]
        else:
            width = 1.0
            bold_labels = []
            normal_labels = G.nodes()

        nx.draw_networkx_nodes(G, pos, node_size=1000, node_shape='H',
                               node_color=colors)
        nx.draw_networkx_labels(G.subgraph(bold_labels), pos,
                                font_weight='bold')
        nx.draw_networkx_labels(G.subgraph(normal_labels), pos,
                                font_weight='normal')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=40,
                               edge_color='lightgrey', width=width)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if target is not None:
            plt.title(target)

    def to_dask(self):
        """ return a dask task graph representation of this model chain """
        dsk = {}
        for key in self.primary_inputs:
            dsk[key] = self.dataset[key]

        for key, provider in self.PROVIDES_REGISTRY.items():
            func = provider.__wrapped__
            reqs = self.REQUIRES_REGISTRY[provider]
            opts = self.OPTIONAL_REGISTRY[provider]

            # if the provider has multiple return values, we have to
            # have separate tasks to index them out.
            # this next bit is the hackiest python I've ever written...
            # don't ever do this
            enclosed_vars = dict(zip(
                provider.__code__.co_freevars,
                [x.cell_contents for x in provider.__closure__]
            ))
            returned_vars = enclosed_vars['provides']
            if len(returned_vars) == 1:
                dsk[key] = (func, *reqs, *opts)
            else:
                # if multiple return values, save the tuple separately.
                # note: this will get overwritten N times for N returned vars.
                # doesn't matter, it's the same every time
                tuple_key = "-".join(returned_vars)
                dsk[tuple_key] = (func, *reqs, *opts)
                # index out the one we want:
                idx = returned_vars.index(key)
                # can't use a lambda since it doesn't enclose scope
                indexer = functools.partial(lambda tup, idx: tup[idx], idx=idx)
                dsk[key] = (indexer, tuple_key)

        return dsk

    def plugin(self, requires, provides, deferred=[], optional=[]):
        """
        Register a model into the plugin architecture.  Intended for use as a
        decorator but can be called manually if needed.

        Parameters
        ----------
        requires : list
            The list of required inputs for the decorated function.  The values
            of these inputs will be passed as keyword arguments, so these names
            must match the argument names of the decorated function.

        provides : list
            The list of outputs of the decorated function.

        deferred : list, optional
            A list of deferred inputs for the decorated function.  These will
            be passed in as functions that calculate and return the value of
            the given input.  Useful for cases when a plugin needs to determine
            required inputs at runtime.

        optional : list, optional
            A list of optional inputs for the decorated function.  These values
            will be provided if they are already present in the dataset or can
            be calculated from the model chain, or passed as None otherwise.
        """

        def decorator(func):
            """
            Create a wrapper around the model to auto-calculate required inputs
            """
            def evaluate(key, ds, optional=False):
                """
                Return value of ``key`` if cached, calculate it otherwise.

                If ``optional`` is True and the value cannot be calculated,
                return ``None`` instead of raising a ValueError.
                """
                if key in ds:
                    # value was provided by user, or already calculated
                    _debug(f'requirement already satisfied: {key}')
                    value = ds[key]
                else:
                    # value not yet known
                    # determine what plugin can calculate it
                    provider = self.PROVIDES_REGISTRY.get(key, None)
                    if provider is None:
                        if optional:
                            value = None
                        else:
                            raise ValueError(
                                f'"{key}" has no registered provider and was '
                                'not defined at model chain creation'
                            )
                    else:
                        _debug(f'calculating requirement {key} '
                               f'with provider {provider.__name__}')
                        # run the plugin
                        # ignore the return value since it might be a tuple,
                        # so using the entries in ds guarantee single values
                        _ = provider(ds)
                        value = ds[key]
                return value

            @functools.wraps(func)
            def model(ds, **user_kwargs):
                """
                Wrapper around the model function.  Calculates input values and
                stores output values.
                """
                _debug(
                    f'checking prerequisites for {func.__name__}: {requires}')
                try:
                    # calculate any necessary params
                    required_args = {
                        key: evaluate(key, ds) for key in requires
                    }
                    # calculate any optional params
                    optional_args = {
                        key: evaluate(key, ds, optional=True)
                        for key in optional
                    }
                    # generate deferred evals for the deferred params.
                    # Note: lambdas don't enclose scope, so use
                    # functools.partial here instead.
                    deferred_args = {
                        key: functools.partial(evaluate, key=key, ds=ds)
                        for key in deferred
                    }
                except ValueError as e:
                    msg = str(e)
                    raise ValueError(
                        f'{func.__name__} -> {msg}'
                    )
                args = {**required_args, **deferred_args, **optional_args}
                _debug(f'calling {func.__name__} with '
                       f'requires={list(required_args.keys())}, '
                       f'deferred={list(deferred_args.keys())}, '
                       f'optional={list(optional_args.keys())}, '
                       f'and kwargs={list(user_kwargs.keys())}')
                # use positional args instead of double-splat for args
                # to allow chained plugins.
                # Requires py3's order-preserving dicts.
                value = func(*args.values(), **user_kwargs)
                if len(provides) > 1:
                    # if a model returns 2 things but should only provide one,
                    # eg it used to provide both and one got replaced by
                    # another model, we should only store the correct values.
                    for key, val in zip(provides, value):
                        # check that we are still the provider for each key
                        our_name = func.__name__
                        provider_name = self.PROVIDES_REGISTRY[key].__name__
                        if our_name != provider_name:
                            _debug(f'skipping return value {key} because '
                                   f'{our_name} has been replaced by '
                                   f'{provider_name}')
                            continue
                        ds[key] = val
                else:
                    ds[provides[0]] = value
                return value

            _debug(
                f'registering plugin {func.__name__}: {requires}->{provides}')
            self.REQUIRES_REGISTRY[model] = requires
            self.DEFERRED_REGISTRY[model] = deferred
            self.OPTIONAL_REGISTRY[model] = optional
            for key in provides:
                provider = self.PROVIDES_REGISTRY.get(key, None)
                if provider is not None:
                    msg = f"Replacing '{key}' provider '{provider.__name__}'" \
                          f" with new provider '{model.__name__}'"
                    _warn(msg)
                self.PROVIDES_REGISTRY[key] = model
            # return func, not model!  this allows us to chain @plugin calls
            # and reuse the same function for multiple models. (and across
            # multiple SAs).
            # The "downside" is that the original function itself is available
            # in its unwrapped form -- not currently a problem, and maybe a
            # good thing since we don't want the user to be able to use the
            # original functions to modify an SA namespace anyway.
            return func
        return decorator


class SystemAnalysis(ModelChain):
    """
    A high-level class for end-to-end PV system performance analysis.
    """

    def __init__(self, pv, poa=None, ghi=None, dni=None, dhi=None, albedo=0.25,
                 ambient_temperature=None, windspeed=None, gamma_pdc=None,
                 system_size=None, dc_model='pvwatts', g_ref=1000, t_ref=25,
                 normalized_low_cutoff=0, normalized_high_cutoff=np.inf,
                 poa_low_cutoff=200, poa_high_cutoff=1200,
                 cell_temperature_low_cutoff=-50,
                 cell_temperature_high_cutoff=110,
                 clip_quantile=0.98, clearsky_index_threshold=0.15,
                 aggregation_frequency='D', pv_tilt=None, pv_azimuth=None,
                 pvlib_location=None, pvlib_localized_pvsystem=None,
                 interpolation_frequency=None, max_timedelta=None,
                 temperature_model=None, rescale_poa=True,
                 clearsky_windspeed=0,
                 **kwargs):

        # It's desirable to use explicit parameters for documentation and
        # autocomplete purposes, but we still want to collect them all into
        # a dictionary for the modeling namespace.  Use locals() to grab them:
        # Important!  Don't create any variables before this line, or else
        # they'll be included in the modeling namespace.
        dataset = locals()
        dataset.pop('self')
        # flatten kwargs into the dataset as well
        dataset = {**dataset, **dataset.pop('kwargs')}
        dataset = {k: v for k, v in dataset.items() if v is not None}
        super().__init__(**dataset)
        if 'system_size' not in self.dataset:
            self.dataset['system_size'] = None

        if interpolation_frequency is not None:
            interpolation_keys = [
                'pv', 'poa', 'ghi', 'dni', 'dhi', 'ambient_temperature',
                'windspeed'
            ]
            for key in ['pv_tilt', 'pv_azimuth']:
                if isinstance(self.dataset[key], (pd.Series, pd.DataFrame)):
                    interpolation_keys.append(key)
            for key in interpolation_keys:
                if key not in self.dataset:
                    continue
                values = self.dataset[key]
                interpolated = normalization.interpolate(
                        values, interpolation_frequency, max_timedelta
                )
                self.dataset[key] = interpolated

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

        if result_to_plot == 'sensor':
            results_dict = self.dataset['sensor_soiling_results']
            aggregated = self.dataset['sensor_aggregated']
        elif result_to_plot == 'clearsky':
            results_dict = self.dataset['clearsky_soiling_results']
            aggregated = self.dataset['clearsky_aggregated']
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

        if result_to_plot == 'sensor':
            results_dict = self.dataset['sensor_soiling_results']
        elif result_to_plot == 'clearsky':
            results_dict = self.results['clearsky_soiling_results']

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

        if result_to_plot == 'sensor':
            results_dict = self.dataset['sensor_soiling_results']
            aggregated = self.dataset['sensor_aggregated']
        elif result_to_plot == 'clearsky':
            results_dict = self.dataset['clearsky_soiling_results']
            aggregated = self.dataset['clearsky_aggregated']

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
            poa = self.dataset['poa']
        elif poa_type == 'clearsky':
            poa = self.dataset['clearsky_poa']

        to_plot = pd.merge(pd.DataFrame(poa),
                           pd.DataFrame(self.dataset['pv_energy']),
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

        if result_to_plot == 'sensor':
            results_dict = self.dataset['sensor_degradation_results']
            aggregated = self.dataset['sensor_aggregated']
        elif result_to_plot == 'clearsky':
            results_dict = self.dataset['clearsky_degradation_results']
            aggregated = self.dataset['clearsky_aggregated']

        fig = plotting.degradation_summary_plots(
                results_dict['p50_rd'],
                results_dict['rd_confidence_interval'],
                results_dict['calc_info'],
                aggregated,
                **kwargs)
        return fig

    def default_plugins(self):
        """ The default set of RdTools PV analysis models, in plugin form. """

        @self.plugin(requires=['pv'], provides=['times'])
        def get_times(pv):
            return pv.index

        @self.plugin(requires=['pvlib_location', 'times'],
                     provides=['solar_position'])
        def get_solarposition(pvlib_location, times):
            return pvlib_location.get_solarposition(times)

        @self.plugin(requires=['pvlib_location', 'times'],
                     provides=['clearsky_irradiance'])
        def get_clearsky_irradiance(pvlib_location, times):
            return pvlib_location.get_clearsky(times)

        @self.plugin(requires=['pv_tilt', 'pv_azimuth', 'albedo',
                               'solar_position', 'clearsky_irradiance'],
                     provides=['clearsky_poa_unscaled'])
        def get_clearsky_poa(pv_tilt, pv_azimuth, albedo, solar_position,
                             clearsky_irradiance, **kwargs):
            poa = pvlib.irradiance.get_total_irradiance(
                    pv_tilt, pv_azimuth,
                    solar_position['apparent_zenith'],
                    solar_position['azimuth'],
                    clearsky_irradiance['dni'],
                    clearsky_irradiance['ghi'],
                    clearsky_irradiance['dhi'],
                    albedo=albedo, **kwargs)
            return poa['poa_global']

        @self.plugin(requires=['clearsky_poa_unscaled', 'poa', 'rescale_poa'],
                     provides=['clearsky_poa'])
        def rescale_clearsky_poa(clearsky_poa_unscaled, poa, rescale_poa):
            if rescale_poa:
                clearsky_poa = normalization.irradiance_rescale(
                    poa, clearsky_poa_unscaled, method='iterative'
                )
            else:
                clearsky_poa = clearsky_poa_unscaled
            return clearsky_poa

        @self.plugin(requires=['pvlib_location', 'times'],
                     provides=['clearsky_ambient_temperature'])
        def clearsky_ambient_temperature(pvlib_location, times):
            cs_tamb = clearsky_temperature.get_clearsky_tamb(
                times, pvlib_location.latitude, pvlib_location.longitude
            )
            return cs_tamb

        @self.plugin(requires=['pv', 'max_timedelta'], provides=['pv_energy'])
        def power_to_energy(pv, max_timedelta):
            energy = normalization.energy_from_power(
                    pv, max_timedelta=max_timedelta
            )
            return energy

        @self.plugin(requires=['clearsky_poa', 'clearsky_windspeed',
                               'clearsky_ambient_temperature'],
                     optional=['temperature_model'],
                     provides=['clearsky_cell_temperature'])
        @self.plugin(requires=['poa', 'windspeed', 'ambient_temperature'],
                     optional=['temperature_model'],
                     provides=['sensor_cell_temperature'])
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

        @self.plugin(requires=['pv_energy', 'clearsky_poa',
                               'clearsky_cell_temperature', 'gamma_pdc',
                               'g_ref', 't_ref', 'system_size'],
                     provides=['clearsky_normalized', 'clearsky_insolation'])
        @self.plugin(requires=['pv_energy', 'poa', 'sensor_cell_temperature',
                               'gamma_pdc', 'g_ref', 't_ref', 'system_size'],
                     provides=['sensor_normalized', 'sensor_insolation'])
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

        @self.plugin(requires=['clearsky_normalized',
                               'normalized_low_cutoff',
                               'normalized_high_cutoff'],
                     provides=['clearsky_normalized_filter'])
        @self.plugin(requires=['sensor_normalized',
                               'normalized_low_cutoff',
                               'normalized_high_cutoff'],
                     provides=['sensor_normalized_filter'])
        def normalized_filter(sensor_normalized, normalized_low_cutoff,
                              normalized_high_cutoff):
            filt = filtering.normalized_filter(
                sensor_normalized,
                normalized_low_cutoff,
                normalized_high_cutoff
            )
            return filt

        @self.plugin(requires=['clearsky_poa', 'poa_low_cutoff',
                               'poa_high_cutoff'],
                     provides=['clearsky_poa_filter'])
        @self.plugin(requires=['poa', 'poa_low_cutoff', 'poa_high_cutoff'],
                     provides=['sensor_poa_filter'])
        def poa_filter(poa, poa_low_cutoff, poa_high_cutoff):
            filt = filtering.poa_filter(poa, poa_low_cutoff, poa_high_cutoff)
            return filt

        @self.plugin(requires=['pv', 'clip_quantile'],
                     provides=['clip_filter'])
        def clip_filter(pv, clip_quantile):
            filt = filtering.clip_filter(pv, clip_quantile)
            return filt

        @self.plugin(requires=['clearsky_cell_temperature',
                               'cell_temperature_low_cutoff',
                               'cell_temperature_high_cutoff'],
                     provides=['clearsky_cell_temperature_filter'])
        @self.plugin(requires=['sensor_cell_temperature',
                               'cell_temperature_low_cutoff',
                               'cell_temperature_high_cutoff'],
                     provides=['sensor_cell_temperature_filter'])
        def cell_temperature_filter(sensor_cell_temperature,
                                    cell_temperature_low_cutoff,
                                    cell_temperature_high_cutoff):
            filt = filtering.tcell_filter(sensor_cell_temperature,
                                          cell_temperature_low_cutoff,
                                          cell_temperature_high_cutoff)
            return filt

        @self.plugin(requires=['poa', 'clearsky_poa',
                               'clearsky_index_threshold'],
                     provides=['clearsky_csi_filter'])
        def clearsky_csi_filter(poa, clearsky_poa, clearsky_index_threshold):
            filt = filtering.csi_filter(poa, clearsky_poa,
                                        clearsky_index_threshold)
            return filt

        @self.plugin(requires=['sensor_normalized_filter',
                               'sensor_poa_filter',
                               'clip_filter',
                               'sensor_cell_temperature_filter'],
                     provides=['sensor_overall_filter'])
        def sensor_filter(sensor_normalized_filter, sensor_poa_filter,
                          clip_filter, sensor_cell_temperature_filter):
            filt = sensor_normalized_filter \
                 & sensor_poa_filter \
                 & clip_filter \
                 & sensor_cell_temperature_filter
            return filt

        @self.plugin(requires=['clearsky_normalized_filter',
                               'clearsky_poa_filter', 'clip_filter',
                               'clearsky_cell_temperature_filter',
                               'clearsky_csi_filter'],
                     provides=['clearsky_overall_filter'])
        def clearsky_filter(clearsky_normalized_filter, clearsky_poa_filter,
                            clip_filter, clearsky_cell_temperature_filter,
                            clearsky_csi_filter):
            filt = clearsky_normalized_filter \
                 & clearsky_poa_filter \
                 & clip_filter \
                 & clearsky_cell_temperature_filter \
                 & clearsky_csi_filter
            return filt

        @self.plugin(requires=['clearsky_normalized',
                               'clearsky_insolation',
                               'clearsky_overall_filter',
                               'aggregation_frequency'],
                     provides=['clearsky_aggregated',
                               'clearsky_aggregated_insolation'])
        @self.plugin(requires=['sensor_normalized',
                               'sensor_insolation',
                               'sensor_overall_filter',
                               'aggregation_frequency'],
                     provides=['sensor_aggregated',
                               'sensor_aggregated_insolation'])
        def aggregate(normalized, insolation, overall_filter,
                      aggregation_frequency):
            norm = normalized[overall_filter]
            insol = insolation[overall_filter]
            aggregated = aggregation.aggregation_insol(norm, insol,
                                                       aggregation_frequency)
            aggregated_insolation = insol.resample(aggregation_frequency).sum()

            return aggregated, aggregated_insolation

        @self.plugin(requires=['clearsky_aggregated',
                               'clearsky_aggregated_insolation'],
                     provides=['clearsky_soiling_results'])
        @self.plugin(requires=['sensor_aggregated',
                               'sensor_aggregated_insolation'],
                     provides=['sensor_soiling_results'])
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

        @self.plugin(requires=['clearsky_aggregated'],
                     provides=['clearsky_degradation_results'])
        @self.plugin(requires=['sensor_aggregated'],
                     provides=['sensor_degradation_results'])
        def sensor_yoy_degradation(aggregated, **kwargs):
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
