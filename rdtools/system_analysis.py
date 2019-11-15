'''
TODO: write this module description
'''

import pandas as pd
import numpy as np
import functools
import pvlib
import warnings
from rdtools import (
    normalization, clearsky_temperature, filtering
)


def _debug(msg):
    print("DEBUG:", msg)


def initialize_rdtools_plugins(sa):
    """
    Populate a SystemAnalysis object with the default set of RdTools plugins
    """

    @sa.plugin(requires=['poa', 'windspeed', 'ambient_temperature'],
               provides=['cell_temperature'])
    def calc_cell_temperature(poa, windspeed, ambient_temperature):
        tcell = pvlib.pvsystem.sapm_celltemp(
            poa_global=poa,
            wind_speed=windspeed,
            temp_air=ambient_temperature
        )
        return tcell['temp_cell']

    @sa.plugin(requires=['poa', 'cell_temperature', 'gamma_pdc', 'system_size',
                         'g_ref', 't_ref'],
               provides=['pvwatts_expected_power'])
    def pvwatts(poa, cell_temperature, gamma_pdc, system_size, g_ref,
                t_ref):
        pvwatts_kwargs = {
            "poa_global": poa,
            "P_ref": system_size,
            "T_cell": cell_temperature,
            "G_ref": g_ref,
            "T_ref": t_ref,
            "gamma_pdc": gamma_pdc
        }
        p_exp = normalization.pvwatts_dc_power(**pvwatts_kwargs)
        return p_exp

    @sa.plugin(requires=['pvlib_localized_pvsystem', 'dni', 'ghi', 'dhi',
                         'ambient_temperature', 'windspeed'],
               provides=['sapm_expected_power'])
    def sapm(pvlib_localized_pvsystem, dni, ghi, dhi, ambient_temperature,
             windspeed):
        sapm_kwargs = {
            "DNI": dni,
            "GHI": ghi,
            "DHI": dhi,
            "Temperature": ambient_temperature,
            "Wind Speed": windspeed,
        }
        df = pd.DataFrame(sapm_kwargs)
        p_exp = normalization.sapm_dc_power(pvlib_localized_pvsystem, df)
        return p_exp

    @sa.plugin(requires=['dc_model'],
               optional=['pvwatts_expected_power', 'sapm_expected_power'],
               provides=['expected_power'])
    def expected_power(dc_model, pvwatts_expected_power, sapm_expected_power):
        if dc_model == 'pvwatts':
            return pvwatts_expected_power()
        elif dc_model == 'sapm':
            return sapm_expected_power()
        raise ValueError(f"invalid dc_model value '{dc_model}', "
                         "must be either 'pvwatts' or 'sapm'")

    @sa.plugin(requires=['pv', 'expected_power'],
               provides=['normalized'])
    def normalize(pv, expected_power):
        return pv.div(expected_power)

    @sa.plugin(requires=['pv'], provides=['times'])
    def get_times(pv):
        return pv.index

    @sa.plugin(requires=['pvlib_location', 'times'],
               provides=['solar_position'])
    def get_solarposition(pvlib_location, times):
        return pvlib_location.get_solarposition(times)

    @sa.plugin(requires=['pvlib_location', 'times'],
               provides=['clearsky_irradiance'])
    def get_clearsky_irradiance(pvlib_location, times):
        return pvlib_location.get_clearsky(times)

    @sa.plugin(requires=['pv_tilt', 'pv_azimuth', 'albedo', 'solar_position',
                         'clearsky_irradiance'],
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

    @sa.plugin(requires=['clearsky_poa_unscaled', 'poa'],
               provides=['clearsky_poa'])
    def rescale_poa(clearsky_poa_raw, poa):
        clearsky_poa = normalization.irradiance_rescale(poa, clearsky_poa_raw,
                                                        method='iterative')
        return clearsky_poa

    @sa.plugin(requires=['pvlib_location', 'times'],
               provides=['clearsky_ambient_temperature'])
    def clearsky_ambient_temperature(pvlib_location, times):
        cs_tamb = clearsky_temperature.get_clearsky_tamb(
            times, pvlib_location.latitude, pvlib_location.longitude
        )
        return cs_tamb

    @sa.plugin(requires=['normalized',
                         'normalized_low_cutoff',
                         'normalized_high_cutoff'],
               provides=['normalized_filter'])
    def normalized_filter(normalized, normalized_low_cutoff,
                          normalized_high_cutoff):
        filt = filtering.normalized_filter(
                normalized, normalized_low_cutoff, normalized_high_cutoff
        )
        return filt

    @sa.plugin(requires=['poa', 'poa_low_cutoff', 'poa_high_cutoff'],
               provides=['poa_filter'])
    def poa_filter(poa, poa_low_cutoff, poa_high_cutoff):
        filt = filtering.poa_filter(poa, poa_low_cutoff, poa_high_cutoff)
        return filt

    @sa.plugin(requires=['pv', 'clip_quantile'],
               provides=['clip_filter'])
    def clip_filter(pv, clip_quantile):
        filt = filtering.clip_filter(pv, clip_quantile)
        return filt

    @sa.plugin(requires=['poa', 'clearsky_poa', 'clearsky_index_threshold'],
               provides=['csi_filter'])
    def csi_filter(poa, clearsky_poa, clearsky_index_threshold):
        filt = filtering.csi_filter(poa, clearsky_poa,
                                    clearsky_index_threshold)
        return filt


class SystemAnalysis:

    def __init__(self, pv, poa=None, ghi=None, ambient_temperature=None,
                 windspeed=None, albedo=0.25, system_size=None,
                 dc_model='pvwatts', g_ref=1000, t_ref=25, gamma_pdc=None,
                 normalized_low_cutoff=0, normalized_high_cutoff=np.inf,
                 poa_low_cutoff=200, poa_high_cutoff=1200,
                 cell_temperature_low_cutoff=-50,
                 cell_temperature_high_cutoff=110,
                 clip_quantile=0.98,
                 clearsky_index_threshold=0.15
                 ):

        # It's desirable to use explicit parameters for documentation and
        # autocomplete purposes, but we still want to collect them all into
        # a dictionary for the modeling namespace.  Use locals() to grab them:
        self.dataset = locals()
        self.dataset.pop('self')
        self.PROVIDES_REGISTRY = {}  # map keys to models
        self.REQUIRES_REGISTRY = {}  # map models to required keys
        self.OPTIONAL_REGISTRY = {}  # map models to optional keys
        initialize_rdtools_plugins(self)

    def trace(self, key):
        provider = self.PROVIDES_REGISTRY.get(key, None)
        requires = self.REQUIRES_REGISTRY.get(provider, [])
        optional = self.OPTIONAL_REGISTRY.get(provider, [])
        return {key: self.trace(key) for key in requires + optional}

    def model_inputs(self):
        return list(sorted(set(sum(self.REQUIRES_REGISTRY.values(), []))))

    def model_outputs(self):
        return list(sorted(self.PROVIDES_REGISTRY.keys()))

    def calculate(self, key, **kwargs):
        provider = self.PROVIDES_REGISTRY[key]
        return provider(self.dataset, **kwargs)

    def diagram(self):
        import networkx as nx
        G = nx.DiGraph()
        all_nodes = set(self.model_inputs() + self.model_outputs())
        G.add_nodes_from(all_nodes)
        for key in self.model_outputs():
            neighbors = self.trace(key).keys()
            for neighbor in neighbors:
                G.add_edge(neighbor, key)

        # poached from https://stackoverflow.com/a/50048063/1641381
        df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
        for row, data in nx.shortest_path_length(G):
            for col, dist in data.items():
                df.loc[row, col] = dist
        df = df.fillna(df.max().max())
        pos = nx.kamada_kawai_layout(G, dist=df.to_dict())

        nx.draw_networkx_nodes(G, pos, node_size=1000, node_shape='H')
        nx.draw_networkx_labels(G, pos, font_weight='bold')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=40,
                               edge_color='grey')

    def plugin(self, requires, provides, optional=[]):
        """
        Register a model into the plugin architecture
        """
        def decorator(func):
            """
            Create a wrapper around the model to auto-calculate required inputs
            """
            def evaluate(key, ds, defer=False):
                """
                Return cached value of key if known, calculate it otherwise.

                If defer is True, return a functools.partial for deferred
                evaluation.
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
                        raise KeyError(
                            f"'{key}' not specified and no provider registered"
                        )
                    _debug(f'calculating requirement {key} '
                           f'with provider {provider.__name__}')
                    # run the plugin
                    if defer:
                        value = functools.partial(provider, ds)
                    else:
                        value = provider(ds)
                return value

            @functools.wraps(func)
            def model(ds, **kwargs):
                _debug(
                    f'checking prerequisites for {func.__name__}: {requires}')
                # calculate any necessary params
                required_args = {key: evaluate(key, ds) for key in requires}
                # generate deferred evals for the optional params
                optional_args = {key: evaluate(
                    key, ds, True) for key in optional}
                args = {**required_args, **optional_args}
                _debug(f'calling {func.__name__} with '
                       f'requires={list(required_args.keys())}, '
                       f'optional={list(optional_args.keys())}, '
                       f'and kwargs={list(kwargs.keys())}')
                # TODO: this is a future bug -- if a model returns 2 things but
                # should only provide one (because eg it used to provide both
                # and one got replaced by another model) this will grab the
                # wrong value.  need to match up return values to provides
                value = func(**args, **kwargs)
                if len(provides) > 1:
                    for key, val in zip(provides, value):
                        ds[key] = val
                else:
                    ds[provides[0]] = value
                return value

            _debug(
                f'registering plugin {func.__name__}: {requires}->{provides}')
            self.REQUIRES_REGISTRY[model] = requires
            self.OPTIONAL_REGISTRY[model] = optional
            for key in provides:
                provider = self.PROVIDES_REGISTRY.get(key, None)
                if provider is not None:
                    warnings.warn(
                            f"Replacing '{key}' provider '{provider.__name__}'"
                            f" with new provider '{model.__name__}'"
                    )
                self.PROVIDES_REGISTRY[key] = model
            return model
        return decorator
