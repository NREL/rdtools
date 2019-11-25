.. currentmodule:: rdtools

##############
SystemAnalysis
##############

The :py:class:`~.system_analysis.SystemAnalysis` class provides a high-level
interface around the low-level functions in the RdTools submodules.
Its purpose is to eliminate boilerplate code when running an RdTools analysis
while still allowing customization as needed.
The class is centered around a dynamic "model chain" that allows the
analysis to only calculate the intermediate values that are necessary.
It also allows the user to swap in their own models to extend or replace the
built-in functionality that RdTools provides.

Basic usage looks like this:

::

    sa = SystemAnalysis(
        pv=df['power'],
        poa=df['poa'],
        ...
    )
        
    degradation_info = sa.calculate('sensor_degradation_results')
    print(f"P50 degradation rate: {degradation_info['p50_rd']}")


Model Chain
===========

Scientific modeling frequently involves connecting many models together.
For instance, a PV cell temperature model might have inputs
POA irradiance, ambient temperature, wind speed, and thermal parameters, and
its output might be an input of an expected DC power model.  Normally those
relationships would be encoded implicitly as a sequence of imperative
programming statements like:

::

    tcell = calculate_tcell(poa, tamb, wind, thermal_params)
    p_exp = pvwatts(poa, tcell, gamma_pmpp)

Each statement mutates the state of the program and the user is responsible for
making sure all of the inputs and outputs of each function are hooked up
correctly.  For example, the ``pvwatts`` function must be called `after`
``calculate_tcell`` because otherwise its input wouldn't have been
calculated yet.
Keeping track of this control flow is an implementation requirement that
is unrelated to the modeling logic itself -- ideally, the user could
just specify the set of available models and the execution environment would be
smart enough to figure out how to chain the models together to achieve some
goal.  
For more information on this distinction, see 
`Imperative <https://en.wikipedia.org/wiki/Imperative_programming>`_ vs 
`Declarative <https://en.wikipedia.org/wiki/Declarative_programming>`_
programming. 


The :py:class:`~.system_analysis.ModelChain` inside a
:py:class:`~.system_analysis.SystemAnalysis` object is
a way of achieving this abstraction.
Each model is associated with its abstract inputs and outputs and
the SystemAnalysis object figures out how to hook them all together.

Here is a diagram of the default RdTools model chain.  The dependencies of
``sensor_degradation_results`` are bolded, showing how the chain traces
relationships when calculating outputs.  Cyan nodes are considered "primary"
inputs in that they are not provided by any default model, while yellow nodes
are considered "derived" inputs in that there is a provider model for them.

.. image:: _static/model_graph_diagram.png
   :alt: A diagram of the default RdTools model chain, highlighting the dependencies of sensor_degradation_results


Plugins
=======

The models needed for an analysis are registered plugin-style.  Instead of
speciyfing when and where a model should be run, you register it with a 
:py:class:`~.system_analysis.SystemAnalysis` object as "requiring"
certain variables and "providing" others based on what the model needs as
inputs and what it calculates as outputs.  
For the above example of cell temperature, here's an example:

::

    @sa.plugin(requires=['poa', 'windspeed', 'ambient_temperature'],
               provides=['cell_temperature'])
    def calc_cell_temperature(poa, windspeed, ambient_temperature):
        tcell = pvlib.pvsystem.sapm_celltemp(
            poa_global=poa,
            wind_speed=windspeed,
            temp_air=ambient_temperature
        )
        return tcell['temp_cell']

Every plugin will follow this basic pattern of defining a model function and
decorating it with the metadata of what its inputs and outputs are.  
For more information on decorators in python, see 
`here <https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators>`_
or any of the numerous high-quality tutorials on the internet. 

If another model is registered like so:

::

    @sa.plugin(requires=['poa', 'cell_temperature', 'gamma_pmpp', 'nameplate'],
               provides=['expected_power'])
    def calc_expected_power(poa, cell_temperature, gamma_pmpp, nameplate):
        p_exp = namplate * (poa/1000) * (1 + gamma_pmpp * (tcell-25))
        return p_exp

The model chain recognizes that the ``calc_expected_power`` plugin
requires ``cell_temperature`` as an input and looks up what plugin can provide
it -- in this case, ``calc_cell_temperature``.  Providing requirement
relationships in a structured way like this allows the model chain to
dynamically calculate required inputs as needed, allowing the user to directly
request the value of ``expected_power`` without having to explicitly calculate
``cell_temperature`` beforehand -- the model chain will do it for you.

Plugins can also be "stacked" if you want to use the same code for multiple
calculation steps.  For instance:  since the default RdTools analysis
calculates POA filters based on both measured POA irradiance and modeled
clearsky POA irradiance, the same function can be used for both plugins by
stacking the ``@sa.plugin`` calls.  Here is the implementation inside RdTools:

::

    @self.plugin(requires=['clearsky_poa', 'poa_low_cutoff', 'poa_high_cutoff'],
                 provides=['clearsky_poa_filter'])
    @self.plugin(requires=['poa', 'poa_low_cutoff', 'poa_high_cutoff'],
                 provides=['sensor_poa_filter'])
    def poa_filter(poa, poa_low_cutoff, poa_high_cutoff):
        filt = filtering.poa_filter(poa, poa_low_cutoff, poa_high_cutoff)
        return filt


Registering Custom Plugins
==========================

The plugin architecture allows two cool things.  First, the default RdTools
models can be overridden with new models simply by registering a new function
that ``provides`` a given variable:

::

   In [1]: from rdtools import SystemAnalysis

   In [2]: sa = SystemAnalysis(pv=None) # make a dummy object

   In [3]: @sa.plugin(requires=['poa', 'ambient_temperature'], 
      ...:            provides=['sensor_cell_temperature'])
      ...: def my_tcell_plugin(poa, ambient_temperature):
      ...:     return 25 # dummy return value
      ...: 
   system_analysis.py:660: UserWarning: Replacing 'sensor_cell_temperature' provider 'sensor_cell_temperature' with new provider 'my_tcell_plugin'

Here the model chain recognizes that this new plugin conflicts with the default
cell temperature plugin and replaces the default with the new version.  This
allows detailed customization of the RdTools analysis pipeline without having
to worry about injecting the changes at just the right point in the calculation
process -- the results from the new plugin will get used anywhere that
``cell_temperature`` is needed.

Second, entirely new models can be inserted into the model chain.
For instance if you want to calculate monthly average cell temperature
weighted by POA irradiance, registering a plugin that provides a new variable
``poa_weighted_cell_temperature`` will automatically connect it into the
rest of the model chain and allow you to calculate it like anything else:

::

    from rdtools import SystemAnalysis
    # make a dummy object -- in reality, you would need to specify all the
    # requirements to calculate cell temperature here
    sa = SystemAnalysis(pv=None)
    @sa.plugin(requires=['poa', 'cell_temperature'],
               provides=['poa_weighted_cell_temperature'])
    def monthly_weighted_tcell(poa, cell_temperature):
        return (poa * cell_temperature).resample('m').sum() / poa.resample('m').sum()
    
    print(sa.calculate('poa_weighted_cell_temperature'))


Looking under the hood
======================

One advantage of using the high-level 
:py:class:`~.system_analysis.SystemAnalysis` API is to not have to worry about
details when running common analyses.  However, there are methods of inspecting
the sequence of calculations that an analysis takes.  

Logging
-------

RdTools outputs debugging information to a standard python logger,
accessible like so:

::
    
    In [1]: from rdtools import SystemAnalysis
       ...: import logging
       ...: 
       ...: console_handler = logging.StreamHandler()
       ...: fmt = '%(levelname)s - %(message)s'
       ...: console_handler.setFormatter(logging.Formatter(fmt))
       ...: logging.getLogger('rdtools').addHandler(console_handler)
       ...: logging.getLogger('rdtools').setLevel(logging.DEBUG)
       ...: 
       ...: sa = SystemAnalysis(pv=0)
       ...: 
    DEBUG - registering plugin get_times: ['pv']->['times']
    DEBUG - registering plugin get_solarposition: ['pvlib_location', 'times']->['solar_position']
    DEBUG - registering plugin get_clearsky_irradiance: ['pvlib_location', 'times']->['clearsky_irradiance']
    DEBUG - registering plugin get_clearsky_poa: ['pv_tilt', 'pv_azimuth', 'albedo', 'solar_position', 'clearsky_irradiance']->['clearsky_poa_unscaled']
    DEBUG - registering plugin rescale_clearsky_poa: ['clearsky_poa_unscaled', 'poa', 'rescale_poa']->['clearsky_poa']
    DEBUG - registering plugin clearsky_ambient_temperature: ['pvlib_location', 'times']->['clearsky_ambient_temperature']
    DEBUG - registering plugin power_to_energy: ['pv', 'max_timedelta']->['pv_energy']
    DEBUG - registering plugin cell_temperature: ['poa', 'windspeed', 'ambient_temperature']->['sensor_cell_temperature']
    DEBUG - registering plugin cell_temperature: ['clearsky_poa', 'clearsky_windspeed', 'clearsky_ambient_temperature']->['clearsky_cell_temperature']
    DEBUG - registering plugin normalize: ['pv_energy', 'poa', 'sensor_cell_temperature', 'gamma_pdc', 'g_ref', 't_ref', 'system_size']->['sensor_normalized', 'sensor_insolation']
    DEBUG - registering plugin normalize: ['pv_energy', 'clearsky_poa', 'clearsky_cell_temperature', 'gamma_pdc', 'g_ref', 't_ref', 'system_size']->['clearsky_normalized', 'clearsky_insolation']
    DEBUG - registering plugin normalized_filter: ['sensor_normalized', 'normalized_low_cutoff', 'normalized_high_cutoff']->['sensor_normalized_filter']
    DEBUG - registering plugin normalized_filter: ['clearsky_normalized', 'normalized_low_cutoff', 'normalized_high_cutoff']->['clearsky_normalized_filter']
    DEBUG - registering plugin poa_filter: ['poa', 'poa_low_cutoff', 'poa_high_cutoff']->['sensor_poa_filter']
    DEBUG - registering plugin poa_filter: ['clearsky_poa', 'poa_low_cutoff', 'poa_high_cutoff']->['clearsky_poa_filter']
    DEBUG - registering plugin clip_filter: ['pv', 'clip_quantile']->['clip_filter']
    DEBUG - registering plugin sensor_cell_temperature_filter: ['sensor_cell_temperature', 'cell_temperature_low_cutoff', 'cell_temperature_high_cutoff']->['sensor_cell_temperature_filter']
    DEBUG - registering plugin sensor_cell_temperature_filter: ['clearsky_cell_temperature', 'cell_temperature_low_cutoff', 'cell_temperature_high_cutoff']->['clearsky_cell_temperature_filter']
    DEBUG - registering plugin clearsky_csi_filter: ['poa', 'clearsky_poa', 'clearsky_index_threshold']->['clearsky_csi_filter']
    DEBUG - registering plugin sensor_filter: ['sensor_normalized_filter', 'sensor_poa_filter', 'clip_filter', 'sensor_cell_temperature_filter']->['sensor_overall_filter']
    DEBUG - registering plugin clearsky_filter: ['clearsky_normalized_filter', 'clearsky_poa_filter', 'clip_filter', 'clearsky_cell_temperature_filter', 'clearsky_csi_filter']->['clearsky_overall_filter']
    DEBUG - registering plugin aggregate: ['sensor_normalized', 'sensor_insolation', 'sensor_overall_filter', 'aggregation_frequency']->['sensor_aggregated', 'sensor_aggregated_insolation']
    DEBUG - registering plugin aggregate: ['clearsky_normalized', 'clearsky_insolation', 'clearsky_overall_filter', 'aggregation_frequency']->['clearsky_aggregated', 'clearsky_aggregated_insolation']
    DEBUG - registering plugin srr_soiling: ['sensor_aggregated', 'sensor_aggregated_insolation']->['sensor_soiling_results']
    DEBUG - registering plugin srr_soiling: ['clearsky_aggregated', 'clearsky_aggregated_insolation']->['clearsky_soiling_results']
    DEBUG - registering plugin sensor_yoy_degradation: ['sensor_aggregated']->['sensor_degradation_results']
    DEBUG - registering plugin sensor_yoy_degradation: ['clearsky_aggregated']->['clearsky_degradation_results']

This shows the process of registering the default set of RdTools plugins.  Now,
let's try to calculate a value that we didn't provide the prerequisites for:

::

    In [2]: sa.calculate('sensor_degradation_results')
    DEBUG - checking prerequisites for sensor_yoy_degradation: ['sensor_aggregated']
    DEBUG - calculating requirement sensor_aggregated with provider aggregate
    DEBUG - checking prerequisites for aggregate: ['sensor_normalized', 'sensor_insolation', 'sensor_overall_filter', 'aggregation_frequency']
    DEBUG - calculating requirement sensor_normalized with provider normalize
    DEBUG - checking prerequisites for normalize: ['pv_energy', 'poa', 'sensor_cell_temperature', 'gamma_pdc', 'g_ref', 't_ref', 'system_size']
    DEBUG - calculating requirement pv_energy with provider power_to_energy
    DEBUG - checking prerequisites for power_to_energy: ['pv', 'max_timedelta']
    DEBUG - requirement already satisfied: pv
    Traceback (most recent call last):
    
      File "<ipython-input-2-13e7e81d0e29>", line 1, in <module>
        sa.calculate('sensor_degradation_results')
    
      File "C:\Users\KANDERSO\projects\rdtools\rdtools\system_analysis.py", line 126, in calculate
        provider(self.dataset, **kwargs)
    
      File "C:\Users\KANDERSO\projects\rdtools\rdtools\system_analysis.py", line 277, in model
        f'{func.__name__} -> {msg}'
    
    ValueError: sensor_yoy_degradation -> aggregate -> normalize -> power_to_energy -> "max_timedelta" not specified and no provider registered

We can follow how the model chain tries to resolve dependencies.  In this case,
the only dependency for ``sensor_degradation_results`` is ``sensor_aggregated``,
which in turn requires the values ``sensor_normalized``, ``sensor_insolation``,
``sensor_overall_filter``, and ``aggregation_frequency``.  The chain iterates
through each of these in turn, starting with ``sensor_normalized``, and so on
down the dependency stack.  Eventually it reaches the point where ``pv_energy``
is required, which depends on ``pv`` and ``max_timedelta``, but there's no way
for it to get the value of ``max_timedelta``.

sa.trace()
----------

Model chains also have a ``.trace(key)`` method that calculates the dependency
graph (but not any of its values) for the given variable.  

::
    
    In [24]: import json
        ...: print(json.dumps(sa.trace('sensor_overall_filter'), indent=4))
    {
        "sensor_normalized_filter": {
            "sensor_normalized": {
                "pv_energy": {
                    "pv": {},
                    "max_timedelta": {}
                },
                "poa": {},
                "sensor_cell_temperature": {
                    "poa": {},
                    "windspeed": {},
                    "ambient_temperature": {},
                    "temperature_model": {}
                },
                "gamma_pdc": {},
                "g_ref": {},
                "t_ref": {},
                "system_size": {}
            },
            "normalized_low_cutoff": {},
            "normalized_high_cutoff": {}
        },
        "sensor_poa_filter": {
            "poa": {},
            "poa_low_cutoff": {},
            "poa_high_cutoff": {}
        },
        "clip_filter": {
            "pv": {},
            "clip_quantile": {}
        },
        "sensor_cell_temperature_filter": {
            "sensor_cell_temperature": {
                "poa": {},
                "windspeed": {},
                "ambient_temperature": {},
                "temperature_model": {}
            },
            "cell_temperature_low_cutoff": {},
            "cell_temperature_high_cutoff": {}
        }
    }

``sa.trace(key)`` returns a nested dictionary of dependencies.  In this case,
it shows that ``sensor_overall_filter`` depends on ``sensor_normalized_filter``, 
``sensor_poa_filter``, ``clip_filter``, and ``sensor_cell_temperature_filter``,
along with each of their respective dependencies.