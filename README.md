# About RdTools

RdTools is a set of Python tools for analysis of photovoltaic data.
In particular, high frequency (hourly or greater) PV production data is evaluated
over several years to obtain rates of performance degradation over time.

## Workflow

0. Import and preliminary calculations
1. Normalize data using a performance metric
2. Filter data that creates bias
3. Aggregate data
4. Analyze aggregated data to estimate the degradation rate


<img src="./screenshots/Workflow1.png" width="600" height="331" alt="RdTools Workflow"/>

## Degradation Results

RdTools output shows filtered historical data along with a linear trend line.
However, the most accurate estimate of degradation over the analysis period in question
is the Year-on-year (YOY) distribution, which is a normal distribution, the central tendancy
of which is the most representative degradation rate.  Distribution width provides information
about the accuracy of the analysis.

<img src="./screenshots/Clearsky_result.png" width="600" height="456" alt="RdTools Result"/>


Two methods are available for system performance ratio calculation.  The sensor-based
approach assumes that site irradiance and temperature sensors are calibrated and in good repair.
Since this is not always the case, a 'Clear-Sky' workflow is provided that is based on
modeled temperature and irradiance.  Note that site irradiance data is still required to identify
clear-sky conditions to be analyzed.  In many cases, the 'Clear-Sky' analysis can identify conditions
of instrument errors, such as in the above analysis.


## Install RdTools using pip

RdTools can be installed into Python from the command line.

1. Clone or download the rdtools repository.
2. Navigate to the repository: `cd rdtools`
3. Install via pip: `pip install .`

## Usage

```
import rdtools
```

The most frequently used functions are:

`normalized, insolation = normalize_with_pvwatts(energy, pvwatts_kws)`
  Inputs: Pandas time series of raw energy, PVwatts dict for system analysis (poa_global, P_ref, T_cell, G_ref, T_ref, gamma_pdc)
  Outputs: Pandas time series of normalized energy and POA insolation

`poa_filter(poa); tcell_filter(Tcell); clip_filter(power); csi_filter(insolation, clearsky_insolation)`
  Inputs: Pandas time series of raw data to be filtered.
  Output: Boolean mask where `True` indicates acceptable data

`aggregation_insol(normalized, insolation, frequency='D')`
  Inputs: Normalized energy and insolation
  Output: Aggregated data, weighted by the insolation.

`degradataion_year_on_year(aggregated)`
  Inputs: Aggregated, normalized, filtered time series data
  Outputs: Tuple: `yoy_rd`: Degradation rate `yoy_ci`: Confidence interval `yoy_info`: associated analysis data

For additional usage examples, look at the notebooks in [rdtools/docs](./docs/degradation_example.ipynb).

## Citing RdTools

The underlying workflow of RdTools has been published in several places.  If you use RdTools in a published work, please cite the most appropriate of:

  - D. Jordan, C. Deline, S. Kurtz, G. Kimball, M. Anderson, "Robust PV Degradation Methodology and Application",
  IEEE Journal of Photovoltaics, 2017
  - D. C. Jordan, M. G. Deceglie, S. R. Kurtz, “PV degradation methodology comparison — A basis for a standard”, in 43rd IEEE Photovoltaic Specialists Conference, Portland, OR, USA, 2016, DOI: 10.1109/PVSC.2016.7749593.
  - E. Hasselbrink, M. Anderson, Z. Defreitas, M. Mikofski, Y.-C.Shen, S. Caldwell, A. Terao, D. Kavulak, Z. Campeau, D. DeGraaff, “Validation of the PVLife model using 3 million module-years of live site data”, 39th IEEE Photovoltaic Specialists Conference, Tampa, FL, USA, 2013, p. 7 – 13, DOI: 10.1109/PVSC.2013.6744087.

## Unit tests

To run tests from the main directory:
```
$ tests/run_tests
```
## Further Instructions and Updates

Check out the [wiki](https://github.com/NREL/rdtools/wiki) for information on development goals and framework.
