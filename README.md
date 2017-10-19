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


## Install using pip

1. Clone or download the rdtools repository.
2. Navigate to repository: `cd rdtools`
3. Install via pip: `pip install .`

## Usage

```
import rdtools
```

For usage examples, look at the notebooks in rdtools/docs.

## Citing RdTools

The underlying workflow of RdTools has been published in several places.  If you use RdTools in a published work, please cite the most appropriate of:

  - D. Jordan, C. Deline, S. Kurtz, G. Kimball, M. Anderson, "Robust PV Degradation Methodology and Application",
  IEEE Journal of Photovoltaics, 2017
  - D. C. Jordan, M. G. Deceglie, S. R. Kurtz, “PV degradation methodology comparison — A basis for a standard”, in 43rd IEEE Photovoltaic Specialists Conference, Portland, OR, USA, 2016, DOI: 10.1109/PVSC.2016.7749593.
  - E. Hasselbrink, M. Anderson, Z. Defreitas, M. Mikofski, Y.-C.Shen, S. Caldwell, A. Terao, D. Kavulak, Z. Campeau, D. DeGraaffE. Hasselbrink, “Validation of the PVLife model using 3 million module-years of live site data”, 39th IEEE Photovoltaic Specialists ConferenceIEEE PVSC, Tampa, FL, USA, 2013, p. 7 – 13, DOI: 10.1109/PVSC.2013.6744087.

## Unit tests

To run tests from the main directory:
```
$ tests/run_tests
```
## Further Instructions and Updates

Check out the [wiki](https://github.com/NREL/rdtools/wiki) for information on development goals and framework.
