#About RdTools

RdTools is a set of Python tools for analysis of photovoltaic data.
In particular, high frequency (hourly or greater) PV production data is evaluated
over several years to obtain rates of performance degradation over time.

## Workflow

The degradation calculations consist of several steps illustrated here:
0. Import and preliminary calculations
1. Normalize data using a performance metric
2. Filter data that creates bias
3. Aggregate data
4. Analyze aggregated data to estimate the degradation rate

<img src="./screenshots/Workflow1.png" width="600" height="300" alt="RdTools Workflow"/>



## Install using pip

1. Clone or download the rdtools repository.
2. Navigate to repository: `cd rdtools`
3. Install via pip: `pip install .`

## Usage

```
import rdtools
```

For usage examples, look at the notebooks in rdtools/docs.

## Unit tests

To run tests from the main directory:
```
$ tests/run_tests
```
## Wiki

Check out the [wiki](https://github.com/NREL/rdtools/wiki) for information on development goals and framework.
