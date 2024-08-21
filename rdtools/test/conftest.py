# from pkg_resources import parse_version
import pytest
from functools import wraps
import numpy as np
import pandas as pd
import itertools
import pvlib
import re

import rdtools

# rdtools_base_version = \
#     parse_version(parse_version(rdtools.__version__).base_version)


# decorator takes one argument: the base version for which it should fail
# for example @fail_on_rdtools_version('3.0.0') will cause a test to fail
# on rdtools versions 3.0.0, 3.0.0-alpha, 3.1.0, etc

# TODO: Check with Mike if we need this decorator
# def fail_on_rdtools_version(version):
#     # second level of decorator takes the function under consideration
#     def wrapper(func):
#         # third level defers computation until the test is called
#         # this allows the specific test to fail at test runtime,
#         # rather than at decoration time (when the module is imported)
#         @wraps(func)
#         def inner(*args, **kwargs):
#             # fail if the version is too high
#             if rdtools_base_version >= parse_version(version):
#                 pytest.fail('the tested function is scheduled to be '
#                             'removed in %s' % version)
#             # otherwise return the function to be executed
#             else:
#                 return func(*args, **kwargs)
#         return inner
#     return wrapper


def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'


def assert_warnings(messages, record):
    """
    Assert that every regex in ``messages`` matches
    a warning message in ``record``.

    Parameters
    ----------
    messages : list of str
        Regexes to match with warning messages
    record : list of warnings.WarningMessage
        A list of warnings, e.g. the one returned by the
        ``warnings.catch_warnings(record=True)`` context manager
    """
    warning_messages = [warning.message.args[0] for warning in record]
    for pattern in messages:
        found_match = any(re.match(pattern, msg) for msg in warning_messages)
        assert found_match, f"warning '{pattern}' not in {warning_messages}"


requires_pvlib_below_090 = \
    pytest.mark.skipif(parse_version(pvlib.__version__) > parse_version('0.8.1'),
                       reason='requires pvlib <= 0.8.1')


# %% Soiling fixtures

@pytest.fixture()
def soiling_times():
    tz = 'Etc/GMT+7'
    times = pd.date_range('2019/01/01', '2019/03/16', freq='D', tz=tz)
    return times


@pytest.fixture()
def soiling_normalized_daily(soiling_times):
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    np.random.seed(1977)
    noise = 0.01 * np.random.rand(75)
    normalized_daily = pd.Series(data=profile, index=soiling_times)
    normalized_daily = normalized_daily + noise

    return normalized_daily


@pytest.fixture()
def soiling_insolation(soiling_times):
    insolation = np.empty((75,))
    insolation[:30] = 8000
    insolation[30:45] = 6000
    insolation[45:] = 7000

    insolation = pd.Series(data=insolation, index=soiling_times)

    return insolation


@pytest.fixture()
def cods_times():
    tz = 'Etc/GMT+7'
    cods_times = pd.date_range('2019/01/01', '2021/01/01', freq='D', tz=tz)
    return cods_times


@pytest.fixture()
def cods_normalized_daily_wo_noise(cods_times):
    N = len(cods_times)
    interval_1 = 1 - 0.005 * np.arange(0, 25, 1)
    interval_2 = 1 - 0.002 * np.arange(0, 25, 1)
    interval_3 = 1 - 0.001 * np.arange(0, 25, 1)
    profile = np.concatenate((interval_1, interval_2, interval_3))
    repeated_profile = np.concatenate([profile for _ in range(int(np.ceil(N / 75)))])
    cods_normalized_daily_wo_noise = pd.Series(data=repeated_profile[:N], index=cods_times)
    return cods_normalized_daily_wo_noise


@pytest.fixture()
def cods_normalized_daily(cods_normalized_daily_wo_noise):
    N = len(cods_normalized_daily_wo_noise)
    np.random.seed(1977)
    noise = 1 + 0.02 * (np.random.rand(N) - 0.5)
    cods_normalized_daily = cods_normalized_daily_wo_noise * noise
    return cods_normalized_daily


@pytest.fixture()
def cods_normalized_daily_small_soiling(cods_normalized_daily_wo_noise):
    N = len(cods_normalized_daily_wo_noise)
    np.random.seed(1977)
    noise = 1 + 0.02 * (np.random.rand(N) - 0.5)
    cods_normalized_daily_small_soiling = cods_normalized_daily_wo_noise.apply(
        lambda row: 1-(1-row)*0.1) * noise
    return cods_normalized_daily_small_soiling


# %% Availability fixtures

ENERGY_PARAMETER_SPACE = list(itertools.product(
    [0, np.nan],  # outage value for power
    [0, np.nan, None],  # value for cumulative energy (None means real value)
    [0, 0.25, 0.5, 0.75, 1.0],  # fraction of comms outage that is power outage
))
# display names for the test cases.  default is just 0..N
ENERGY_PARAMETER_IDS = ["_".join(map(str, p)) for p in ENERGY_PARAMETER_SPACE]


def _generate_energy_data(power_value, energy_value, outage_fraction):
    """
    Generate an artificial mixed communication/power outage.
    """
    # a few days of clearsky irradiance for creating a plausible power signal
    times = pd.date_range('2019-01-01', '2019-01-15 23:59', freq='15min',
                          tz='US/Eastern')
    location = pvlib.location.Location(40, -80)
    # use haurwitz to avoid dependency on `tables`
    clearsky = location.get_clearsky(times, model='haurwitz')

    # just set base inverter power = ghi+clipping for simplicity
    base_power = clearsky['ghi'].clip(upper=0.8*clearsky['ghi'].max())

    inverter_power = pd.DataFrame({
        'inv0': base_power,
        'inv1': base_power*0.7,
        'inv2': base_power*1.3,
    })
    expected_power = inverter_power.sum(axis=1)
    # dawn/dusk points
    expected_power[expected_power < 10] = 0
    # add noise and bias to the expected power signal
    np.random.seed(2020)
    expected_power *= 1.05 + np.random.normal(0, scale=0.05, size=len(times))

    # calculate what part of the comms outage is a power outage
    comms_outage = slice('2019-01-03 00:00', '2019-01-06 00:00')
    start = times.get_loc(comms_outage.start)
    stop = times.get_loc(comms_outage.stop)
    power_outage = slice(start, int(start + outage_fraction * (stop-start)))
    expected_loss = inverter_power.iloc[power_outage, :].sum().sum() / 4
    inverter_power.iloc[power_outage, :] = 0
    meter_power = inverter_power.sum(axis=1)
    meter_energy = meter_power.cumsum() / 4
    # add an offset because in practice cumulative meter data never
    # actually starts at 0:
    meter_energy += 100

    meter_power[comms_outage] = power_value
    if energy_value is not None:
        meter_energy[comms_outage] = energy_value
    inverter_power.loc[comms_outage, :] = power_value

    expected_type = 'real' if outage_fraction > 0 else 'comms'

    return (meter_power,
            meter_energy,
            inverter_power,
            expected_power,
            expected_loss,
            expected_type)


@pytest.fixture(params=ENERGY_PARAMETER_SPACE, ids=ENERGY_PARAMETER_IDS)
def energy_data(request):
    # fixture sweeping across the entire parameter space
    power_value, energy_value, outage_fraction = request.param
    return _generate_energy_data(power_value, energy_value, outage_fraction)


@pytest.fixture
def energy_data_outage_single():
    # fixture only using a single parameter combination, for simpler tests.
    # has one real outage.
    outage_value, outage_fraction = np.nan, 0.25
    return _generate_energy_data(outage_value, outage_value, outage_fraction)


@pytest.fixture
def energy_data_comms_single():
    # fixture only using a single parameter combination, for simpler tests.
    # has one comms outage.
    outage_value, outage_fraction = np.nan, 0
    return _generate_energy_data(outage_value, outage_value, outage_fraction)


@pytest.fixture
def availability_analysis_object(energy_data_outage_single):
    (meter_power,
     meter_energy,
     inverter_power,
     expected_power,
     _, _) = energy_data_outage_single

    aa = rdtools.availability.AvailabilityAnalysis(meter_power, inverter_power, meter_energy,
                                                   expected_power)
    aa.run()
    return aa
