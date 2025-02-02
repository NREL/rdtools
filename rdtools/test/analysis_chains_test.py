from rdtools import TrendAnalysis, normalization, filtering
from conftest import assert_isinstance, assert_warnings
from rdtools.analysis_chains import ValidatedFilterDict
import pytest
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture
def basic_parameters():
    # basic parameters (no time series data) for the TrendAnalysis class

    parameters = dict(
        gamma_pdc=-0.005, temperature_model={"a": -3.47, "b": -0.0594, "deltaT": 3}
    )

    return parameters


@pytest.fixture
def cs_input():
    # basic parameters (no time series data) for the TrendAnalysis class
    loc = pvlib.location.Location(-23.762028, 133.874886, tz="Australia/North")
    cs_input = dict(
        pvlib_location=loc,
        pv_tilt=20,
        pv_azimuth=0,
        solar_position_method="ephemeris",  # just to improve test execution speed
    )

    return cs_input


@pytest.fixture
def degradation_trend(basic_parameters, cs_input):
    # smooth linear multi-year decline from 1.0 from degradation_test.py

    # hide this import inside the function so that pytest doesn't find it
    # and run the degradation tests as a part of this module
    from degradation_test import DegradationTestCase

    rd = -0.05
    input_freq = "h"
    degradation_trend = DegradationTestCase.get_corr_energy(rd, input_freq)
    tz = cs_input["pvlib_location"].tz
    return degradation_trend.tz_localize(tz)


@pytest.fixture
def sensor_parameters(basic_parameters, degradation_trend):
    # basic parameters plus time series data
    power = degradation_trend
    poa_global = power * 1000
    temperature_ambient = power * 0 + 25
    basic_parameters["pv"] = power
    basic_parameters["poa_global"] = poa_global
    basic_parameters["temperature_ambient"] = temperature_ambient
    basic_parameters["interp_freq"] = "h"
    return basic_parameters


@pytest.fixture
def sensor_analysis(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def sensor_analysis_nans(sensor_parameters):
    def randomly_replace_with(series, replace_with=0, fraction=0.1, seed=None):
        """
        Randomly replace a fraction of entries in a pandas Series with input value `replace_with`.

        Parameters:
        series (pd.Series): The input pandas Series.
        fraction (float): The fraction of entries to replace with 0. Default is 0.1 (10%).
        seed (int, optional): Seed for the random number generator for reproducibility.

        Returns:
        pd.Series: The modified pandas Series with some entries replaced by 0.
        """
        if seed is not None:
            np.random.seed(seed)

        # Determine the number of entries to replace
        n_replace = int(len(series) * fraction)

        # Randomly select indices to replace
        replace_indices = np.random.choice(series.index, size=n_replace, replace=False)

        # Replace selected entries with
        series.loc[replace_indices] = replace_with

        return series

    sensor_parameters_zeros = sensor_parameters.copy()
    sensor_parameters_nans = sensor_parameters.copy()

    sensor_parameters_zeros["pv"] = randomly_replace_with(sensor_parameters["pv"], seed=0)
    sensor_parameters_nans["pv"] = sensor_parameters_zeros["pv"].replace(0, np.nan)

    rd_analysis_zeros = TrendAnalysis(**sensor_parameters_zeros)
    rd_analysis_zeros.sensor_analysis(analyses=["yoy_degradation"])

    rd_analysis_nans = TrendAnalysis(**sensor_parameters_nans)
    rd_analysis_nans.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis_zeros, rd_analysis_nans


@pytest.fixture
def sensor_analysis_exp_power(sensor_parameters):
    power_expected = normalization.pvwatts_dc_power(
        sensor_parameters["poa_global"], power_dc_rated=1
    )
    sensor_parameters["power_expected"] = power_expected
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def sensor_analysis_aggregated_no_filter(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params_aggregated = {}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def clearsky_example_data(basic_parameters):
    # Import the example data
    file_url = (
        "https://datahub.duramat.org/dataset/"
        "a49bb656-7b36-437a-8089-1870a40c2a7d/"
        "resource/d2c3fcf4-4f5f-47ad-8743-fc29"
        "f1356835/download/pvdaq_system_4_2010"
        "-2016_subset_soil_signal.csv"
    )
    cache_file = "PVDAQ_system_4_2010-2016_subset_soilsignal.pickle"

    try:
        df = pd.read_pickle(cache_file)
    except FileNotFoundError:
        df = pd.read_csv(file_url, index_col=0, parse_dates=True)
        df.to_pickle(cache_file)

    # Specify the Metadata
    meta = {
        "latitude": 39.7406,
        "longitude": -105.1774,
        "timezone": "Etc/GMT+7",
        "gamma_pdc": -0.005,
        "azimuth": 180,
        "tilt": 40,
        "power_dc_rated": 1000.0,
        "temp_model_params": "open_rack_glass_polymer",
    }

    # Set the timezone
    df.index = df.index.tz_localize(meta["timezone"])

    # Select two years of data
    df_crop = df[df.index < (df.index[0] + pd.Timedelta(days=2 * 365 + 1))]

    basic_parameters["pv"] = df_crop["ac_power"]
    basic_parameters["poa_global"] = df_crop["poa_irradiance"]
    basic_parameters["temperature_ambient"] = df_crop["ambient_temp"]
    basic_parameters["interp_freq"] = "1min"

    # Set the pvlib location
    loc = pvlib.location.Location(meta["latitude"], meta["longitude"], tz=meta["timezone"])

    cs_input = dict(
        pvlib_location=loc,
        pv_tilt=meta["tilt"],
        pv_azimuth=meta["azimuth"],
        solar_position_method="ephemeris",  # just to improve test execution speed
    )
    return basic_parameters, cs_input


def test_interpolation(basic_parameters, degradation_trend):

    power = degradation_trend
    shifted_index = power.index + pd.to_timedelta("8 minutes")

    dummy_series = power * 0 + 25
    dummy_series.index = shifted_index

    basic_parameters["pv"] = power
    basic_parameters["poa_global"] = dummy_series
    basic_parameters["temperature_ambient"] = dummy_series
    basic_parameters["temperature_cell"] = dummy_series
    basic_parameters["windspeed"] = dummy_series
    basic_parameters["power_expected"] = dummy_series
    basic_parameters["interp_freq"] = "h"

    rd_analysis = TrendAnalysis(**basic_parameters)

    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.poa_global.index[1:]
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.temperature_ambient.index[1:]
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.temperature_cell.index[1:]
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.windspeed.index[1:]
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.power_expected.index[1:]
    )

    rd_analysis.set_clearsky(
        pv_azimuth=dummy_series,
        pv_tilt=dummy_series,
        poa_global_clearsky=dummy_series,
        temperature_cell_clearsky=dummy_series,
        temperature_ambient_clearsky=dummy_series,
    )

    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.pv_azimuth.index
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.pv_tilt.index
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.poa_global_clearsky.index
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.temperature_cell_clearsky.index
    )
    pd.testing.assert_index_equal(
        rd_analysis.pv_energy.index, rd_analysis.temperature_ambient_clearsky.index
    )


def test_sensor_analysis(sensor_analysis):
    yoy_results = sensor_analysis.results["sensor"]["yoy_degradation"]
    rd = yoy_results["p50_rd"]
    ci = yoy_results["rd_confidence_interval"]

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_nans(sensor_analysis_nans):
    rd_analysis_zeros, rd_analysis_nans = sensor_analysis_nans

    yoy_results_zeros = rd_analysis_zeros.results["sensor"]["yoy_degradation"]
    rd_zeros = yoy_results_zeros["p50_rd"]
    ci_zeros = yoy_results_zeros["rd_confidence_interval"]

    yoy_results_nans = rd_analysis_nans.results["sensor"]["yoy_degradation"]
    rd_nans = yoy_results_nans["p50_rd"]
    ci_nans = yoy_results_nans["rd_confidence_interval"]

    assert rd_zeros == pytest.approx(rd_nans, abs=1e-2)
    assert ci_zeros == pytest.approx(ci_nans, abs=1e-1)


def test_sensor_analysis_filter_components(sensor_analysis):
    columns = sensor_analysis.sensor_filter_components_aggregated.columns
    assert {'two_way_window_filter'} == set(columns)

    expected_columns = {'normalized_filter', 'poa_filter', 'tcell_filter', 'clip_filter'}
    columns = sensor_analysis.sensor_filter_components.columns
    assert expected_columns == set(columns)


def test_sensor_analysis_energy(sensor_parameters, sensor_analysis):
    sensor_parameters["pv"] = sensor_analysis.pv_energy
    sensor_parameters["pv_input"] = "energy"
    sensor_analysis2 = TrendAnalysis(**sensor_parameters)
    sensor_analysis2.pv_power = sensor_analysis.pv_power
    sensor_analysis2.sensor_analysis(analyses=["yoy_degradation"])
    yoy_results = sensor_analysis2.results["sensor"]["yoy_degradation"]
    rd = yoy_results["p50_rd"]
    ci = yoy_results["rd_confidence_interval"]

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_exp_power(sensor_analysis_exp_power):
    yoy_results = sensor_analysis_exp_power.results["sensor"]["yoy_degradation"]
    rd = yoy_results["p50_rd"]
    ci = yoy_results["rd_confidence_interval"]

    assert 0 == pytest.approx(rd, abs=1e-2)
    assert [0, 0] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_power_dc_rated(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    yoy_results = rd_analysis.results["sensor"]["yoy_degradation"]
    rd = yoy_results["p50_rd"]
    ci = yoy_results["rd_confidence_interval"]

    assert -1 == pytest.approx(rd, abs=1e-2)
    assert [-1, -1] == pytest.approx(ci, abs=1e-2)


def test_sensor_analysis_ad_hoc_filter(sensor_parameters):
    # by excluding all but a few points, we should trigger the <2yr error
    filt = pd.Series(False, index=sensor_parameters["pv"].index)
    filt.iloc[-100:] = True
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params["ad_hoc_filter"] = filt
    with pytest.raises(
        ValueError, match="Less than two years of data left after filtering"
    ):
        rd_analysis.sensor_analysis(analyses=["yoy_degradation"])


def test_sensor_analysis_aggregated_ad_hoc_filter(sensor_parameters):
    # by excluding all but a few points, we should trigger the <2yr error
    filt = pd.Series(False, index=sensor_parameters["pv"].index)
    filt = filt.resample("1D").first().dropna(how="all")
    filt.iloc[-500:] = True
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params_aggregated["ad_hoc_filter"] = filt
    with pytest.raises(
        ValueError, match="Less than two years of data left after filtering"
    ):
        rd_analysis.sensor_analysis(analyses=["yoy_degradation"])


def test_filter_components_poa(sensor_parameters):
    poa = sensor_parameters["poa_global"]
    poa_filter = (poa > 200) & (poa < 1200)
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    assert (poa_filter == rd_analysis.sensor_filter_components["poa_filter"]).all()


def test_filter_components_hour_angle(sensor_parameters, cs_input):
    lat = cs_input["pvlib_location"].latitude
    lon = cs_input["pvlib_location"].longitude
    hour_angle_filter = filtering.hour_angle_filter(sensor_parameters["pv"], lat, lon)
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.pvlib_location = cs_input['pvlib_location']
    rd_analysis.filter_params = {'hour_angle_filter': {}}
    rd_analysis.filter_params_aggregated = {}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    assert (hour_angle_filter[1:] ==
            rd_analysis.sensor_filter_components["hour_angle_filter"]).all()


def test_aggregated_filter_components(sensor_parameters):
    daily_ad_hoc_filter = pd.Series(True, index=sensor_parameters["pv"].index)
    daily_ad_hoc_filter[:600] = False
    daily_ad_hoc_filter = daily_ad_hoc_filter.resample("1D").first().dropna(how="all")
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params_aggregated["ad_hoc_filter"] = daily_ad_hoc_filter
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    assert (
        daily_ad_hoc_filter
        == rd_analysis.sensor_filter_components_aggregated["ad_hoc_filter"]
    ).all()


def test_filter_components_no_filters(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all filters
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    expected = pd.Series(True, index=rd_analysis.pv_energy.index)
    pd.testing.assert_series_equal(rd_analysis.sensor_filter, expected)
    assert rd_analysis.sensor_filter_components.empty


def test_aggregated_filter_components_no_filters(sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params_aggregated = {}  # disable all daily filters
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    expected = pd.Series(True, index=rd_analysis.pv_energy.index)
    daily_expected = expected.resample("1D").first().dropna(how="all")
    pd.testing.assert_series_equal(rd_analysis.sensor_filter_aggregated, daily_expected)
    assert rd_analysis.sensor_filter_components.empty


def test_aggregated_filter_components_two_way_window_filter(sensor_analysis_aggregated_no_filter):
    rd_analysis = sensor_analysis_aggregated_no_filter
    aggregated_no_filter = rd_analysis.sensor_aggregated_performance
    rd_analysis.filter_params_aggregated = {"two_way_window_filter": {}}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    daily_expected = filtering.two_way_window_filter(aggregated_no_filter)
    pd.testing.assert_series_equal(
        rd_analysis.sensor_filter_aggregated, daily_expected, check_names=False
    )


def test_aggregated_filter_components_insolation_filter(sensor_analysis_aggregated_no_filter):
    rd_analysis = sensor_analysis_aggregated_no_filter
    aggregated_no_filter = rd_analysis.sensor_aggregated_performance
    rd_analysis.filter_params_aggregated = {"insolation_filter": {}}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    daily_expected = filtering.insolation_filter(aggregated_no_filter)
    pd.testing.assert_series_equal(
        rd_analysis.sensor_filter_aggregated, daily_expected, check_names=False
    )


def test_aggregated_filter_components_hampel_filter(sensor_analysis_aggregated_no_filter):
    rd_analysis = sensor_analysis_aggregated_no_filter
    aggregated_no_filter = rd_analysis.sensor_aggregated_performance
    rd_analysis.filter_params_aggregated = {"hampel_filter": {}}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    daily_expected = filtering.hampel_filter(aggregated_no_filter)
    pd.testing.assert_series_equal(
        rd_analysis.sensor_filter_aggregated, daily_expected, check_names=False
    )


def test_aggregated_filter_components_directional_tukey_filter(
        sensor_analysis_aggregated_no_filter):
    rd_analysis = sensor_analysis_aggregated_no_filter
    aggregated_no_filter = rd_analysis.sensor_aggregated_performance
    rd_analysis.filter_params_aggregated = {"directional_tukey_filter": {}}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    daily_expected = filtering.directional_tukey_filter(aggregated_no_filter)
    pd.testing.assert_series_equal(
        rd_analysis.sensor_filter_aggregated, daily_expected, check_names=False
    )


@pytest.mark.parametrize("workflow", ["sensor", "clearsky"])
def test_filter_ad_hoc_warnings(workflow, sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.set_clearsky(
        pvlib_location=pvlib.location.Location(40, -80),
        poa_global_clearsky=rd_analysis.poa_global,
    )
    # warning for incomplete index
    ad_hoc_filter = pd.Series(True, index=sensor_parameters["pv"].index[:-5])
    rd_analysis.filter_params["ad_hoc_filter"] = ad_hoc_filter
    with pytest.warns(UserWarning, match="ad_hoc_filter index does not match index"):
        if workflow == "sensor":
            rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.sensor_filter_components
        else:
            rd_analysis.filter_params["clearsky_filter"] = {"model": "csi"}
            rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.clearsky_filter_components

    # missing values set to True
    assert components["ad_hoc_filter"].all()

    # warning about NaNs
    ad_hoc_filter = pd.Series(True, index=sensor_parameters["pv"].index, dtype="boolean")
    ad_hoc_filter.iloc[10] = pd.NA
    rd_analysis.filter_params["ad_hoc_filter"] = ad_hoc_filter
    with pytest.warns(
        UserWarning, match="ad_hoc_filter contains NaN values; setting to False"
    ):
        if workflow == "sensor":
            rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.sensor_filter_components
        else:
            rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.clearsky_filter_components

    # NaN values set to False
    assert not components["ad_hoc_filter"].iloc[10]
    assert components.drop(components.index[10])["ad_hoc_filter"].all()


@pytest.mark.parametrize("workflow", ["sensor", "clearsky"])
def test_aggregated_filter_ad_hoc_warnings(workflow, sensor_parameters):
    rd_analysis = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis.set_clearsky(
        pvlib_location=pvlib.location.Location(40, -80),
        poa_global_clearsky=rd_analysis.poa_global,
    )
    # disable all filters outside of CSI
    rd_analysis.filter_params = {"clearsky_filter": {"model": "csi"}}
    # warning for incomplete index
    daily_ad_hoc_filter = pd.Series(True, index=sensor_parameters["pv"].index[:-5])
    daily_ad_hoc_filter = daily_ad_hoc_filter.resample("1D").first().dropna(how="all")
    rd_analysis.filter_params_aggregated["ad_hoc_filter"] = daily_ad_hoc_filter
    with pytest.warns(UserWarning, match="ad_hoc_filter index does not match index"):
        if workflow == "sensor":
            rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.sensor_filter_components_aggregated
        else:
            rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
            components = rd_analysis.clearsky_filter_components_aggregated

    # missing values set to True
    assert components["ad_hoc_filter"].all()

    # warning about NaNs
    rd_analysis_2 = TrendAnalysis(**sensor_parameters, power_dc_rated=1.0)
    rd_analysis_2.set_clearsky(
        pvlib_location=pvlib.location.Location(40, -80),
        poa_global_clearsky=rd_analysis_2.poa_global,
    )
    # disable all filters outside of CSI
    rd_analysis_2.filter_params = {"clearsky_filter": {"model": "csi"}}
    daily_ad_hoc_filter = pd.Series(True, index=sensor_parameters["pv"].index)
    daily_ad_hoc_filter = (
        daily_ad_hoc_filter.resample("1D").first().dropna(how="all").astype("boolean")
    )
    daily_ad_hoc_filter.iloc[10] = pd.NA
    rd_analysis_2.filter_params_aggregated["ad_hoc_filter"] = daily_ad_hoc_filter
    with pytest.warns(
        UserWarning, match="ad_hoc_filter contains NaN values; setting to False"
    ):
        if workflow == "sensor":
            rd_analysis_2.sensor_analysis(analyses=["yoy_degradation"])
            components = rd_analysis_2.sensor_filter_components_aggregated
        else:
            rd_analysis_2.clearsky_analysis(analyses=["yoy_degradation"])
            components = rd_analysis_2.clearsky_filter_components_aggregated

    # NaN values set to False
    assert not components["ad_hoc_filter"].iloc[10]
    assert components.drop(components.index[10])["ad_hoc_filter"].all()


def test_cell_temperature_model_invalid(sensor_parameters):
    wind = pd.Series(0, index=sensor_parameters["pv"].index)
    sensor_parameters.pop("temperature_model")
    rd_analysis = TrendAnalysis(
        **sensor_parameters, windspeed=wind, temperature_model={"bad": True}
    )
    with pytest.raises(ValueError, match="pvlib temperature_model entry is neither"):
        rd_analysis.sensor_analysis()


def test_no_gamma_pdc(sensor_parameters):
    sensor_parameters.pop("gamma_pdc")
    rd_analysis = TrendAnalysis(**sensor_parameters)

    with pytest.warns(UserWarning) as record:
        rd_analysis.sensor_analysis()

    assert_warnings(["Temperature coefficient not passed"], record)


@pytest.fixture
def clearsky_parameters(basic_parameters, sensor_parameters, cs_input, degradation_trend):
    # clear-sky weather data.  Uses TrendAnalysis's internal clear-sky
    # functions to generate the data.
    rd_analysis = TrendAnalysis(**sensor_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.filter_params["clearsky_filter"] = {"model": "csi"}
    rd_analysis._clearsky_preprocess()
    poa = rd_analysis.poa_global_clearsky
    clearsky_parameters = basic_parameters
    clearsky_parameters["poa_global"] = poa
    clearsky_parameters["pv"] = poa * degradation_trend
    return clearsky_parameters


@pytest.fixture
def clearsky_analysis(cs_input, clearsky_parameters):
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.filter_params["clearsky_filter"] = {"model": "csi"}
    rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def clearsky_pvlib_analysis(clearsky_example_data):
    clearsky_parameters_example, cs_input_example = clearsky_example_data
    rd_analysis = TrendAnalysis(**clearsky_parameters_example)
    rd_analysis.set_clearsky(**cs_input_example)
    rd_analysis.filter_params["clearsky_filter"] = {"model": "pvlib"}
    rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def clearsky_optional(cs_input, clearsky_analysis):
    # optional parameters to exercise other branches
    times = clearsky_analysis.poa_global.index
    extras = dict(
        poa_global_clearsky=clearsky_analysis.poa_global_clearsky,
        temperature_cell_clearsky=clearsky_analysis.temperature_cell_clearsky,
        temperature_ambient_clearsky=clearsky_analysis.temperature_ambient_clearsky,
        pv_tilt=pd.Series(cs_input["pv_tilt"], index=times),
        pv_azimuth=pd.Series(cs_input["pv_azimuth"], index=times),
        solar_position_method="ephemeris",  # just to improve test execution speed
    )
    return extras


@pytest.fixture
def sensor_clearsky_analysis(cs_input, clearsky_parameters):
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**cs_input)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params["sensor_clearsky_filter"] = {"model": "csi"}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis


@pytest.fixture
def sensor_clearsky_pvlib_analysis(clearsky_example_data):
    clearsky_parameters_example, cs_input_example = clearsky_example_data
    rd_analysis = TrendAnalysis(**clearsky_parameters_example)
    rd_analysis.set_clearsky(**cs_input_example)
    rd_analysis.filter_params = {}  # disable all index-based filters
    rd_analysis.filter_params["sensor_clearsky_filter"] = {"model": "pvlib"}
    rd_analysis.sensor_analysis(analyses=["yoy_degradation"])
    return rd_analysis


def test_clearsky_analysis(clearsky_analysis):
    yoy_results = clearsky_analysis.results["clearsky"]["yoy_degradation"]
    ci = yoy_results["rd_confidence_interval"]
    rd = yoy_results["p50_rd"]
    assert pytest.approx(rd, abs=1e-2) == -5.15
    assert pytest.approx(ci, abs=1e-2) == [-5.17, -5.13]


def test_clearsky_pvlib_analysis(clearsky_pvlib_analysis):
    yoy_results = clearsky_pvlib_analysis.results["clearsky"]["yoy_degradation"]
    ci = yoy_results["rd_confidence_interval"]
    rd = yoy_results["p50_rd"]
    assert pytest.approx(rd, abs=1e-2) == -1.589
    assert pytest.approx(ci, abs=1e-2) == [-2.417, -0.861]


def test_clearsky_analysis_filter_components(clearsky_analysis):
    columns = clearsky_analysis.clearsky_filter_components_aggregated.columns
    assert {'two_way_window_filter'} == set(columns)

    expected_columns = {'normalized_filter', 'poa_filter', 'tcell_filter',
                        'clip_filter', 'clearsky_filter'}
    columns = clearsky_analysis.clearsky_filter_components.columns
    assert expected_columns == set(columns)


def test_clearsky_analysis_optional(
    clearsky_analysis, clearsky_parameters, clearsky_optional
):

    clearsky_analysis.set_clearsky(**clearsky_optional)
    clearsky_analysis.clearsky_analysis()
    yoy_results = clearsky_analysis.results["clearsky"]["yoy_degradation"]
    ci = yoy_results["rd_confidence_interval"]
    rd = yoy_results["p50_rd"]
    print(f"ci:{ci}")
    assert pytest.approx(rd, abs=1e-2) == -5.15
    assert pytest.approx(ci, abs=1e-2) == [-5.17, -5.13]


def test_sensor_clearsky_analysis(sensor_clearsky_analysis):
    yoy_results = sensor_clearsky_analysis.results["sensor"]["yoy_degradation"]
    ci = yoy_results["rd_confidence_interval"]
    rd = yoy_results["p50_rd"]
    assert -5.18 == pytest.approx(rd, abs=1e-2)
    assert [-5.18, -5.18] == pytest.approx(ci, abs=1e-2)


def test_sensor_clearsky_pvlib_analysis(sensor_clearsky_pvlib_analysis):
    yoy_results = sensor_clearsky_pvlib_analysis.results["sensor"]["yoy_degradation"]
    ci = yoy_results["rd_confidence_interval"]
    rd = yoy_results["p50_rd"]
    assert -1.478 == pytest.approx(rd, abs=1e-2)
    assert [-2.495, -0.649] == pytest.approx(ci, abs=1e-2)


@pytest.fixture
def clearsky_analysis_exp_power(clearsky_parameters, clearsky_optional):
    power_expected = normalization.pvwatts_dc_power(
        clearsky_parameters["poa_global"], power_dc_rated=1
    )
    clearsky_parameters["power_expected"] = power_expected
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    rd_analysis.set_clearsky(**clearsky_optional)
    rd_analysis.filter_params["clearsky_filter"] = {"model": "csi"}
    rd_analysis.clearsky_analysis(analyses=["yoy_degradation"])
    return rd_analysis


def test_clearsky_analysis_exp_power(clearsky_analysis_exp_power):
    yoy_results = clearsky_analysis_exp_power.results["clearsky"]["yoy_degradation"]
    rd = yoy_results["p50_rd"]
    ci = yoy_results["rd_confidence_interval"]

    assert -5.128 == pytest.approx(rd, abs=1e-2)
    assert [-5.128, -5.127] == pytest.approx(ci, abs=1e-2)


def test_no_set_clearsky(clearsky_parameters):
    rd_analysis = TrendAnalysis(**clearsky_parameters)
    with pytest.raises(
        AttributeError, match="No poa_global_clearsky. 'set_clearsky' must be run"
    ):
        rd_analysis.clearsky_analysis()


def test_solar_position_method_passthrough(sensor_analysis, mocker):
    # verify that the solar_position_method kwarg is passed through to pvlib correctly
    spy = mocker.spy(pvlib.solarposition, "get_solarposition")
    for method in ["nrel_numpy", "ephemeris"]:
        sensor_analysis.set_clearsky(
            pvlib.location.Location(40, -80),
            pv_tilt=20,
            pv_azimuth=180,
            solar_position_method=method,
        )
        sensor_analysis._calc_clearsky_poa()
        assert spy.call_args[1]["method"] == method


def test_index_mismatch():
    # GH #277
    times = pd.date_range("2019-01-01", "2022-01-01", freq="15min")
    pv = pd.Series(1.0, index=times)
    # low-frequency weather inputs
    dummy_series = pd.Series(1.0, index=times[::4])
    keys = [
        "poa_global",
        "temperature_cell",
        "temperature_ambient",
        "power_expected",
        "windspeed",
    ]
    kwargs = {key: dummy_series.copy() for key in keys}
    rd_analysis = TrendAnalysis(pv, **kwargs)
    for key in keys:
        interpolated_series = getattr(rd_analysis, key)
        assert interpolated_series.index.equals(times)

    cs_keys = [
        "poa_global_clearsky",
        "temperature_cell_clearsky",
        "temperature_ambient_clearsky",
        "pv_azimuth",
        "pv_tilt",
    ]
    cs_kwargs = {key: dummy_series.copy() for key in cs_keys}
    rd_analysis.set_clearsky(**cs_kwargs)
    for key in cs_keys:
        interpolated_series = getattr(rd_analysis, key)
        assert interpolated_series.index.equals(times[1:])


@pytest.fixture
def soiling_parameters(basic_parameters, soiling_normalized_daily, cs_input):
    # parameters for soiling analysis with TrendAnalysis
    power = soiling_normalized_daily.resample("1h").interpolate()
    return dict(
        pv=power,
        poa_global=power * 0 + 1000,
        temperature_cell=power * 0 + 25,
        gamma_pdc=0,
        interp_freq="D",
    )


@pytest.fixture
def soiling_analysis_sensor(soiling_parameters):
    soiling_analysis = TrendAnalysis(**soiling_parameters)
    np.random.seed(1977)
    soiling_analysis.sensor_analysis(analyses=["srr_soiling"], srr_kwargs={"reps": 10})
    return soiling_analysis


@pytest.fixture
def soiling_analysis_clearsky(soiling_parameters, cs_input):
    soiling_analysis = TrendAnalysis(**soiling_parameters)
    soiling_analysis.set_clearsky(**cs_input)
    np.random.seed(1977)
    soiling_analysis.filter_params["clearsky_filter"] = {"model": "csi"}
    with pytest.warns(UserWarning, match="20% or more of the daily data"):
        soiling_analysis.clearsky_analysis(
            analyses=["srr_soiling"], srr_kwargs={"reps": 10}
        )
    return soiling_analysis


def test_srr_soiling(soiling_analysis_sensor):
    srr_results = soiling_analysis_sensor.results["sensor"]["srr_soiling"]
    sratio = srr_results["p50_sratio"]
    ci = srr_results["sratio_confidence_interval"]
    renorm_factor = srr_results["calc_info"]["renormalizing_factor"]
    print(f"soiling ci:{ci}")
    assert 0.965 == pytest.approx(
        sratio, abs=1e-3
    ), "Soiling ratio different from expected value in TrendAnalysis.srr_soiling"
    assert [0.96, 0.97] == pytest.approx(
        ci, abs=1e-2
    ), "Soiling confidence interval different from expected value in TrendAnalysis.srr_soiling"
    assert pytest.approx(
        renorm_factor, abs=1e-3
    ) == 0.977, "Renormalization factor different from expected value in TrendAnalysis.srr_soiling"


def test_plot_degradation(sensor_analysis):
    assert_isinstance(sensor_analysis.plot_degradation_summary("sensor"), plt.Figure)
    assert_isinstance(sensor_analysis.plot_pv_vs_irradiance("sensor"), plt.Figure)


def test_plot_cs(clearsky_analysis):
    assert_isinstance(
        clearsky_analysis.plot_degradation_summary("clearsky"), plt.Figure
    )
    assert_isinstance(clearsky_analysis.plot_pv_vs_irradiance("clearsky"), plt.Figure)


def test_plot_soiling(soiling_analysis_sensor):
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_monte_carlo("sensor"), plt.Figure
    )
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_interval("sensor"), plt.Figure
    )
    assert_isinstance(
        soiling_analysis_sensor.plot_soiling_rate_histogram("sensor"), plt.Figure
    )


def test_plot_soiling_cs(soiling_analysis_clearsky):
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_monte_carlo("clearsky"), plt.Figure
    )
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_interval("clearsky"), plt.Figure
    )
    assert_isinstance(
        soiling_analysis_clearsky.plot_soiling_rate_histogram("clearsky"), plt.Figure
    )


def test_errors(sensor_parameters, clearsky_analysis):

    rdtemp = TrendAnalysis(sensor_parameters["pv"])
    with pytest.raises(ValueError, match="poa_global must be available"):
        rdtemp._sensor_preprocess()

    # no temperature
    rdtemp = TrendAnalysis(
        sensor_parameters["pv"], poa_global=sensor_parameters["poa_global"]
    )
    with pytest.raises(ValueError, match="either cell or ambient temperature"):
        rdtemp._sensor_preprocess()

    # clearsky analysis with no tilt/azm
    del clearsky_analysis.pv_tilt
    clearsky_analysis.poa_global_clearsky = (
        None  # just needs to exist to test these errors
    )
    with pytest.raises(ValueError, match="pv_tilt and pv_azimuth must be provided"):
        clearsky_analysis._clearsky_preprocess()

    # clearsky analysis with no pvlib.loc
    del clearsky_analysis.pvlib_location
    with pytest.raises(ValueError, match="pvlib location must be provided"):
        clearsky_analysis._clearsky_preprocess()


@pytest.mark.parametrize(
    "method_name",
    [
        "plot_degradation_summary",
        "plot_soiling_monte_carlo",
        "plot_soiling_interval",
        "plot_soiling_rate_histogram",
        "plot_pv_vs_irradiance",
    ],
)
def test_plot_errors(method_name, sensor_analysis):
    func = getattr(sensor_analysis, method_name)
    with pytest.raises(ValueError, match="case must be either 'sensor' or 'clearsky'"):
        func(case="bad")


def test_plot_degradation_timeseries(sensor_analysis, clearsky_analysis):
    assert_isinstance(sensor_analysis.plot_degradation_timeseries("sensor"), plt.Figure)
    assert_isinstance(
        clearsky_analysis.plot_degradation_timeseries("clearsky"), plt.Figure
    )


def test_energy_from_power_hourly_data():

    times = pd.date_range("2019-01-01 00:00:00", periods=3, freq="h")
    pv = pd.Series([1.2, 2.8, 2.0], index=times)

    energy = normalization.energy_from_power(pv)
    pd.testing.assert_series_equal(energy, pv[1:], check_names=False)


def test_energy_from_power_shifted_hourly_data():

    times = pd.date_range("2019-01-01 00:30:00", periods=3, freq="h")
    pv = pd.Series([1.2, 2.8, 2.0], index=times)

    energy = normalization.energy_from_power(pv)
    pd.testing.assert_series_equal(energy, pv[1:], check_names=False)


def test_validated_filter_dict_initialization():
    valid_keys = ["key1", "key2"]
    filter_dict = ValidatedFilterDict(valid_keys, key1="value1", key2="value2")
    assert filter_dict["key1"] == "value1"
    assert filter_dict["key2"] == "value2"


def test_validated_filter_dict_invalid_key_initialization():
    valid_keys = ["key1", "key2"]
    with pytest.raises(KeyError, match="Key 'key3' is not a valid filter parameter."):
        ValidatedFilterDict(valid_keys, key1="value1", key3="value3")


def test_validated_filter_dict_setitem():
    valid_keys = ["key1", "key2"]
    filter_dict = ValidatedFilterDict(valid_keys)
    filter_dict["key1"] = "value1"
    assert filter_dict["key1"] == "value1"


def test_validated_filter_dict_setitem_invalid_key():
    valid_keys = ["key1", "key2"]
    filter_dict = ValidatedFilterDict(valid_keys)
    with pytest.raises(KeyError, match="Key 'key3' is not a valid filter parameter."):
        filter_dict["key3"] = "value3"


def test_validated_filter_dict_update():
    valid_keys = ["key1", "key2"]
    filter_dict = ValidatedFilterDict(valid_keys)
    filter_dict.update({"key1": "value1", "key2": "value2"})
    assert filter_dict["key1"] == "value1"
    assert filter_dict["key2"] == "value2"


def test_validated_filter_dict_update_invalid_key():
    valid_keys = ["key1", "key2"]
    filter_dict = ValidatedFilterDict(valid_keys)
    with pytest.raises(KeyError, match="Key 'key3' is not a valid filter parameter."):
        filter_dict.update({"key1": "value1", "key3": "value3"})


@pytest.mark.parametrize(
    "filter_param",
    [
        "normalized_filter",
        "poa_filter",
        "tcell_filter",
        "clip_filter",
        "hour_angle_filter",
        "clearsky_filter",
        "sensor_clearsky_filter",
        "ad_hoc_filter",
    ],
)
def test_valid_filter_params(sensor_analysis, filter_param):
    sensor_analysis.filter_params[filter_param] = {}
    assert filter_param in sensor_analysis.filter_params


def test_invalid_filter_params(sensor_analysis, filter_param="invalid_filter"):
    with pytest.raises(KeyError, match=f"Key '{filter_param}' is not a valid filter parameter."):
        sensor_analysis.filter_params[filter_param] = {}


@pytest.mark.parametrize(
    "filter_param_aggregated",
    [
        "two_way_window_filter",
        "insolation_filter",
        "hampel_filter",
        "directional_tukey_filter",
        "ad_hoc_filter",
    ],
)
def test_valid_filter_params_aggregated(sensor_analysis, filter_param_aggregated):
    sensor_analysis.filter_params_aggregated[filter_param_aggregated] = {}
    assert filter_param_aggregated in sensor_analysis.filter_params_aggregated


def test_invalid_filter_params_aggregated(
    sensor_analysis, filter_param_aggregated="invalid_filter"
):
    with pytest.raises(
        KeyError, match=f"Key '{filter_param_aggregated}' is not a valid filter parameter."
    ):
        sensor_analysis.filter_params_aggregated[filter_param_aggregated] = {}
