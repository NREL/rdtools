"""
The `losses` module contains functions for quantifying PV system performance.
"""

import pandas as pd
import numpy as np
import rdtools.normalization as normalization
import pvlib


def calculate_pr(power, expected_power, freq=None, filt=None, filter_na=True):
    """
    Calculate a system Performance Ratio (PR) using measured and expected
    system power measurements.

    Parameters
    ----------
    power : pd.Series
        Measured system power.
    expected_power : pd.Series
        Modeled system power, coicident with `power`.  To follow the NREL
        Weather-Corrected PR methodology, this should be calculated using the
        PVWatts-style model P = kWdc * POA/1000 * (1 - gamma_pmpp * [T*-Tcell])
    freq : pandas offset string, default None
        Optionally, calculate PR on a rolled-up basis.  For example, pass
        `freq='m'` to return a monthly PR series.  If omitted, a single value
        is returned that represents the PR for the entire dataset.
    filt : pd.Series, default None
        Optionally, a boolean Series to filter the timeseries data before
        calculating PR.  `True` values indicate measurements to keep.  This is
        useful for filtering out things like clipping or low-light conditions.
    filter_na : bool, default True
        If `True`, remove timestamps where either `power` or `expected_power`
        is missing.  Otherwise, treat null values like zero.

    Returns
    -------
    PR : float or pd.Series
        The calculated performance ratio as a single float (`freq is None`) or
        a pd.Series (otherwise).
    """

    df = pd.DataFrame({'observed': power, 'expected': expected_power})

    if filter_na:
        # if either column is null, null out the entire timestamp
        df.loc[df.isnull().any(axis=1), :] = np.nan
    else:
        df = df.fillna(0)

    if filt is not None:
        df.loc[~filt, :] = np.nan

    if freq is not None:
        df = df.resample(freq)

    rollup = df.sum()
    PR = rollup['observed'] / rollup['expected']
    return PR


def performance_ratio(system_size, gamma_pdc, power, poa, tamb=None,
                      tcell=None, wind=0.0, freq=None, clip_limit=None,
                      low_light_limit=0.0, tcell_ref=25,
                      temperature_model=None):
    """
    Calculate performance using the NREL Weather-Corrected Performance Ratio.

    Parameters
    ----------
    system_size : float
        System DC nameplate capacity.
    gamma_pdc : float
        Linear array efficiency temperature coefficient [1 / degree celsius].
    power : pd.Series
        Measured system power.  Units must match `system_size`.
    poa : pd.Series
        Plane-of-array irradiance measurements in W/m^2.
    tamb : pd.Series, default None
        Ambient temperature measurements in C.
        Either `tamb` or `tmod` must be specified.
    tcell : pd.Series, default None
        Back-of-module temperature measurements in C.
        Either `tamb` or `tcell` must be specified.
    wind : pd.Series, default 0.0
        Wind speed measurements in m/s.  If omitted, 0 m/s is used.
    freq : pandas offset string, default None
        Optionally, calculate PR on a rolled-up basis.  For example, pass
        `freq='m'` to return a monthly PR series.  If omitted, a single value
        is returned that represents the PR for the entire dataset.
    clip_limit : float, default None
        If specified, filter out times when *expected* power (not actual power)
        is above the system's clipping limit.  Note that this filter is not
        included in [1].
    low_light_limit : float, default 0.0
        If specified, filter out times when POA irradiance is below the
        low-light limit.  Note that this filter is not included in [1].
    tcell_ref : float, default 25
        The reference cell temperature in C.  In [1], the POA-weighted average
        Tcell is used so that annual temperature-adjusted PR equals the
        standard PR.  The default value of 25 C will yield PRs closer to 100%.
    temperature_model : str, default None
        An optional cell temperature model to use when calculating cell
        temperature with `pvlib.pvsystem.sapm_celltemp`.  Only used if `tcell`
        is not specified.

    Returns
    -------
    PR : float or pd.Series
        The calculated performance ratio as a single float (`freq is None`) or
        a pd.Series (otherwise).  Values are a ratio [0-1], not a percentage.

    Reference
    ---------
    [1] T. Dierauf et. al. "Weather-Corrected Performance Ratio" 2013 NREL
    Technical Report.  https://www.nrel.gov/docs/fy13osti/57991.pdf
    """

    if tcell is None:
        if temperature_model is None:
            tcell = pvlib.pvsystem.sapm_celltemp(poa, wind, tamb)
        else:
            tcell = pvlib.pvsystem.sapm_celltemp(poa, wind, tamb,
                                                 model=temperature_model)

    expected_power = normalization.pvwatts_dc_power(
            poa, system_size, tcell, T_ref=tcell_ref, gamma_pdc=gamma_pdc
    )

    filt = pd.Series(index=power.index, data=True)
    if clip_limit is not None:
        filt = filt & (expected_power < clip_limit)
    if low_light_limit is not None:
        filt = filt & (poa > low_light_limit)

    PR = calculate_pr(power, expected_power, freq=freq, filt=filt)
    return PR
