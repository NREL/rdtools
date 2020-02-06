'''Implementation of the QCRad algorithm for validating irradiance
obseervations.

@author: Cliff Hansen
'''

import pandas as pd
import numpy as np
from pvlib.tools import cosd

QCRAD_LIMITS = {'ghi_ub': {'mult': 1.5, 'exp': 1.2, 'min': 100},
                'dhi_ub': {'mult': 0.95, 'exp': 1.2, 'min': 50},
                'dni_ub': {'mult': 1.0, 'exp': 0.0, 'min': 0},
                'ghi_lb': -4, 'dhi_lb': -4, 'dni_lb': -4}

QCRAD_CONSISTENCY = {
    'ghi_ratio': {
        'low_zenith': {
            'zenith_bounds': [0.0, 75],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.92, 1.08]},
        'high_zenith': {
            'zenith_bounds': [75, 93],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.85, 1.15]}},
    'dhi_ratio': {
        'low_zenith': {
            'zenith_bounds': [0.0, 75],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.0, 1.05]},
        'high_zenith': {
            'zenith_bounds': [75, 93],
            'ghi_bounds': [50, np.Inf],
            'ratio_bounds': [0.0, 1.10]}}}


def _QCRad_ub(dni_extra, sza, lim):
    cosd_sza = cosd(sza)
    cosd_sza[cosd_sza < 0] = 0
    return lim['mult'] * dni_extra * cosd_sza**lim['exp'] + lim['min']


def _check_limits(val, lb=None, ub=None, lb_ge=False, ub_le=False):
    """ Returns True where lb < (or <=) val < (or <=) ub
    """
    if lb_ge:
        lb_op = np.greater_equal
    else:
        lb_op = np.greater
    if ub_le:
        ub_op = np.less_equal
    else:
        ub_op = np.less

    if (lb is not None) & (ub is not None):
        return lb_op(val, lb) & ub_op(val, ub)
    elif lb is not None:
        return lb_op(val, lb)
    elif ub is not None:
        return ub_op(val, ub)
    else:
        raise ValueError('must provide either upper or lower bound')


def check_ghi_limits(ghi, solar_zenith, dni_extra, limits=None):
    '''
    Tests for physical limits on GHI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as
    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    ghi_limit_flag : Series
        True if value passes physically-possible test
    '''
    if not limits:
        limits = QCRAD_LIMITS
    ghi_ub = _QCRad_ub(dni_extra, solar_zenith, limits['ghi_ub'])

    ghi_limit_flag = _check_limits(ghi, limits['ghi_lb'], ghi_ub)
    ghi_limit_flag.name = 'ghi_limit_flag'

    return ghi_limit_flag


def check_dhi_limits(dhi, solar_zenith, dni_extra, limits=None):
    '''
    Tests for physical limits on DHI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as
    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    dhi : Series
        Diffuse horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    dhi_limit_flag : Series
        True if value passes physically-possible test
    '''
    if not limits:
        limits = QCRAD_LIMITS

    dhi_ub = _QCRad_ub(dni_extra, solar_zenith, limits['dhi_ub'])

    dhi_limit_flag = _check_limits(dhi, limits['dhi_lb'], dhi_ub)
    dhi_limit_flag.name = 'dhi_limit_flag'

    return dhi_limit_flag


def check_dni_limits(dni, solar_zenith, dni_extra, limits=None):
    '''
    Tests for physical limits on DNI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as
    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    dni : Series
        Direct normal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    dni_limit_flag : Series
        True if value passes physically-possible test
    '''
    if not limits:
        limits = QCRAD_LIMITS

    dni_ub = _QCRad_ub(dni_extra, solar_zenith, limits['dni_ub'])

    dni_limit_flag = _check_limits(dni, limits['dni_lb'], dni_ub)
    dni_limit_flag.name = 'dni_limit_flag'

    return dni_limit_flag


def check_irradiance_limits(solar_zenith, dni_extra, ghi=None, dhi=None,
                            dni=None, limits=None):
    '''
    Tests for physical limits on GHI, DHI or DNI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as
    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    ghi : Series or None, default None
        Global horizontal irradiance in W/m^2
    dhi : Series or None, default None
        Diffuse horizontal irradiance in W/m^2
    dni : Series or None, default None
        Direct normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    ghi_limit_flag : Series or None, default None
        True if value passes physically-possible test
    dhi_limit_flag : Series or None, default None
    dhi_limit_flag : Series or None, default None

    References
    ----------
    [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.
    '''
    if not limits:
        limits = QCRAD_LIMITS

    if ghi is not None:
        ghi_limit_flag = check_ghi_limits(ghi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        ghi_limit_flag = None

    if dhi is not None:
        dhi_limit_flag = check_dhi_limits(dhi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dhi_limit_flag = None

    if dni is not None:
        dni_limit_flag = check_dni_limits(dni, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dni_limit_flag = None

    return ghi_limit_flag, dhi_limit_flag, dni_limit_flag


def _get_bounds(bounds):
    return (bounds['ghi_bounds'][0], bounds['ghi_bounds'][1],
            bounds['zenith_bounds'][0], bounds['zenith_bounds'][1],
            bounds['ratio_bounds'][0], bounds['ratio_bounds'][1])


def _check_irrad_ratio(ratio, ghi, sza, bounds):
    # unpack bounds dict
    ghi_lb, ghi_ub, sza_lb, sza_ub, ratio_lb, ratio_ub = _get_bounds(bounds)
    # for zenith set lb_ge to handle edge cases, e.g., zenith=0
    return ((_check_limits(sza, lb=sza_lb, ub=sza_ub, lb_ge=True)) &
            (_check_limits(ghi, lb=ghi_lb, ub=ghi_ub)) &
            (_check_limits(ratio, lb=ratio_lb, ub=ratio_ub)))


def check_irradiance_consistency(ghi, solar_zenith, dni_extra, dhi, dni,
                                 param=None):
    '''
    Checks consistency of GHI, DHI and DNI. Not valid for night time.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    dhi : Series
        Diffuse horizontal irradiance in W/m^2
    dni : Series
        Direct normal irradiance in W/m^2
    param : dict
        keys are 'ghi_ratio' and 'dhi_ratio'. For each key, value is a dict
        with keys 'high_zenith' and 'low_zenith'; for each of these keys,
        value is a dict with keys 'zenith_bounds', 'ghi_bounds', and
        'ratio_bounds' and value is an ordered pair [lower, upper]
        of float.

    Returns
    -------
    consistent_components : Series
        True if ghi, dhi and dni components are consistent.
    diffuse_ratio_limit : Series
        True if diffuse to ghi ratio passes limit test.

    References
    ----------
    [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.
    '''

    if not param:
        param = QCRAD_CONSISTENCY

    # sum of components
    component_sum = dni * cosd(solar_zenith) + dhi
    ghi_ratio = ghi / component_sum
    dhi_ratio = dhi / ghi

    bounds = param['ghi_ratio']
    consistent_components = (
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=solar_zenith, bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=solar_zenith, bounds=bounds['low_zenith']))
    consistent_components.name = 'consistent_components'

    bounds = param['dhi_ratio']
    diffuse_ratio_limit = (
        _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                           bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                           bounds=bounds['low_zenith']))
    diffuse_ratio_limit.name = 'diffuse_ratio_limit'

    return consistent_components, diffuse_ratio_limit
