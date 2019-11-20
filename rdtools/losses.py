
import pandas as pd
import numpy as np

def is_online(inverters, meter, inverter_limits=None):
    times = inverters.index
    inverters = inverters.fillna(0)

    if inverter_limits is None:
        inverter_limits = inverters.quantile(0.99)

    low_limit = 0.01 * inverter_limits

    # estimate the number of inverters online based on the mean reported
    # inverter power.  this is more robust to cases where some inverters
    # are online but not reporting data.  note that it assumes that the
    # mean reporting power is the same as the mean power -- not robust to
    # inverters of vastly different sizes with some not reporting.
    mean_inverter_power = inverters[inverters > low_limit].mean(axis=1)

    # apply correction for relative sizing based on who's reporting
    relative_sizing = inverters.divide(mean_inverter_power, axis=0).median()
    mean_inverter_power = inverters[inverters > low_limit] \
                                   .divide(relative_sizing, axis=1) \
                                   .mean(axis=1)

    # if no inverters appear online, just use meter / N
    n_inv = inverters.shape[1]
    mean_inverter_power[~(inverters > low_limit).any(axis=1)] = meter / n_inv

    # estimate number of inverters online
    n_inv_online = (meter / mean_inverter_power).round().fillna(0)
    n_inv_online = n_inv_online.clip(lower=0, upper=n_inv)

    # assume that if at least 1 inv is offline, any inv < 1% of max is
    # offline.  this falls down if invs are online but not reporting.
    # the less than 1% rule is to accomodate the rare case that an inverter
    # is reporting some nonzero production but is effectively offline.
    online_mask = pd.DataFrame(index=times, columns=inverters.columns,
                               data=True)
    online_mask.loc[n_inv_online < n_inv, :] = \
        inverters.gt(0.01 * inverter_limits)

    return online_mask
