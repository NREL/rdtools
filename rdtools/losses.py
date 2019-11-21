
import pandas as pd
import numpy as np

def is_online(inverters, meter, inverter_limits=None):
    times = inverters.index
    inverters = inverters.fillna(0)
    meter = meter.fillna(0)
    n_inv = inverters.shape[1]

    if inverter_limits is None:
        inverter_limits = inverters.quantile(0.99)

    # inverter-specific threshold to determine online or offline
    low_limit = 0.01 * inverter_limits

    # detect inverter downtime based on the mean reported
    # inverter power.  this is more robust to cases where inverters
    # are online but not reporting data.  note that it assumes that the
    # mean reporting power is the same as the mean power -- not robust to
    # inverters of vastly different sizes with some not reporting.
    mean_inverter_power = inverters[inverters > low_limit].mean(axis=1)

    # apply correction for relative sizing based on who's reporting
    relative_sizing = inverters.divide(mean_inverter_power, axis=0).median()
    mean_inverter_power = inverters[inverters > low_limit] \
                                   .divide(relative_sizing, axis=1) \
                                   .mean(axis=1)

    # if no inverters appear online, can't determine mean inverter power
    all_inverters_appear_offline = (inverters < low_limit).all(axis=1)
    mean_inverter_power[all_inverters_appear_offline] = meter / n_inv

    # if both meter and inverters look offline, we'll say it's offline
    meter_appears_offline = meter < low_limit.sum()
    site_offline = all_inverters_appear_offline & meter_appears_offline

    # calculate % diff between theoretical production if all invs were online
    # and actual meter readings
    meter_delta = 1 - meter / (n_inv * mean_inverter_power)

    # calculate the expected delta if the smallest inverter is offline:
    smallest_delta = (relative_sizing / relative_sizing.sum()).min()
    meter_appears_low = meter_delta > (0.75 * smallest_delta)

    # if meter is low enough relative to inverters that one might be offline,
    # AND some actually look offline, assume there are offline inverters
    inverters_appear_offline = ~(inverters > low_limit).all(axis=1)
    inverters_offline = inverters_appear_offline & meter_appears_low

    # assume that if at least 1 inv is offline, any inv < 1% of max is
    # offline.  this falls down if some invs are online but not reporting.
    online_mask = pd.DataFrame(index=times,
                               columns=inverters.columns,
                               data=True)
    online_mask.loc[inverters_offline, :] = inverters.gt(low_limit)
    online_mask.loc[site_offline, :] = False
    return online_mask
