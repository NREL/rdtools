"""Utility functions for rdtools."""


def robust_quantile(x, q):
    """
    Compute the q-th quantile of a time series (x), ignoring small values and NaN's.
    NaN's and small values [x < Q(x,q)/1000] are removed before calculating the quantile.
    This function ensures that time series with NaN's and distributions without
    NaN's return the same results. Should only be used if x is expected to be ≥0.

    Parameters
    ----------
    x : pandas.Series
        Input time series.
    q : float
        Probability value.

    Returns
    -------
    quantile : float
        The q-th quantile of x, ignoring small values and NaN's.
    """

    small = x.astype(float).fillna(0).quantile(q) / 1000
    q = x[x > small].quantile(q)

    return q


def robust_median(x, q=0.99):
    """
    Compute the median of a time series (x), ignoring small values and NaN's.
    NaN's and small values [Q(x,q)/1000] are removed before calculating the mean.
    This function ensures that time series with NaN's and distributions without
    NaN's return the same results.  Should only be used if x is expected to be ≥0.

    Parameters
    ----------
    x : pandas.Series
        Input time series.
    q : float, default 0.99
        Probability value to use for the small values threshold calculation [Q(x,q)/1000].

    Returns
    -------
    quantile : float
        The q-th quantile of x, ignoring small values and NaN's.
    """

    small = x.astype(float).fillna(0).quantile(q) / 1000
    mdn = x[x > small].median()

    return mdn


def robust_mean(x, q=0.99):
    """
    Compute the mean of a time series (x), ignoring small values and NaN's.
    NaN's and small values [x < Q(x,q)/1000] are removed before calculating the mean.
    This function ensures that time series with NaN's and distributions without
    NaN's return the same results.  Should only be used if x is expected to be ≥0.

    Parameters
    ----------
    x : pandas.Series
        Input time series.
    q : float, default 0.99
        Probability value to use for the small values threshold calculation.

    Returns
    -------
    quantile : float
        The q-th quantile of x, ignoring small values and NaN's.
    """

    small = x.astype(float).fillna(0).quantile(q) / 1000
    m = x[x > small].mean()

    return m
