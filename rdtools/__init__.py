from rdtools.normalization import normalize_with_pvwatts
from rdtools.normalization import irradiance_rescale
from rdtools.normalization import energy_from_power
from rdtools.normalization import interpolate
from rdtools.normalization import normalize_with_expected_power
from rdtools.degradation import degradation_ols
from rdtools.degradation import degradation_classical_decomposition
from rdtools.degradation import degradation_year_on_year
from rdtools.aggregation import aggregation_insol
from rdtools.clearsky_temperature import get_clearsky_tamb
from rdtools.filtering import clearsky_filter
from rdtools.filtering import csi_filter
from rdtools.filtering import pvlib_clearsky_filter
from rdtools.filtering import poa_filter
from rdtools.filtering import tcell_filter
from rdtools.filtering import clip_filter
from rdtools.filtering import quantile_clip_filter
from rdtools.filtering import logic_clip_filter
from rdtools.filtering import xgboost_clip_filter
from rdtools.filtering import normalized_filter
from rdtools.filtering import two_way_window_filter
from rdtools.filtering import insolation_filter
from rdtools.filtering import hampel_filter
from rdtools.filtering import hour_angle_filter
from rdtools.filtering import directional_tukey_filter
# from rdtools.soiling import soiling_srr
# from rdtools.soiling import soiling_cods
# from rdtools.soiling import monthly_soiling_rates
# from rdtools.soiling import annual_soiling_ratios
from rdtools.analysis_chains import TrendAnalysis
from rdtools.plotting import degradation_summary_plots
from rdtools.plotting import tune_filter_plot
from rdtools.plotting import degradation_timeseries_plot
# from rdtools.plotting import soiling_monte_carlo_plot
# from rdtools.plotting import soiling_interval_plot
# from rdtools.plotting import soiling_rate_histogram
from rdtools.plotting import availability_summary_plots
from rdtools.availability import AvailabilityAnalysis
from rdtools.utilities import robust_quantile
from rdtools.utilities import robust_median
from rdtools.utilities import robust_mean

from . import _version
__version__ = _version.get_versions()['version']
