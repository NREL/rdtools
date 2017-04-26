from normalization import normalize_with_sapm
from normalization import normalize_with_pvwatts
from normalization import sapm_dc_power
from degradation import degradation_ols
from degradation import degradation_classical_decomposition
from degradation import degradation_year_on_year
from filtering import get_period
from filtering import get_clearsky_irrad
from filtering import get_clearsky_poa
from filtering import detect_clearsky_params
from filtering import remove_cloudy_times
from filtering import remove_cloudy_days


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
