from normalization import normalize_with_sapm
from degradation import degradation_with_ols
from degradaion import degradation_classical_decomposition
from degradaion import degradation_year_on_year

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
