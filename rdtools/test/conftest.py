from pkg_resources import parse_version
import pytest
from functools import wraps

import rdtools

rdtools_base_version = \
    parse_version(parse_version(rdtools.__version__).base_version)


# decorator takes one argument: the base version for which it should fail
# for example @fail_on_rdtools_version('3.0.0') will cause a test to fail
# on rdtools versions 3.0.0, 3.0.0-alpha, 3.1.0, etc
def fail_on_rdtools_version(version):
    # second level of decorator takes the function under consideration
    def wrapper(func):
        # third level defers computation until the test is called
        # this allows the specific test to fail at test runtime,
        # rather than at decoration time (when the module is imported)
        @wraps(func)
        def inner(*args, **kwargs):
            # fail if the version is too high
            if rdtools_base_version >= parse_version(version):
                pytest.fail('the tested function is scheduled to be '
                            'removed in %s' % version)
            # otherwise return the function to be executed
            else:
                return func(*args, **kwargs)
        return inner
    return wrapper


def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'
