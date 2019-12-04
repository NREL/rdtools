"""
Tests for the ModelChain base class
"""

from rdtools import ModelChain
import pytest
import logging

# pytest will capture and display log output for failed tests, so let's be sure
# to set up logging
console_handler = logging.StreamHandler()
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
console_handler.setFormatter(logging.Formatter(fmt))
logging.getLogger('rdtools').addHandler(console_handler)
logging.getLogger('rdtools').setLevel(logging.DEBUG)


@pytest.fixture
def mc():
    """make a ModelChain instance"""
    chain = ModelChain(x=5, y=10)

    @chain.plugin(requires=['x'], provides=['2x'])
    def double_x(x):
        return 2*x

    @chain.plugin(requires=['x', 'y'], provides=['x*y'])
    def x_times_y(x, y):
        return x * y

    @chain.plugin(requires=['x', 'y'], provides=['correct_order'])
    def xy_order(x, y):
        return x == chain.dataset['x'] and y == chain.dataset['y']

    return chain


def test_modelchain_simple(mc):
    """ basic functionality """
    expected = 10
    actual = mc.calculate('2x')
    assert expected == actual

    expected = 50
    actual = mc.calculate('x*y')
    assert expected == actual

    assert mc.calculate('correct_order')


def test_modelchain_cache(mc):
    """ make sure results are cached and not recalculated """
    already_run = False

    @mc.plugin(requires=['x'], provides=['z'])
    def only_works_once(x):
        nonlocal already_run
        if already_run:
            raise RuntimeError()
        already_run = True
        return x

    mc.calculate('z')
    # if it uses the cached value then this won't error:
    mc.calculate('z')
    assert already_run

    # hack out the cached value, forcing it to recalculate
    mc.dataset.pop('z')
    with pytest.raises(RuntimeError):
        mc.calculate('z')


def test_modelchain_override(mc):
    """ override default plugins with new ones """

    with pytest.warns(UserWarning):
        @mc.plugin(requires=['x'], provides=['2x'])
        def triple_x(x):
            return 3*x

    assert mc.calculate('2x') == 15

    # test multiple outputs
    @mc.plugin(requires=['x'], provides=['multiple1', 'multiple2'])
    def multiple(x):
        return x, -x

    # multiple outputs, full swap out
    with pytest.warns(UserWarning):
        @mc.plugin(requires=['x'], provides=['multiple1', 'multiple2'])
        def multiple_full_replacement(x):
            return 2*x, -2*x

    assert mc.calculate('multiple1') == 10  # calculated with replacement
    assert mc.calculate('multiple2') == -10  # calculated with replacement

    # multiple outputs, partial swap out
    @mc.plugin(requires=['x'], provides=['multiple3', 'multiple4'])
    def multiple_again(x):
        return x, -x

    with pytest.warns(UserWarning):
        @mc.plugin(requires=['x'], provides=['multiple3'])
        def multiple_partial_replacement(x):
            return 2*x

    assert mc.calculate('multiple3') == 10  # calculated with replacement
    assert mc.calculate('multiple4') == -5  # calculate with original


def test_modelchain_inputs_outputs(mc):
    assert mc.model_inputs() == ['x', 'y']
    assert mc.model_outputs() == ['2x', 'correct_order', 'x*y']


def test_modelchain_trace(mc):
    """ test that the recursive tracing works """
    trace = mc.trace('2x')
    assert trace == {'x': {}}

    @mc.plugin(requires=['x'], provides=['x1'])
    def x1(x):
        return x

    @mc.plugin(requires=['x1', 'y'], provides=['x2'])
    def x2(x1, y):
        return x1

    trace = mc.trace('x2')
    assert trace == {'y': {}, 'x1': {'x': {}}}


def test_modelchain_plugin_stacking(mc):
    """ make sure @plugin calls can be stacked for code reuse """

    @mc.plugin(requires=['x'], provides=['x^2'])
    @mc.plugin(requires=['y'], provides=['y^2'])
    def square(n):
        return n*n

    assert mc.calculate('x^2') == 25
    assert mc.calculate('y^2') == 100


def test_modelchain_multiple_returns(mc):
    """ make sure plugins with multiple return values are handled correctly """

    @mc.plugin(requires=['x'], provides=['ret1', 'ret2'])
    def multiple_returns(x):
        return (x*10, x*100)

    # ask for one value, get one value
    assert mc.calculate('ret1') == 50
    # but the other is still cached for later use
    assert mc.dataset['ret2'] == 500


def test_modelchain_subclass():
    """ test subclassing and default plugins """

    class Fibonacci(ModelChain):

        def default_plugins(self):
            @self.plugin(requires=['F0', 'F1'], provides=['F2'])
            @self.plugin(requires=['F1', 'F2'], provides=['F3'])
            @self.plugin(requires=['F2', 'F3'], provides=['F4'])
            @self.plugin(requires=['F3', 'F4'], provides=['F5'])
            @self.plugin(requires=['F4', 'F5'], provides=['F6'])
            def step(a, b):
                return a+b

    fib = Fibonacci(F0=0, F1=1)
    assert fib.calculate('F6') == 8


def test_modelchain_plugin_optional_inputs(mc):
    """ test optional inputs """

    def func(x, flag):
        if flag is not None:
            raise RuntimeError(f"flag value: {flag}")
        return x

    mc.plugin(requires=['x'], optional=['flag1'], provides=['output1'])(func)
    mc.plugin(requires=['x'], optional=['flag2'], provides=['output2'])(func)

    # specify flag1 but not flag2, so only flag1 will raise
    mc.dataset['flag1'] = True

    with pytest.raises(RuntimeError):
        mc.calculate('output1')

    assert mc.calculate('output2') == 5


def test_modelchain_plugin_deferred_inputs(mc):
    """ test deferred inputs """

    @mc.plugin(requires=['flag'], deferred=['x', 'z'], provides=['value'])
    def func(flag, x, z):
        if flag:
            return z()
        else:
            return x()

    # this fails because z can't be evaluated
    with pytest.raises(ValueError):
        mc.dataset['flag'] = True
        mc.calculate('value')

    # this works since the impossible z evaluation is deferred
    mc.dataset['flag'] = False
    assert mc.calculate('value') == 5
