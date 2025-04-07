import numpy as np
from finiteelementanalysis import example_functions as ef


def test_add_x_y():
    x = 10
    y = 50
    assert np.isclose(ef.add_x_y(x, y), 60)
    x = 10.0
    y = 20.0
    assert np.isclose(ef.add_x_y(x, y), 30.0)
    x = -10.0
    y = 20.0
    assert np.isclose(ef.add_x_y(x, y), 10.0)
    x = 10.0
    y = -20.0
    assert np.isclose(ef.add_x_y(x, y), -10.0)
    x = -0.45
    y = 0.50
    assert np.isclose(ef.add_x_y(x, y), 0.05)


def test_hello_world():
    output = ef.hello_world()
    assert output == "hello world!"
