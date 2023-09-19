import numpy as np

from pyROGER.roger import RogerModel


def createmodel():
    xdata = np.random.normal(size=(1000, 5))
    ydata = np.round(np.random.uniform(size=(1000)))

    modelo = RogerModel(xdata, ydata, comments="some comment")
    return modelo


def test_createmodel():
    modelo = createmodel()
    assert modelo.comments[:12] == "some comment"
