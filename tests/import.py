from pyROGER.roger import RogerModel
import numpy as np

def CreateModel():
    xdata = np.random.normal(size = (1000, 5))
    ydata = np.round(np.random.uniform(size = (1000)))
    
    modelo = RogerModel(xdata, ydata, comments = 'some comment')
    return modelo

def CheckModel():
    modelo = CreateModel()
    assert modelo.comments[:12] == 'some comment'

CheckModel()
