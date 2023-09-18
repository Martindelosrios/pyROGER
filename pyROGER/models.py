from pyROGER.roger import roger_model
import pandas as pd
import os
import numpy as np
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('pyROGER', '../dataset/')

data = pd.read_csv(DATA_PATH + '/highMass_trainset_roger1.csv', sep = ' ')
data = np.asarray(data)
data[np.where(data[:,2] == 'CL')[0], 2] = 0
data[np.where(data[:,2] == 'RIN')[0], 2] = 1
data[np.where(data[:,2] == 'BS')[0], 2] = 2
data[np.where(data[:,2] == 'IN')[0], 2] = 3
data[np.where(data[:,2] == 'ITL')[0], 2] = 4

data = data.astype('float64')

comments = """ ROGER model for isolated galaxy clusters with masses bigger than >10^{13} M_{sun}.
  This model is equivalent to the one presented in XXXX.XXXX and in ROGERWebsite.
"""
highMass_roger1 = roger_model(data[:,(0,1)], data[:,2], comments = comments)
#highMass_roger1.train()
