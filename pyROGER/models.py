import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import importlib.resources

from pyROGER.roger import RogerModel


with importlib.resources.path("pyROGER", "dataset") as p:
    DATA_PATH = str(p)

def list_saved_models():
    '''
    Function for listing the available models.
    '''
    saved_model_names = glob.glob(DATA_PATH + '/*.joblib')
    return saved_model_names

data = pd.read_csv(DATA_PATH + "/highMass_trainset_roger1.csv", sep=" ")
data = np.asarray(data)
data[np.where(data[:, 2] == "CL")[0], 2] = 0
data[np.where(data[:, 2] == "RIN")[0], 2] = 1
data[np.where(data[:, 2] == "BS")[0], 2] = 2
data[np.where(data[:, 2] == "IN")[0], 2] = 3
data[np.where(data[:, 2] == "ITL")[0], 2] = 4

data = data.astype("float64")

comments = """ ROGER model for isolated galaxy clusters with masses
bigger than >10^{13} M_{sun}.
  This model is equivalent to the one presented in 2010.11959 and
  in https://mdelosrios.shinyapps.io/roger_shiny/.
  The saved models located in the datasets are:
   * HighMassRoger1_RF.joblib
   * HighMassRoger1_SVM.joblib
   * HighMassRoger1_KNN.joblib

  The input must be a np.array with shape (Ngal, 2) where,
  the first column must be the cluster-centric distance normalized to R200,
  and the second must be the relative velocity normalized to the cluster
  velocity dispersion, i.e:

    [:,0] = R / R200
    [:,1] = |\Delta V| / sigma

  The output for predict_prob() is a np.array with shape (Ngal, 5)
    [:,0] = P_cl: Probability of being a cluster galaxy.
    [:,1] = P_rin: Probability of being a backsplash galaxy.
    [:,2] = P_bs: Probability of being a recent infaller galaxy.
    [:,3] = P_in: Probability of being an infalling galaxy.
    [:,4] = P_itl: Probability of being a iterloper galaxy.
  
  The output for predict_class() is a np.array with shape (Ngal, ) with
  the predicted classes. 
     0: Cluster galaxy 
     1: Recent infaller galaxy 
     2: Backsplash galaxy 
     3: Infalling galaxy 
     4: Iterlooper galaxy 
"""
HighMassRoger1 = RogerModel(data[:, (0, 1)], data[:, 2], comments=comments)


comments = """ ROGER2 model for isolated galaxy clusters with masses
bigger than >10^{13} M_{sun}.
  The saved models located in the datasets are:
   * roger2_KNN.joblib

  The input must be a np.array with shape (Ngal, 3) where,
  the first column must be the log10 of the cluster mass,
  the second column must be the cluster-centric distance normalized to R200,
  and the third must be the relative velocity normalized to the cluster
  velocity dispersion, i.e:

    [:,0] = log10(M [M_{sun}])
    [:,1] = R / R200
    [:,2] = |\Delta V| / sigma

  The output predict_prob() is a np.array with shape (Ngal, 5)
    [:,0] = P_cl: Probability of being a cluster galaxy.
    [:,1] = P_bs: Probability of being a backsplash galaxy.
    [:,2] = P_rin: Probability of being a recent infaller galaxy.
    [:,3] = P_in: Probability of being an infalling galaxy.
    [:,4] = P_itl: Probability of being a iterloper galaxy.
  
  The output for predict_class() is a np.array with shape (Ngal, ) with
  the predicted classes. 
     0: Cluster galaxy 
     1: Backsplash galaxy 
     2: Recent infaller galaxy 
     3: Infalling galaxy 
     4: Iterlooper galaxy 
"""
Roger2 = RogerModel(data[:, (0, 1)], data[:, 2], comments=comments,  ml_models = [KNeighborsClassifier(n_neighbors=63)])
