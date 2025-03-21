import numpy as np
import pandas as pd
import pkg_resources
import glob

from pyROGER.roger import RogerModel

DATA_PATH = pkg_resources.resource_filename("pyROGER", "../dataset/")

def list_saved_models():
    #saved_models = os.listdir(DATA_PATH)
    #saved_model_names = [name for name in saved_models if name.endswith('.joblib')]
    saved_model_names = glob.glob(DATA_PATH + '/*.joblib')
    print(saved_model_names)
     
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
"""
HighMassRoger1 = RogerModel(data[:, (0, 1)], data[:, 2], comments=comments)
