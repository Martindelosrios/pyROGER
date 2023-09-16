import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# +
# Color scheme
cl_col = 'red'
rin_col = 'green'
bs_col = 'orange'
inf_col = 'blue'
itl_col = 'gray'

labels = ['CL', 'RIN', 'BS', 'IN', 'ITL']
# -

data = np.loadtxt('../data/tabla.txt', skiprows=1)

data.shape
# data[:,0] subhalo id (z=0)
# data[:,1] orbit class donde 0 es CL, 1 es RIN, 2 es BS, 3 es IN y 4 es ITL
# data[:,2] r
# data[:,3] v

# +
cl  = data[np.where(data[:,1] == 0)[0]]
rin = data[np.where(data[:,1] == 1)[0]]
bs  = data[np.where(data[:,1] == 2)[0]]
inf = data[np.where(data[:,1] == 3)[0]]
itl = data[np.where(data[:,1] == 4)[0]]

print('Hay ' + str(len(cl)) + ' cluster galaxies')
print('Hay ' + str(len(rin)) + ' recent infalling galaxies')
print('Hay ' + str(len(bs)) + ' backsplash galaxies')
print('Hay ' + str(len(inf)) + ' infalling galaxies')
print('Hay ' + str(len(itl)) + ' interlooper galaxies')
# -

plt.scatter(inf[:,2], inf[:,3], c = inf_col, marker = '<')
plt.scatter(itl[:,2], itl[:,3], c = itl_col, marker = '>')
plt.scatter(cl[:,2], cl[:,3], c = cl_col)
plt.scatter(rin[:,2], rin[:,3], c = rin_col, marker = '+')
plt.scatter(bs[:,2], bs[:,3], c = bs_col, marker = '*')
plt.xlabel('r')
plt.xlabel('v')

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# +
xtrainset = data[100:,(2,3)]
ytrainset = data[100:,1]

xtestset = data[:100,(2,3)]
ytestset = data[:100,1]


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(xtrainset, ytrainset)
# -

pred_class = clf.predict(xtestset)
real_class = ytestset

conf_mat = sk.metrics.confusion_matrix(real_class, pred_class)

disp = sk.metrics.ConfusionMatrixDisplay(conf_mat, display_labels= labels)
disp.plot()

int(1.8)


class roger_model():
    """
    Main class of pyROGER. This object contains all the information
    of the classification.

    Attributes
    ----------
    x_dataset : np.array
        Numpy array containing the features used for the classification. 
        Usually these will be the relative velocity normalized to the velocity
        dispersion and the normalized radii.    
    y_dataset : np.array
        Numpy array containing the real classes of the galaxies. Usually these are
        cluster, recent infalling, backsplash, infalling and interloper galaxies.
    ml_models : list
        List with the machine learning methods that will be used. The default
        methods are KNN, random forest and SVM. 
    train_percentage : float
        Float between 0 and 1 representing the percentage of observations that will
        be used for training.

    Methods
    -------
    splt()
        Split the data into training and testing set. It returns the indices
        of both sets.
        
    train()
        Train all the methods contained in ml_models.
        
    predict(data, n_model)
        Predict the class of the observation contained in data using the model
        number n_model.
        
    """
    def __init__(self, x_dataset:np.array, y_dataset:np.array, 
                 ml_models = [KNeighborsClassifier(n_neighbors=3),
                              RandomForestClassifier(max_depth=2, random_state=0),
                              svm.SVC(probability = True)], 
                 train_percentage = 0.75,
                labels = ['CL', 'RIN', 'BS', 'IN', 'ITL']):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.n_obs = len(x_dataset)
        self.ml_models = ml_models
        self.num_models = len(ml_models)
        self.train_percentage = train_percentage
        self.train_indices, self.test_indices = self.split() 
        self.labels = labels
        
    def split(self):
        """
        Function for splitting the dataset into train and test set.

        Returns
        -------

        Tuple with the indices of the training and testing sets.
        """
        ran_indices = np.random.choice(np.arange(self.n_obs), size = self.n_obs)
        train_indices = ran_indices[:round(self.train_percentage * self.n_obs)]
        test_indices = ran_indices[round(self.train_percentage * self.n_obs):]
        
        return train_indices, test_indices
        
    def train(self):
        """
        Function for training the machine learning methods.       
        """
        
        for i in self.ml_models:
            i.fit(self.x_dataset[self.train_indices,:], self.y_dataset[self.train_indices])
        return None
        
    def predict_class(self, data:np.array, n_model:int) -> np.array:
        """
        Function for predicting the orbital class.

        Parameters
        ----------

        data:np.array
            Numpy array containing the features that will be used for
            the classification.
            
        n_model:int
            Number of the machine learning methods that will be used
            for the classification.

        Returns
        -------

        Orbital class.
        """
        orbital_class = self.ml_models[n_model].predict(data)
        return orbital_class
        
    def predict_prob(self, data:np.array, n_model:int) -> np.array:
        """
        Function for predicting the orbital class probability.

        Parameters
        ----------

        data:np.array
            Numpy array containing the features that will be used for
            the classification.
            
        n_model:int
            Number of the machine learning methods that will be used
            for the classification.

        Returns
        -------

        Probability for each galaxy of belonging to each orbital class.
        """
        orbital_prob = self.ml_models[n_model].predict_proba(data)
        return orbital_prob

    def plot_confusion_matrix(self, n_model:int, real_class = None, pred_class = None):
        """
        Function for plotting the confusion matrix.

        Parameters
        ----------
        n_model:int
            Number of the machine learning methods that will be used
            for the classification.
            
        real_class:np.array
            Numpy array containing the real classes. If nothing is pass, the
            function will use the testset classes.
            
        pred_class:np.array
            Numpy array containing the predicted classes. If nothing is pass, the
            function will use the classes predicted for the testset.
        """
        if real_class is None: 
            real_class = self.y_dataset[self.test_indices]
        if pred_class is None: 
            pred_class = self.predict_class(data = self.x_dataset[self.test_indices,:], n_model = n_model)
        conf_mat = sk.metrics.confusion_matrix(real_class, pred_class)
        return sk.metrics.ConfusionMatrixDisplay(conf_mat, display_labels = self.labels)


modelo = roger_model(data[:,(2,3)], data[:,1])
modelo.train()

modelo.ml_models

modelo.predict_class(xtestset, 2)

disp = modelo.plot_confusion_matrix(2)
disp.plot()

type(disp)


