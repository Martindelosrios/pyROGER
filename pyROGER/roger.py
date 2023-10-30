import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class RogerModel:
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
        Numpy array containing the real classes of the galaxies. Usually these
        are cluster, recent infalling, backsplash, infalling and interloper
        galaxies.
    ml_models : list
        List with the machine learning methods that will be used. The default
        methods are KNN, random forest and SVM.
    train_percentage : float
        Float between 0 and 1 representing the percentage of observations that
        will be used for training.
    comments: str
        String adding information for the trained model.

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

    def __init__(
        self,
        x_dataset: np.array,
        y_dataset: np.array,
        ml_models=[
            KNeighborsClassifier(n_neighbors=63),
            RandomForestClassifier(max_depth=2, random_state=0),
            svm.SVC(probability=True),
        ],
        train_percentage=0.75,
        labels=["CL", "RIN", "BS", "IN", "ITL"],
        comments=None,
    ):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.n_obs = len(x_dataset)
        self.ml_models = ml_models
        self.num_models = len(ml_models)
        self.train_percentage = train_percentage
        self.train_indices, self.test_indices = self.split()
        self.labels = labels
        self.trained = False
        if comments is not None:
            self.comments = comments

    def __repr__(self):
        if self.trained is False:
            output = self.comments + "\n NOT TRAINED YET"
        if self.trained is True:
            output = self.comments + "\n Ready to use \n"
            output = output + 'Available models: \n'
            for i, imod in enumerate(self.ml_models):
                output = output + '\n n_model:' + str(i) + '   ' + str(imod)
            #output = output + '\n' + str(output_aux)
        return output

    def split(self):
        """
        Function for splitting the dataset into train and test set.

        Returns
        -------

        Tuple with the indices of the training and testing sets.
        """
        ran_ind = np.random.choice(np.arange(self.n_obs), size=self.n_obs)
        train_indices = ran_ind[: round(self.train_percentage * self.n_obs)]
        test_indices = ran_ind[round(self.train_percentage * self.n_obs):]

        return train_indices, test_indices

    def train(self):
        """
        Function for training the machine learning methods.
        """

        for i in self.ml_models:
            i.fit(
                self.x_dataset[self.train_indices, :],
                self.y_dataset[self.train_indices],
            )
        self.trained = True
        return None

    def predict_class(self, data: np.array, n_model: int) -> np.array:
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
        print('Using model:' + '\n' + str(self.ml_models[n_model]) + '  n_model=' + str(n_model))
        orbital_class = self.ml_models[n_model].predict(data)
        return orbital_class

    def predict_prob(self, data: np.array, n_model: int) -> np.array:
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
        print('Using model:' + '\n' + str(self.ml_models[n_model]) + '  n_model=' + str(n_model))
        orbital_prob = self.ml_models[n_model].predict_proba(data)
        return orbital_prob

    def confusion_matrix(self, real_class=None, pred_class=None, n_model: int = 0):
        """
        Function for computing the confusion matrix.

        Parameters
        ----------
        n_model:int
            Number of the machine learning methods that will be used
            for the classification.

        real_class:np.array
            Numpy array containing the real classes. If nothing is pass, the
            function will use the testset classes.

        pred_class:np.array
            Numpy array containing the predicted classes. If nothing is pass,
            the function will use the classes predicted for the testset.
        """
        if real_class is None:
            real_class = self.y_dataset[self.test_indices]
        if pred_class is None:
            pred_class = self.predict_class(
                data=self.x_dataset[self.test_indices, :], n_model=n_model
            )
        conf_mat = sk.metrics.confusion_matrix(real_class, pred_class)
        return conf_mat
