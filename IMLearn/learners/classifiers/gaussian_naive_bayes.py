from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import pandas as pd
from scipy.stats import norm

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / len(X)

        data = pd.DataFrame(X)
        data["class"] = y
        groups = data.groupby("class")
        self.mu_ = groups.mean().to_numpy()

        self.vars_ = np.zeros([len(self.classes_), len(X[0])])
        for k in range(len(self.classes_)):
            for j in range(len(X[0])):
                for i in range(len(X)):
                    self.vars_[k][j] += (y[i] == self.classes_[k])*((X[i][j] - self.mu_[self.classes_[k]][j])**2)
                self.vars_[k][j] = self.vars_[k][j] / counts[self.classes_[k]]
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)
        responses = np.zeros(len(X))

        for j in range(len(X)):
            best_class = 0
            max_prob = 0
            for k in range(len(self.classes_)):
                if max_prob < likelihoods[j][k]:
                    max_prob = likelihoods[j][k]
                    best_class = k
            responses[j] = self.classes_[best_class]

        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.ones([len(X), len(self.classes_)])

        for j in range(len(X)):
            for k in range(len(self.classes_)):
                for i in range(len(X[0])):
                    nd = norm(self.mu_[self.classes_[k]][i], self.vars_[k][i]**0.5)
                    likelihoods[j][k] *= self.pi_[k] * nd.pdf(X[j][i])

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        responses = self.predict(X)
        return misclassification_error(y, responses)
