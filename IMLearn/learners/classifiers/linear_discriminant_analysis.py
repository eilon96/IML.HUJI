from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import pandas as pd
from scipy.stats import multivariate_normal



class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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


        self.cov_ = np.zeros([len(X[0]), len(X[0])])
        for j, row in enumerate(X):
            self.cov_ += np.outer(row - self.mu_[y[j]], row - self.mu_[y[j]]) / (len(X) - len(self.classes_))

        self._cov_inv = np.linalg.inv(self.cov_)
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

        likelihoods = np.zeros([len(X), len(self.classes_)])

        for j in range(len(X)):
            for k in range(len(self.classes_)):
                nd = multivariate_normal(mean=self.mu_[self.classes_[k]], cov=self.cov_)
                likelihoods[j][k] = self.pi_[k] * nd.pdf(X[j])

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
