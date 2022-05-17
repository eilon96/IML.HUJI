from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

def classification_error( X: np.ndarray, y: np.ndarray, threshold: float, sign: int)->float:
    error = 0
    sign_y = np.sign(y)
    for i in range(X.shape[0]):
        if (X[i] >= threshold and sign_y[i] == -sign) or (X[i] < threshold and sign_y[i] == sign):
            error += 1
    return error / len(y)

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        data = {}
        for s in [-1, 1]:
            threshold_and_errors = [self._find_threshold(X[:, i], y, s) for i in range(X.shape[1])]
            best_feature = threshold_and_errors.index(min(threshold_and_errors, key=lambda value: value[1]))
            data[s] = best_feature, threshold_and_errors[best_feature][0], threshold_and_errors[best_feature][1]


        if data[-1][2] < data[1][2]:
            self.sign_ =  -1
        else:
            self.sign_ = 1
        self.j_ = data[self.sign_][0]
        self.threshold_ = data[self.sign_][1]


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y = []
        for sample in X[:, self.j_]:
            if sample >= self.threshold_:
                y.append(self.sign_)
            else:
                y.append(-self.sign_)

        return np.array(y)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        new_array = np.column_stack([values, labels])
        new_array = np.array(sorted(new_array.tolist(), key= lambda x: x[0]))
        possible_thresh_error = []
        num_of_positive = 0
        for j in range(len(new_array)):
            possible_thresh_error.append(num_of_positive)
            if np.sign(new_array[j][1]) == sign:
                num_of_positive += abs(new_array[j][1])
        possible_thresh_error = np.array(possible_thresh_error)
        num_of_negative = 0
        for j in reversed(range(len(new_array))):
            if np.sign(new_array[j][1]) == -sign:
                num_of_negative += abs(new_array[j][1])
            possible_thresh_error[j] += num_of_negative

        possible_thresh_error = np.array(possible_thresh_error)
        min_error = min(possible_thresh_error)
        min_index = possible_thresh_error.tolist().index(min_error)

        return new_array[min_index][0], min_error

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
        predicted = self.predict(X)
        sign_y = np.sign(y)
        error = len(y) - (sum(sign_y+predicted)/2)

        return error/len(y)



