import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import misclassification_error
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi



def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class
    Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets
    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("/Users/eilon/private/university/3/B/IML/תרגילים/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(p: Perceptron, x: np.ndarray, c: int):
            losses.append(p.loss(X, y))

        my_per = Perceptron(callback=callback)
        my_per.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plot = px.line(np.array(losses))


        plot.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("/Users/eilon/private/university/3/B/IML/תרגילים/IML.HUJI/datasets/" + f)

        # Fit models and predict over training set
        my_LDA = LDA()
        my_LDA.fit(X, y)
        my_LDA_predictions = my_LDA.predict(X)

        my_guassian = GaussianNaiveBayes()
        my_guassian.fit(X, y)
        my_guassian_predictions = my_guassian.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=
                            ["LDA data set: " + f + " accuracy: {0} ".format
                            (accuracy(y, my_LDA_predictions)),
                             "GaussianNaiveBayes data set: " + f + " accuracy: {0} ".format
                             (accuracy(y, my_guassian_predictions))],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                  marker=dict(color=my_LDA_predictions, symbol=y,
                                              line=dict(color="black", width=1),
                                              size=10)), row=1, col=1)

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                  marker=dict(color=my_guassian_predictions, symbol=y,
                                              line=dict(color="black", width=1),
                                              size=10)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        for k in my_LDA.classes_:
            fig.add_trace(go.Scatter(x=[my_LDA.mu_[k][0]], y=[my_LDA.mu_[k][1]], mode="markers",
                          marker=dict(symbol='x', color="black",
                                      line=dict(color="black", width=1),
                                      size=15)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[my_guassian.mu_[k][0]], y=[my_guassian.mu_[k][1]], mode="markers",
                                     marker=dict(symbol='x', color="black",
                                                 line=dict(color="black", width=1),
                                                 size=15)), row=1, col=2)

            fig.add_trace(get_ellipse(my_LDA.mu_[k], my_LDA.cov_), row=1, col=1)
            fig.add_trace(get_ellipse(my_guassian.mu_[k],  np.diag(my_guassian.vars_[k])), row=1, col=2)


        # Add ellipses depicting the covariances of the fitted Gaussians

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
