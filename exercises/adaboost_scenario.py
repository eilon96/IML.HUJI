import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    adaboost_model = AdaBoost(DecisionStump, n_learners)
    adaboost_model.fit(train_X, train_y)

    train_errors = []
    test_errors = []
    for i in range(n_learners):
        train_errors.append(adaboost_model.partial_loss(train_X, train_y, i + 1))
        test_errors.append(adaboost_model.partial_loss(test_X, test_y, i + 1))

    x = list(range(n_learners))
    go.Figure([
        go.Scatter(x=x, y=train_errors, mode='markers + lines', name='train'),
        go.Scatter(x=x, y=test_errors, mode='markers + lines', name='test')
    ]).update_layout(
        title="Training and test errors as a function of the number of fitted learners",
        xaxis=dict(title="number of learners")
    ).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{T={t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i, t in enumerate(T):
        fig.add_traces([
            decision_surface(lambda x: adaboost_model.partial_predict(x, T=t), lims[0], lims[1], showscale=False),
            go.Scatter(
                x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                marker=dict(
                    color=test_y,
                    colorscale=[custom[0], custom[-1]],
                    line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{Decision boundary for ensemble up to iteration T}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    size_lowest_test_error = int(np.argmin(test_errors))
    # lowest_test_error = test_errors[size_lowest_test_error]
    pred_test_y = adaboost_model.partial_predict(test_X, T=size_lowest_test_error)
    test_accuracy = accuracy(test_y, pred_test_y)

    go.Figure([
        decision_surface(lambda x: adaboost_model.partial_predict(x, T=size_lowest_test_error), lims[0], lims[1]),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                   marker=dict(color=test_y, colorscale=custom), showlegend=False)
    ], layout=go.Layout(
        title="Ensemble size with lowest test error - {0}, Accuracy - {1}".format(
            size_lowest_test_error, test_accuracy
        )
    )).show()

    # Question 4: Decision surface with weighted samples
    weights = (adaboost_model.D_ / np.max(adaboost_model.D_)) * 10

    go.Figure([
        decision_surface(adaboost_model.predict, lims[0], lims[1]),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                   marker=dict(color=train_y, colorscale=custom), marker_size=weights, showlegend=False)
    ], layout=go.Layout(title="Decision surface with weighted samples")).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)