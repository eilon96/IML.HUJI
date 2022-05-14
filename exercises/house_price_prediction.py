from pandas import DataFrame

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
import pandas as pd


def process_date(data: DataFrame):
    # reformat date
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT000000')

    # splits date into different columns
    data['day'] = data['date'].dt.day
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month

    # drop date column
    data = data.drop(columns=['date'])

    # gives year, month and day dummy values
    one_hot_day = pd.get_dummies(data['day'], prefix='day')
    one_hot_year = pd.get_dummies(data['year'], prefix='year')
    one_hot_month = pd.get_dummies(data['month'], prefix='month')
    data = data.drop(columns=['day', 'year', 'month'])
    data = data.join([one_hot_day, one_hot_month, one_hot_year])

    return data


def clean_data(data: DataFrame):
    """

    :param data:
    :return: cleans data from invalid values and irrelevant values
    """

    # drop all bad values
    data = data[data['sqft_living'] > 0]
    data = data[data['floors'] > 0]
    data = data[data['sqft_above'] >= 0]
    data = data[data['sqft_basement'] >= 0]
    data = data[data['yr_built'] > 0]
    data = data[data['price'] > 0]

    # drop the id lat and long columns
    data = data.drop(columns=['id', 'long', 'lat'])

    return data


def process_zipcode(data: DataFrame):
    one_hot = pd.get_dummies(data['zipcode'], prefix="zipcode")
    data = data.join(one_hot)

    return data


def process_data(data: DataFrame):
    """
    process the given data according to the rules specified in the Answer.pdf file
    :param data: the unprocessed data
    :return: the data after proceeding
    """

    data = clean_data(data)
    data = process_date(data)
    data = process_zipcode(data)

    return data


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    data = process_date(data)

    return data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = np.std(y)
    i = 0
    for column in X:
        x_variable = np.array(X[column])
        y_variable = np.array(y)

        mat = np.array([x_variable, y_variable])
        cov = np.cov(mat, bias=True)[0][1]
        feature_std = np.std(X[column])
        pearson_correlation = cov / (feature_std * y_std)
        scatter_plot = px.scatter(x=X[column], y=y,
                                  title="feature name:{fn} Pearson Correlation :{pc}"
                                  .format(fn=column, pc=pearson_correlation))
        scatter_plot.update_layout(xaxis_title=column, yaxis_title='price')
        scatter_plot.write_image(output_path + "/" + column + ".png", format='png')


if __name__ == '__main__':
    np.random.seed(0)
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])

    print(mean_square_error(y_true, y_pred))
    """
    # Question 1 - Load and preprocessing of housing prices dataset
    data = pd.read_csv("/Users/eilon/private/university/3/B/IML/תרגילים/IML.HUJI/datasets/house_prices.csv")
    data = process_data(data)

    # Question 2 - Feature evaluation with respect to response
    y = data['price']
    X = data.drop(columns=['price'])
    path = "/Users/eilon/private/university/3/B/IML/תרגילים/IML.HUJI/ex2_plot_pc"
    feature_evaluation(X, y, path)

    # Question 3 - Split samples into training- and testing sets.
    train_data, train_response, test_data, test_response = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lr_model = LinearRegression()
    all_loss_results = np.zeros(91)
    all_loss_std = np.zeros(91)
    for p in range(10, 101):
        loss_results = np.zeros(10)
        for k in range(10):
            p_train_data = train_data.sample(frac=p / 100)
            p__train_response = train_response.loc[p_train_data.index]
            lr_model.fit(np.array(p_train_data), np.array(p__train_response))
            loss_results[k] = lr_model.loss(np.array(test_data), np.array(test_response))

        std_loss = loss_results.std()
        average_loss = loss_results.mean()
        print(average_loss)
        all_loss_std[p - 10] = std_loss
        all_loss_results[p - 10] = average_loss

    fig = go.Figure(
        [go.Scatter(x=np.array(range(10, 101)), y=all_loss_results, mode="markers+lines", name="avrage loss",
                    line=dict(dash="dash"), marker=dict(color="green", opacity=.7)), \
         go.Scatter(x=np.array(range(10, 101)), y=all_loss_results - 2 * all_loss_std, fill=None, mode="lines",
                    line=dict(color="lightgrey"), showlegend=False), \
         go.Scatter(x=np.array(range(10, 101)), y=all_loss_results + 2 * all_loss_std, fill='tonexty', mode="lines",
                    line=dict(color="lightgrey"), showlegend=False)])
    fig.update_layout(
        yaxis_title='loss average',
        title='average loss per percent of train data',
        hovermode="x"
    )

    fig.show()
    """
