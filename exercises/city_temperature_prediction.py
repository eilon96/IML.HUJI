from pandas import DataFrame

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from datetime import datetime


def process_date(data: DataFrame):

    day_of_the_year = data["Date"].apply(lambda x: x.strftime("%j"))
    data["DayOfTheYear"] = day_of_the_year
    data['DayOfTheYear'] = data["DayOfTheYear"].astype(int)
    return data


def clean_data(data: DataFrame):
    # drop all bad values
    data = data[data['Year'] > 0][data["Year"] <= 2022]
    data = data[data['Day'] > 0][data["Day"] <= 31]
    data = data[data['Month'] > 0][data["Month"] <= 12]
    data = data[data['Temp'] > -70][data["Temp"] <= 57]


    return data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)

    data = clean_data(data)
    data = process_date(data)

    return data



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    file = "/Users/eilon/private/university/3/B/IML/תרגילים/IML.HUJI/datasets/City_Temperature.csv"
    X = load_data(file)



    # Question 2 - Exploring data for specific country
    X_Israel = X[X["Country"] == "Israel"]
    X_Israel["Year"] = X_Israel["Year"].astype(str)
    fig_1 = px.scatter(X_Israel, "DayOfTheYear", "Temp", color="Year", )
    fig_1.show()

    X_Israel_by_month_std = X_Israel.groupby(["Month"])["Temp"].agg('std')
    print(X_Israel_by_month_std)

    fig_2 = px.bar(X_Israel_by_month_std).update_layout(title="Month temp std",
                                                        xaxis_title="Month", yaxis_title="temp std")
    fig_2.show()

    # Question 3 - Exploring differences between countries
    X_grouped_by_month_and_country = X.groupby(['Month', 'Country']).agg({"Temp": ["mean", "std"]})
    X_grouped_by_month_and_country.columns = ['temp_mean', "temp_std"]
    X_grouped_by_month_and_country = X_grouped_by_month_and_country.reset_index()
    print(X_grouped_by_month_and_country)
    fig_3 = px.bar(X_grouped_by_month_and_country, x="Month", y="temp_mean",
                   error_y="temp_std",color="Country", barmode="group")
    fig_3.show()



    # Question 4 - Fitting model for different values of `k`
    train_data, train_response, test_data, test_response = \
        split_train_test(X_Israel["DayOfTheYear"], X_Israel["Temp"], 0.75)

    recorded_loss_per_deg = np.zeros(10)
    best_deg = 0
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(np.array(train_data), np.array(train_response))
        recorded_loss_per_deg[k - 1] = round(poly_model.loss(np.array(test_data), np.array(test_response)), 2)
        print ("loss value with degree {deg} : {loss}".format(loss=recorded_loss_per_deg[k - 1], deg=k))

    best_deg = recorded_loss_per_deg.argmin() + 1
    recorded_loss_per_deg = DataFrame(recorded_loss_per_deg, list(range(1, 11)))
    fig_4 = px.bar(recorded_loss_per_deg).\
        update_layout(title="loss for degree", xaxis_title="degree", yaxis_title="loss")
    fig_4.show()


    print(best_deg)
    # Question 5 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(best_deg)
    best_model.fit(np.array(X_Israel["DayOfTheYear"]), np.array(X_Israel["Temp"]))
    X = X[X["Country"] != "Israel"]
    recorded_loss_per_country = []
    for country in X["Country"].unique():
        if country != "Israel":
            X_country = X[X["Country"] == country]
            loss = best_model.loss(np.array(X_country["DayOfTheYear"]), np.array(X_country["Temp"]))
            recorded_loss_per_country.append(loss)
            print(country, loss)

    recorded_loss_per_country = DataFrame({"Country" :X["Country"].unique(), "loss": recorded_loss_per_country})
    fig_5 = px.bar(recorded_loss_per_country, x="Country", y="loss")
    fig_5.show()

