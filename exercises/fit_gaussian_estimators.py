from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    sample_size = 1000
    mu = 10
    sigma = 1
    # Question 1 - Draw samples and print fitted model
    my_uni_dis = UnivariateGaussian()
    uni_random_sample = np.random.normal(mu, sigma, size=sample_size)
    my_uni_dis.fit(uni_random_sample)
    print(my_uni_dis.mu_, my_uni_dis.var_)


    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean_distance = []
    for n in range(10, 1000, 10):
        estimated_mean_distance.append(abs(np.mean(uni_random_sample[:n]) - mu))

    go.Figure([go.Scatter(x=list(range(10, sample_size, 10)), y=estimated_mean_distance, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Estimation of divariation from  Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$", yaxis_title="r$\hat\mu - E[X]$", height=300)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = my_uni_dis.pdf(uni_random_sample)
    X = np.sort(uni_random_sample)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{pdfs of Samples}$",
                  xaxis_title="$\\text{samples values}$", yaxis_title="$\\text{pdfs values}$", height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    sample_size = 1000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    random_sample = np.random.multivariate_normal(mu, sigma, size=sample_size)
    my_multi_dis = MultivariateGaussian()
    my_multi_dis.fit(random_sample)

    print(my_multi_dis.mu_, "\n")
    print(my_multi_dis.cov_, "\n")

    # Question 5 - Likelihood evaluation
    f_1_space = sorted(np.linspace(-10, 10, 200))
    f_3_space = sorted(np.linspace(-10, 10, 200))

    log_likelihood_matrix = np.ones((len(f_1_space), len(f_3_space)))
    max_log_likelihood_value = 0
    max_index_f_1 = 0
    max_index_f_3 = 0

    for i in range(len(f_1_space)):
        for j in range(len(f_3_space)):
            mu = np.array([f_1_space[i], 0, f_3_space[j], 0])
            log_likelihood_value = my_multi_dis.log_likelihood(mu, sigma, random_sample)
            log_likelihood_matrix[i][j] = log_likelihood_value
            if log_likelihood_value > max_log_likelihood_value or (i == 0 and j == 0):
                max_log_likelihood_value = log_likelihood_value
                max_index_f_1 = i
                max_index_f_3 = j

    go.Figure(data=go.Contour(z=log_likelihood_matrix, x=f_3_space, y=f_1_space),
              layout=go.Layout(
                  title="log likelihood as function of the third and second entry in the expected value matrix",
                  xaxis_title="third entry of expected value matrix",
                  yaxis_title="first entry of expected value matrix")).show()

    # Question 6 - Maximum likelihood
    print(f_1_space[max_index_f_1], f_3_space[max_index_f_3])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
