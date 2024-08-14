# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) + 2.0 * theta * 1.0 / m

    return grad


def logistic_regression(X, Y, learning_rate):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    #learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0 or i %10000 == 1:
            hx = 1. / (1 + np.exp(-X.dot(theta)))
            log_loss = np.sum(Y * np.log(hx) + (1-Y) * np.log(1-hx))
            print(grad, theta, log_loss, "hx", np.max(hx), np.min(hx), "X", np.max(X), np.min(X), "X*theta", np.max(X.dot(theta)), np.min(X.dot(theta)), "theta_diff", np.linalg.norm(prev_theta - theta), prev_theta, theta)
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    #print('==== Training model on data set A ====')
    #Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    #logistic_regression(Xa, Ya, 10)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, 1)


if __name__ == '__main__':
    main()
