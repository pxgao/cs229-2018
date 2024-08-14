import matplotlib.pyplot as plt
import numpy as np
import util
import tensorflow as tf

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    mse = np.mean((y_pred - y_eval)**2)
    print(f'MSE={mse}')

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = tf.cast(x, tf.float32)
        self.y = tf.cast(y, tf.float32)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = tf.cast(x, tf.float32)
        [m, n] = tf.shape(x)
        result = []

        [tm, tn] = tf.shape(self.x)

        for i in range(m):
            diff = (self.x - tf.ones([tm,1]) * x[i])
            w = tf.exp(-tf.reduce_sum(diff * diff, 1) / (2 * self.tau * self.tau))
            #tf.print(w)
            xwx = tf.linalg.matmul(tf.transpose(self.x) * w, self.x)
            #tf.print(xwx)
            theta = tf.linalg.matvec(tf.linalg.matmul(tf.linalg.inv(xwx), tf.transpose(self.x)) *  w,  self.y)
            #tf.print(theta)
            y_p = tf.tensordot(theta, x[i], 1)
            #print(y_p)
            result = tf.concat([result, [y_p]], 0)
        return result
        # *** END CODE HERE ***
