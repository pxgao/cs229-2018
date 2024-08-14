import numpy as np
import util
import tensorflow as tf
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model = PoissonRegression(step_size=lr, eps=1e-5)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # [m, n]
        x = tf.convert_to_tensor(np.asarray(x, np.float32), np.float32)
        # [m]
        y = tf.convert_to_tensor(np.asarray(y, np.float32), np.float32)
        # [n]
        theta_ = tf.zeros(tf.shape(x)[1])
        # scalar
        m_ = tf.cast(tf.shape(x)[0], tf.float32)
        while True:
            # [m]
            h_x = tf.math.exp(tf.linalg.matvec(x,  theta_))

            # [m]
            v = (h_x - y)
            # [n]
            gradient = 1.0 / m_ * tf.linalg.matvec(tf.transpose(x), v)
            old_theta = theta_
            theta_ = theta_ - self.step_size * gradient
            #print("theta_:", theta_, "\n\n")
            if (tf.norm(old_theta - theta_) < self.eps):
                break
        self.theta = theta_
         # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        x = tf.convert_to_tensor(np.asarray(x, np.float32), np.float32)
        return tf.math.exp(tf.linalg.matvec(x, self.theta))
         # *** END CODE HERE ***
