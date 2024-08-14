import numpy as np
import util
import tensorflow as tf

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    #clf.predict(x_eval)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit1(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Init theta
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's method
        while True:
            # Save old theta
            theta_old = np.copy(self.theta)
            
            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            print("h_x", h_x.shape, "x.T", x.T.shape, "x.T * h_x", (x.T * h_x).shape, "x.T * h_x * (1 - h_x)", (x.T * h_x * (1 - h_x)).shape)
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m

            # Updata theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)
            print(self.theta)
            # End training
            if np.linalg.norm(self.theta-theta_old, ord=1) < self.eps:
                break
        # *** END CODE HERE ***


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

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
            h_x = tf.math.sigmoid(tf.linalg.matvec(x,  theta_))

            # [m]
            v = (h_x - y)
            # [n]
            gradient = 1.0 / m_ * tf.linalg.matvec(tf.transpose(x), v)
            #print("gradient:", tf.squeeze(gradient))
       
            # [n, m]
            xt_h_x = tf.transpose(x) * h_x
            #print("xt_h_x:", xt_h_x)

            # [m]
            one_m_h_x = 1 - h_x

            # [n, m]
            exp1 = xt_h_x * one_m_h_x
            #print("exp1", exp1)

            # [n, n]
            hessian = 1.0 / m_ * tf.linalg.matmul(exp1, x)
            #print("hessian:", hessian)
            old_theta = theta_
            theta_ = theta_ - tf.linalg.matvec(tf.linalg.inv(hessian), gradient)
            #print("theta_:", theta_, "\n\n")
            if (tf.norm(old_theta - theta_) < self.eps):
                break
        self.theta = theta_
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = tf.convert_to_tensor(np.asarray(x, np.float32), np.float32)
        return 1.0 / (1 + tf.math.exp(-tf.linalg.matvec(x, self.theta)))
        # *** END CODE HERE ***
