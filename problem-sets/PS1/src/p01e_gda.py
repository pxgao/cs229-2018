import numpy as np
import util
import tensorflow as tf

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    #util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    print(y_pred)
    #np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit_sol(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        # Init theta
        m, n = x.shape
        self.theta = np.zeros(n+1)

        # Compute phi, mu_0, mu_1, sigma
        y_1 = sum(y == 1)
        phi = y_1 / m
        mu_0 = np.sum(x[y == 0], axis=0) / (m - y_1)
        mu_1 = np.sum(x[y == 1], axis=0) / y_1
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m

        print(sigma)
        # Compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (mu_0 + mu_1).dot(sigma_inv).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)
        
        print(self.theta)
        # Return theta
        return self.theta

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # m, n
        x = tf.convert_to_tensor(np.asarray(x, np.float32), np.float32)
        # m
        y = tf.convert_to_tensor(np.asarray(y, np.float32), np.float32)

        sz = tf.cast(tf.size(y), tf.float32)
        ct1 = tf.cast(tf.size(tf.where(y)), tf.float32)
        ct0 = sz - ct1
        phi = ct1 / sz
    
        # n
        mu0 = tf.reduce_sum( tf.transpose(x) * (1-y), 1) / ct0
        mu1 = tf.reduce_sum( tf.transpose(x) * y, 1) / ct1
        
        # n
        mu_y0 = (1-y)[:,tf.newaxis] * mu0[tf.newaxis, :]
        mu_y1 = y[:,tf.newaxis] * mu1[tf.newaxis, :]
        x_mu_y = x - mu_y0 - mu_y1
        # n, n
        sigma = tf.linalg.matmul(tf.transpose(x_mu_y), x_mu_y) / sz
        
        # n, n
        sigma_inv = tf.linalg.inv(sigma)
        # n
        theta = tf.linalg.matvec(sigma_inv, (mu1 - mu0))
        theta0 = 0.5 * tf.linalg.matmul((mu0+mu1)[tf.newaxis,:], sigma_inv) 
        theta0 = tf.linalg.matvec(theta0, (mu0 - mu1))
        theta0 = theta0 - tf.math.log((1-phi)/phi)
        self.theta = theta = tf.concat([theta0, theta], 0)
        # tf.print(self.theta)
        self.theta = tf.cast(self.theta, tf.float64)
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1.0 / (1.0 + tf.math.exp(-(tf.linalg.matvec(x, self.theta))))
        # *** END CODE HERE
