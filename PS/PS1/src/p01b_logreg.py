import numpy as np
import util

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
    
    # train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    # load dataset from eval_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    # to save outputs from validation set to pred_path
    np.savetxt(pred_path, clf.predict(x_eval))
    # *** END CODE HERE ***

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1]) # start from 0 vector
        newtheta = np.zeros(x.shape[1]) # (3,)
        gradient = np.zeros(x.shape[1]) #(3,)
        hesse = np.zeros((x.shape[1], x.shape[1])) #(3,3)
        # step0 : loop until convergence
        iter = 0
        while (iter < self.max_iter):
            iter += 1
            # step1 : calculate gradient
            for i in range(self.theta.shape[0]): # 3
                for j in range(x.shape[0]): # 800
                    gradient[i] += (y[j] - 1 / (1 + np.exp(-np.dot(self.theta, x[j])))) * x[j][i]
                gradient[i] = -1/x.shape[0] * gradient[i]
            # step2 : calculate hesse matrix
            for i in range(hesse.shape[0]):
                for j in range(hesse.shape[1]):
                    for k in range(x.shape[0]):
                        hesse[i][j] += (1 / (1 + np.exp(-np.dot(self.theta, x[k])))) * (1 - 1 / (1 + np.exp(-np.dot(self.theta,x[k])))) * x[k][i] * x[k][j]
                    hesse[i][j] = 1/x.shape[0] * hesse[i][j]
            # step3 : update theta
            if (np.linalg.det(hesse) == 0):
                raise ValueError('Hesse matrix is singular')
            self.theta = newtheta
            newtheta = newtheta - np.dot(np.linalg.inv(hesse), gradient)
            if( np.linalg.norm(newtheta - self.theta) < self.eps):
                self.theta = newtheta
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        output = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            output[i] = 1 / (1 + np.exp(-np.dot(self.theta, x[i])))
        return output
        # *** END CODE HERE ***
