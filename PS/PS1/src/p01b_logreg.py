import numpy as np
import util
from linear_model import LinearModel

import time

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    print("Start training and prediction - Logistic Regression with Newton's Method")
    # STEP1 : training session
    start_time = time.time()
    # train a logistic regression classifier
    clf = LogisticRegression(eps=1e-5)
    clf.fit(x_train, y_train)
    # Stop recording time for training and output the elapsed time
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.6f} seconds")
    
    # load dataset from eval_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    #! y_eval is not used in this problem
    
    # STEP2 : prediction session
    # Restart recording time for prediction
    start_time = time.time()
    pred_result = clf.predict(x_eval)
    prediction_time = time.time() - start_time
    print(f"Prediction Time: {prediction_time:.6f} seconds")
    
    # STEP3 : save outputs
    np.savetxt(pred_path, pred_result, fmt='%d')
    
    # STEP4 : plot the decision boundary
    util.plot(x_train, y_train, clf.theta, 'output/p01b_logistic_regression_train_{}.png'.format(pred_path[-5]))
    print("--------------------------End training and prediction - Logistic Regression with Newton's Method--------------------------")
    
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
        
        # loop until convergence
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
            if(1 / (1 + np.exp(-np.dot(self.theta, x[i]))) >= 0.5):
                output[i] = 1
            else :
                output[i] = 0
        return output
        # *** END CODE HERE ***
