import numpy as np
import util

from linear_model import LinearModel

import time

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
    print("Start training and prediction - Gaussian Discriminant Analysis (GDA)")
    # check shape of x_train and y_train
    clf = GDA()
    
    # STEP1 : fitting session
    start_time = time.time()
    clf.fit(x_train, y_train)
    fitting_time= time.time() - start_time
    print(f"Training Time: {fitting_time:.6f} seconds")
    
    # Load evaluation dataset
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    #! y_eval is not used in this problem
    
    # STEP2 : prediction session
    start_time = time.time()
    pred_result = clf.predict(x_eval)
    prediction_time = time.time() - start_time
    print(f"Prediction Time: {prediction_time:.6f} seconds")
    
    # STEP3 : save outputs
    np.savetxt(pred_path, pred_result, fmt='%d')
    
    # STEP4 : plot the decision boundary
    
    
    print("End training and prediction - Gaussian Discriminant Analysis (GDA)")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # STEP1 : caculate phi, mu_0, mu_1, and sigma
            # phi
        phi = np.sum(y == 1) / len(y)
            # mu_0, mu_1
        mu_0 = np.mean(x[y == 0], axis=0)
        mu_1 = np.mean(x[y == 1], axis=0)
            # sigma
        sigma = np.zeros((x.shape[1], x.shape[1]))
        for i in range(len(y)):
            # gurantee that x[i]- mu_0 is of (2,1) so that the result of np.dot is a matrix
            if y[i] == 0:
                sigma += np.dot((x[i] - mu_0).reshape(-1,1), (x[i] - mu_0).reshape(1,-1))
            else:
                sigma += np.dot((x[i] - mu_1).reshape(-1,1), (x[i] - mu_1).reshape(1,-1))
        sigma /= len(y)
        
        # STEP2 : calculate theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta = np.dot(sigma_inv, (mu_0 - mu_1).reshape(-1,1))
        self.theta_0 = 1/2 * (np.dot(np.dot((mu_1 + mu_0).reshape(1,-1),sigma_inv),(mu_0 - mu_1).reshape(-1,1))) - np.log((1-phi)/phi)
        
        # (OPTIONAL) print the value of parameters
        print("phi = ", phi)
        print("mu_0 = ", mu_0)
        print("mu_1 = ", mu_1)
        print("sigma = ", sigma)
        print("theta = ", self.theta)
        print("theta_0 = ", self.theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # theta is of shape (2,1)
        # calculate p(y=1|x)
        output = np.zeros(x.shape[0])
        for i in range(x.shape[0]): # x_i is a vector of (2,)
            output[i] = 1 if 1 / (1 + np.exp(-(np.dot(self.theta.reshape(1,-1), x[i].reshape(-1,1)) + self.theta_0))) > 0.5 else 0
        return output
        # *** END CODE HERE
