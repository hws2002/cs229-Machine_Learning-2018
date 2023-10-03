import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

import time

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
    print("Start training and prediction - Locally weighted linear regression PROBLEM5B")
    # # check dimensions
    # print("x_train.shape:",x_train.shape)
    # print("x_train[1].shape:",x_train[1].shape)
    # print("y_train.shape:",y_train.shape)
    
    # Fit a LWR model
    start_time = time.time()
    lwlr = LocallyWeightedLinearRegression(tau)
    lwlr.fit(x_train, y_train)
    
    # load validation data
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = lwlr.predict(x_eval)
    prediction_time = time.time() - start_time
    print(f"Prediction Time: {prediction_time:.6f} seconds")
    
    # Get MSE value on the validation set
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    mse = mean_squared_error(y_eval,y_pred)
    print("MSE value on the validation set : ", mse)
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    util.plot_lwr(x_train,y_train,x_eval,y_pred,'output/p05b.png')
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
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n). # (200,2)

        Returns:
            Outputs of shape (m,).
        """
        def L2_distance(x_i,x):
            # x.shape: (2,)
            return np.sum((x_i - x) ** 2)
        # *** START CODE HERE ***
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]): # there are 200 query points
            w = [0] * self.x.shape[0] # should be of length 300
            for j in range(self.x.shape[0]): # there are 300 training examples
                w[j] = np.exp(-1*L2_distance(self.x[j],x[i])/(2*(self.tau **2)))
            W = np.diag(w) # 300 x 300
            # print("W : ",W)
            X = self.x.reshape(self.x.shape[0],self.x.shape[1])
            theta = np.linalg.inv(np.transpose(X)@W@X)@np.transpose(X)@W@(self.y.reshape(-1,1))
            y_pred[i] = theta.reshape(1,-1)@(x[i].reshape(-1,1))
        return y_pred
        # *** END CODE HERE ***
