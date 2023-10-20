import numpy as np
import util

from linear_model import LinearModel
import matplotlib.pyplot as plt
import time

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
    print("Start training and prediction - Poisson Regression")
    # create a PoissonRegressionClassificater instance
    clf = PoissonRegression(lr,max_iter=1000)
    # Fit a Poisson Regression model
    clf.fit(x_train,y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # load validation set
    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    start_time = time.time()
    y_pred = clf.predict(x_eval)
    prediction_time = time.time() - start_time
    print(f"Prediction Spent: {prediction_time:.6f} seconds")
    np.savetxt(pred_path,y_pred)

    print("--------------------------End training and prediction - Poisson Regression--------------------------")
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
        self.theta = np.zeros(x.shape[1]).reshape(-1,1)# theta.shape = (4,1)

        # Stochastic
        # start_time = time.time()
        # indices = np.arange(x.shape[0])  # create an array of indices
        # iter = 0
        # while(iter < self.max_iter):
        #     iter+=1
        #     np.random.shuffle(indices)  # shuffle the indices in place
        #     for i in indices: # 2500
        #         old_theta = self.theta
        #         gradient = (np.exp(np.dot(self.theta.reshape(1,-1),x[i].reshape(-1,1))) - y[i])
        #         self.theta = self.theta - (self.step_size*gradient*x[i].reshape(-1,1))
        #         if( np.linalg.norm(self.theta - old_theta, ord=1) < self.eps):
        #             break
        # fitting_time= time.time() - start_time
        # print("Iterated for :",iter,"times. Ended with theta : ",self.theta)
        # print(f"Training Spent(Stochastic gradient descent): {fitting_time:.6f} seconds")
        
        # Batch Gradient descent
        start_time = time.time()
        
        iter = 0
        while(True):
            iter+=1
            old_theta = self.theta
            gradient = 0
            for i in range(x.shape[0]):
                gradient += (np.exp(np.dot(self.theta.reshape(1,-1),x[i].reshape(-1,1))) - y[i])*x[i].reshape(-1,1)
            self.theta = self.theta - self.step_size*gradient/x.shape[0]
            if( np.linalg.norm(self.theta - old_theta, ord=1) < self.eps):
                break
            
        fitting_time= time.time() - start_time
        print("Iterated for :",iter,"times. Ended with theta : ",self.theta)
        print(f"Training Spent(Batch gradient descent): {fitting_time:.6f} seconds")
        # *** END CODE HERE ***
        


    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            pred[i] = np.exp(np.dot(self.theta.reshape(1,-1),x[i].reshape(-1,1)))
        return pred
        # *** END CODE HERE ***
