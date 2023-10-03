import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    print("Start comparing MSE for different tau - PROBLEM5C")
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    lowest_mse = 1
    best_tau = -1
    for tau in tau_values:
        lwlr = LocallyWeightedLinearRegression(tau)
    # Fit a LWR model with the best tau value
        lwlr.fit(x_train,y_train)
    # Run on the test set to get the MSE value
        y_pred = lwlr.predict(x_eval)
        mse = mean_squared_error(y_eval,y_pred)
        if lowest_mse > mse :
            lowest_mse = mse
            best_tau = tau
        print(f"MSE value on the validation set with tau={tau}: ", mse)
    # Save predictions to pred_path
    # Plot data
        util.plot_lwr(x_train,y_train,x_eval,y_pred,f'output/p05c_tau_{tau}.png')
    print(f"lowest MSE : {lowest_mse}, achieved by tau={best_tau}")
    print(f"Staring prediction for the test dataset")
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    lwlr = LocallyWeightedLinearRegression(best_tau)
    lwlr.fit(x_train,y_train)
    y_pred = lwlr.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    print(f"MSE value on the test set with tau={best_tau}: ",mse)
    util.plot_lwr(x_train,y_train,x_test,y_pred,f'output/p05c_test_tau_{best_tau}.png')
    # *** END CODE HERE ***
