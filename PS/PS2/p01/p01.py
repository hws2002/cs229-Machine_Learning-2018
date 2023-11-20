import sys
import os

# Get the absolute path of the src directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add the src directory to the Python path
sys.path.append(src_dir)

# Now you can import util and p01_lr
from p01_lr import main as p01
import util


# from p01_lr import main as p01
# import util
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sp_num', nargs = '?', type = str, default = 'a',
                    help = 'Subproblem to run, _ for all subproblems.')

parser.add_argument('c_num', nargs='?', type = int, default = 1,
                    help = 'subsubproblem for c(1~5), default value is 1')

parser.add_argument('learning_rate',nargs='?', type = int, default = 10, 
                    help = 'Learning rate to run for (c) i')
args = parser.parse_args()

def plot_a_b():
    #load data
    x, y = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    x, y = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    plt.figure(figsize=(10,6))
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=1)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    
    
def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y, learning_rate = 10, savepath = 'output/?.png'):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        learning_rate *= 1/(i**2)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            # print("Theta ended with :", theta)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            util.plot2(X,Y,theta,savepath)
            break
    return


def p01c2(learning_rate = 10):
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya,learning_rate, savepath = '../p01/output/p01c_iiA.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, learning_rate, savepath ='../p01/output/p01c_iiB.png')

def p01c3(learning_rate = 10):
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya,learning_rate, savepath = '../p01/output/p01c_iiA.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, learning_rate, savepath ='../p01/output/p01c_iiB.png')




if args.sp_num == 'a':
    plot_a_b()

if args.sp_num == 'c':
    if args.c_num == 1:
        p01(args.learning_rate)
    if args.c_num == 2:
        p01c2()
    if args.c_num == 4:
        p01c3()
        
    