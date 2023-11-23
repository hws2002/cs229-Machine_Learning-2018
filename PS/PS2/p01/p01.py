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

subparsers = parser.add_subparsers(dest = 'sp_num', help = 'Subproblem commands')

# a parser for subproblem 'b'
parcer_b = subparsers.add_parser('b', help='Run subproblem b')
parcer_b.add_argument(dest = 'b_subtask',nargs='?',
                        type = str, 
                        default = "theta",
                        help='subtask for b')

# a partser for subproblem 'c'
parser_c = subparsers.add_parser('c', help='Run subproblem c')

parser_c.add_argument(dest ='c_num', nargs='?', type = int, default = 1,
                    help = 'subsubproblem for c(1~5), default value is 1')

parser_c.add_argument('value', type=float, help='Value for learning_rate or std_dev')

# parser_c.add_argument(dest = 'learning_rate',nargs='?', type = float, default = 10, 
                    # help = 'Learning rate to run for (c) i')

# parser_c.add_argument(dest = 'std_dev', nargs='?', type = float, default = 0.1,
                    # help = 'standard deviation of the gaussian distribution (noise)')

args = parser.parse_args()


def plot_a_b(dataset = 'a'):
    #load data
    if dataset == 'A':
        x, y = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    else :
        x, y = util.load_csv('../data/ds1_b.csv', add_intercept=True)
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

def logistic_regression_b(X, Y):
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
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print("Theta ended with :", theta)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def logistic_regression(X, Y, learning_rate = 10, decrase_lr = False, savepath = 'output/?.png', regularizer = False):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = ( (1-2*learning_rate/m)*theta - learning_rate * grad ) if regularizer else (theta - learning_rate * grad)
        learning_rate = learning_rate * 1/(i**2) if decrase_lr  else learning_rate
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            if(savepath != 'output/?.png'):
                util.plot2(X,Y,theta,savepath)
            break
    return


def p01b():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression_b(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression_b(Xb, Yb)

def p01c1(learning_rate = 10):
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya,learning_rate)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, learning_rate)

def p01c2():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, decrase_lr = True, savepath = '../p01/output/p01c_iiA.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, decrase_lr = True, savepath ='../p01/output/p01c_iiB.png')

def p01c3():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, savepath = '../p01/output/p01c_iiiA.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, savepath ='../p01/output/p01c_iiiB.png')

def p01c4():
    # add regularizer
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, savepath = '../p01/output/p01c_ivA.png', regularizer = True)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, savepath ='../p01/output/p01c_ivB.png', regularizer = True)

def add_gaussian_noise(data, std_dev):
    """
    Adds Gaussian noise to the data.

    Args:
        data : Original data (can be training data or labels)
        std_dev : Standard deviation of the Gaussian noise.
        return : Data with added Guassian noise.
    """
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise
def p01c5(std_dev = 0.1):
    # add zero-mena gaussian noise to the training data
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xa = add_gaussian_noise(Xa,std_dev)
    logistic_regression(Xa, Ya, savepath = '../p01/output/p01c_vA_trainnoise.png')
    
    print('\n==== Training model on data set B ====\n')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    Xb = add_gaussian_noise(Xb,std_dev)
    logistic_regression(Xb, Yb, savepath ='../p01/output/p01c_vB_trainnoise.png')

    # add zero-mena gaussian noise to the label
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Ya = add_gaussian_noise(Ya,std_dev)
    logistic_regression(Xa, Ya, savepath = '../p01/output/p01c_vA_labelnoise.png')
    
    print('\n==== Training model on data set B ====\n')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    Yb = add_gaussian_noise(Yb,std_dev)
    logistic_regression(Xb, Yb, savepath ='../p01/output/p01c_vB_labelnoise.png')
    
    # add zero-mena gaussian noise both
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xa = add_gaussian_noise(Xa,std_dev)
    Ya = add_gaussian_noise(Ya,std_dev)
    logistic_regression(Xa, Ya, savepath = '../p01/output/p01c_vA_bothnoise.png')
    
    print('\n==== Training model on data set B ====\n')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    Xb = add_gaussian_noise(Xb,std_dev)
    Yb = add_gaussian_noise(Yb,std_dev)
    logistic_regression(Xb, Yb, savepath ='../p01/output/p01c_vB_bothnoise.png')

if args.sp_num == 'a':
    p01()
    
if args.sp_num == 'b':
    if args.b_subtask == 'theta':
        p01b()
        
    if args.b_subtask =='plot':
        plot_a_b('A')
        plot_a_b('B')

if args.sp_num == 'c':
    if args.c_num == 1:
        p01c1(args.value)
    if args.c_num == 2:
        p01c2()
    if args.c_num == 3:
        p01c3()
    if args.c_num == 4:
        p01c4()
    if args.c_num == 5:
        p01c5(args.value)