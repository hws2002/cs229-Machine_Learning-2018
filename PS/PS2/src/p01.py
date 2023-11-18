import argparse

from p01_lr import main as p01
import util
import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('sp_num', nargs = '?', type = str, default = 'a',
                    help = 'Subproblem to run, _ for all subproblems.')

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

if args.sp_num == 'a':
    plot_a_b()

if args.sp_num == 'c':
    p01(args.learning_rate)
    
    