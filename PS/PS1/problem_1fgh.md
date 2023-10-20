# 问题 (f)

## Requirement
plot a training data with $x_1$ on the horizontal axis, and $x_2$ on the vertical axis.  
Use different colors to indicate examples from different classes. (0 and 1)  
Also plot the decision boundary of logistic regression and GDA.

## Logistic Regression : 分析
we have already implemented logistic regresssion plot function in `util.py`.
```python
def plot(x, y, theta, save_path=None, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.
    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    if save_path is not None:
        plt.savefig(save_path)
```
among which, `x` is the training data, `y` is the label, `theta` is the parameter of logistic regression model.  
## GDA
from problem_1c we know that GDA results in a classifer that has a linear decision boundary with the following form:  
$p(y=1 | x ; \phi ; \mu_0 ; \mu_1 ; \Sigma) = {1\over 1 + \exp(-(\theta^Tx + \theta_0))}$  
thus, the decision boundary would be the line where  
$p(y=1 | x ; \phi ; \mu_0 ; \mu_1 ; \Sigma) = 0.5$  
equivalent to points who satisfy  
$\theta^Tx + \theta_0 = 0$  
thus, the modification would be like the follows:  
```python
    # Plot decision boundary (found by solving for theta^T x + theta_0 = 0)
    x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2]) + margin1, 0.1)
    x2 = (-1/theta[-1] * (theta_0 + theta[-2] * x1)).reshape(-1)
    plt.plot(x1, x2, c='red', linewidth=2)
```
Here we have to notice that we are already assuming that $\theta$ and $x$ is a 2-dimension vector, and $\theta_0$ is a scalar.  
for other cases, $\theta_0$ would be reamin the same, and $\theta$ would be a vector with the same dimension as $x$, but different dimension.  

## Result
refer to `src/output/p01b*.png` and  `src/output/p01e*.png` for the result.

# 问题 (g)

## Requirement
Analyse the difference between the two classifiers.  
On which dataset does GDA seem to perform worse than logistic regression? Why might this be the case?

## Analysis
First of all, it's obvious from the slope of the boundary, that the first dataset got more worse performance than the second dataset.  
We can easily conjecture the reason from the lecture, that GDA is a special case of logistic regression, which assumes that the $p(x|y)$ follows Gaussian distribution. (and covariance matrix of $x$ is the same for both classes)  
thus, the reason why the first dataset got worse performance is that the conditional probability of $x$ given $y$ is not Gaussian distributed.  

!recall that when the conditional probability of $x$ given $y$ is Gaussian distributed(assumptions are correct), the GDA will find better fits to the data than logistic regression.

# 问题 (h)
 For the dataset where GDA performed worse in parts (f) and (g),
can you find a transformation of the x(i)’s such that GDA performs significantly better?
What is this transformation?
