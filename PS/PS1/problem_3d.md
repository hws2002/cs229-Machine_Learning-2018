# 问题 (d)

## Requirement
dataset is about the past traffic of a website.  
we will assume that the data follows a Poisson distribution, and implement Poisson regression for this dataset and use gradient ascent to maximize the log-likelihood of $\theta$.

### shape (dimension)
```bash
x_train.shape: (2500, 4)
```
!note that $x_0$ is not 1, which is demonstrated bellow:  
```python
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
```
### Gradient
each element of gradient vector is calculated as following:  
${\partial \over \partial \theta_j}l(\theta) = (y^{(i)}-exp(\theta^Tx^{(i)}))x_j^{(i)}$  
where $l(\theta)$ is a log-likelihood of a single example set $(x^{(i)},y^{(i)})$ parametrized by $\theta$.

thus, we can update each elements like follows:  
$\theta_j := \theta_j + \alpha(y^{(i)} - exp(\theta^Tx^{(i)}))x_j^{(i)}$

## Questions
About the 