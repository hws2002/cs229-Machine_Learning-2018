# 问题
To train a logistic regression classifier with Newton's method.
starting with $\theta = 0$

## Linear Model
have following parameters:
```python
    self.theta = theta_0
    self.step_size = step_size
    self.max_iter = max_iter
    self.eps = eps
    self.verbose = verbose
```
## shapes (dimensions)
```bash
x_train.shape:  (800, 3)
y_train.shape:  (800,)
```
!note that $x_0 = 1$. demonstrated with code below:
```python
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
```
## Newton's method
update $\theta$ with following formula:  
$\theta := \theta - H^{-1}\nabla_{\theta}l(\theta)$
where $H$ is Hessian matrix, and $\nabla_{\theta}l(\theta)$ is gradient vector.
### Gradient
each element of gradient vector is calculated as following:  
${\partial\over\partial \theta_i}l(\theta) = -{1 \over m}\sum_j^m (y^{(j)} - g(\theta^Tx^{(j)}))x_i^{(j)}$

where $g(z) = {1\over 1 + e^{-z}}$
and $m = x.shape[0] = 800$

### Hessian
each element of Hessian matrix is calculated as following:  
${\partial \over \partial \theta_i \partial\theta_j}l(\theta) = {1 \over m}\sum_k^m g(\theta^Tx^{(k)})(1 - g(\theta^Tx^{(k)}))x_i^{(k)}x_j^{(k)}$

where $g(z) = {1\over 1 + e^{-z}}$
and $m = x.shape[0] = 800$
