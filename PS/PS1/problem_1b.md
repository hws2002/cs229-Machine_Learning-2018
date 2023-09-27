# 问题
To train a logistic regression classifier with Newton's method.
starting with $\theta = 0$


## shapes (dimensions)
```bash
x_train.shape:  (800, 3)
y_train.shape:  (800,)
```
!note that $x_0 = 1$
## Newton's method
### Gradient
${\partial\over\partial \theta_i}l(\theta) = -{1 \over m}\sum_j^m (y^{(j)} - g(\theta^Tx^{(j)}))x_i^{(j)}$

where $g(z) = {1\over 1 + e^{-z}}$
and $m = x.shape[0] = 800$

### Hessian
${\partial \over \partial \theta_i \partial\theta_j}l(\theta) = {1 \over m}\sum_k^m g(\theta^Tx^{(k)})(1 - g(\theta^Tx^{(k)}))x_i^{(k)}x_j^{(k)}$

where $g(z) = {1\over 1 + e^{-z}}$
and $m = x.shape[0] = 800$
