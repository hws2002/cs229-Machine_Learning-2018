# 问题 (b)
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

### The role of 1/m in Gradient Calculation.
In the context of logistic regression, when using Newton's method (also known as the Newton-Raphson method) for optimization, the factor \( \frac{1}{m} \) in the gradient calculation is typically included for averaging the gradients across all \( m \) training examples. This is a common practice in machine learning to ensure that the gradient is a measure of the average change in the cost function per data point, rather than the total sum of changes across all data points.

### The Role of \( \frac{1}{m} \) in Gradient Calculation:

1. **Normalization**: The \( \frac{1}{m} \) term normalizes the gradient by the number of training examples. Without this normalization, the magnitude of the gradient would be proportional to the number of examples, making the gradient and the subsequent updates to the model parameters (weights) dependent on the size of the dataset.

2. **Stability and Learning Rate**: Normalizing the gradient leads to more stable updates. It ensures that the step size taken in parameter space does not become excessively large when the dataset is large, which could potentially lead to overshooting the minimum of the cost function. It also means you don't have to adjust the learning rate as much when the size of the dataset changes.

3. **Interpretability**: When normalized, the gradient reflects the average change in the cost function per data point. This makes it easier to interpret the magnitude of the gradient and to set a learning rate that is independent of the number of training examples.

4. **Consistency Across Different Dataset Sizes**: Normalization ensures that the training process is consistent regardless of the number of examples in the dataset. If the training set size changes, the model's learning behavior remains relatively consistent, provided other factors remain constant.

### In Summary:

In logistic regression, the goal is typically to minimize the average cost (loss) across all training examples. The inclusion of \( \frac{1}{m} \) in the gradient calculation for Newton's method aligns with this goal by averaging the contributions of each training example to the gradient, leading to more stable and consistent training behavior.
### Hessian
each element of Hessian matrix is calculated as following:  
${\partial \over \partial \theta_i \partial\theta_j}l(\theta) = {1 \over m}\sum_k^m g(\theta^Tx^{(k)})(1 - g(\theta^Tx^{(k)}))x_i^{(k)}x_j^{(k)}$

where $g(z) = {1\over 1 + e^{-z}}$
and $m = x.shape[0] = 800$
