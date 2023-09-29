# 问题
calculate $\phi , \mu_0 , \mu_1, \Sigma$, and use these to derive $\theta$ (gradient descent).
And use $\theta$ to predict $y$ for $x_{valid}$

## shapes (dimensions)
```bash
x_train.shape (800, 2)
y_train.shape (800,)
```
!note that $x_0$ is not 1, which is different from problem_1b.   demonstrated with code below:
```python
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
```


## Parameters
refer to problem_1d.
## $\theta$ & Prediction
refer to problem_1c.  
$p(y=1|x;\phi ; \mu_0 ; \mu_1 ; \Sigma) = {1\over 1+\exp(-(\theta^Tx + \theta_0))}$  
where  $\theta = \Sigma^{-1}(\mu_1 - \mu_0)$  
and $\theta_0 = {1\over 2}(\mu_0 + \mu_1)^T\Sigma^{-1}(\mu_0 - \mu_1) + \log{\phi \over 1 - \phi}$

