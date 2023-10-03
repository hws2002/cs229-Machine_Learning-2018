# 问题 (b)

## Requirement
implement locally weighted linear regression using the normal equations derived in Part (a) and using
$w^{(i)} = exp(−{‖x^{(i)} −x‖_2^2 \over 2τ^2})$.  
Train model on the train split using τ = 0.5, then run model on the valid split
and report the mean squared error (MSE). Finally plot your model’s predictions on the validation set (plot the training set with blue ‘x’ markers and the validation set with a red ‘o’ markers).

## Train Session

### shapes (dimensions)
```bash
x_train.shape: (300, 2)
y_train.shape: (300,)
```
!note that $x_0 = 1$. demonstrated with code below:  
```python
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
```

```bash
x_eval.shape: (200, 2)
y_eval.shape: (200,)
```

### Steps to train
Actually there is nothing have to be done at training, as our weights are affected by query points.  
## Prediction Session
### squared Eucilidean(L2) distance
Given two vectors $x^{(i)}$ and $x$, the squared L2 distance between them can be computed as:  
$\lVert x^{(i)} - x\rVert_2^2 = \Sigma_{j=1}^n(x_j^{(i)} - x_j)^2$  
thus, we can implement it like following:  
```python
    def L2_distance(x_i,x):
        return np.sum((x_i - x) ** 2)
```

### Mean Squared Error (MSE)
definition of MSE is as follows:
> The Mean Squared Error (MSE) is a measure of the average of the squares of the errors or deviations, i.e., the difference between the estimator and what is estimated. It is commonly used to evaluate the accuracy of a regression model in predicting values.
>
To compute the MSE for a set of predicted values relative to the true values, we can use the following formula:  
$MSE = {1 \over n} \Sigma_{i=1}^n(y_i - \hat{y_i})^2$  
thus, we can implement it like following:
```python
    # Get MSE value on the validation set
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
```
### weights
given in the question.  
$w^{(i)} = exp(-{\lVert x^{(i) - x}\rVert^2_2\over 2\tau^2})$
### Normal equation
$\theta = (X^TWX)^{-1}X^TWy$  
refer to problem 5(a).  

# 问题 (b)

## Requirement
tune the hyperparameter τ.  
find the MSE value of model on the validation set for each of the values of τ specified in the code. For each τ, plot model’s predictions on the validation set in the format described in part (b). Report the value of τ which achieves the lowest MSE on the valid split, and finally report the MSE on the test split using this τ-value.

