# 问题 (c)
Write a logistic regression classifier that uses
$x_1$ and $x_2$ as input features, and train it using the t-labels.


## Shape
Here, I'm going to add intercept for convention.
```bash
train_x.shape :  (1250, 3)
train_t.shape :  (1250,)
```

Just let you know, I got same result regardless of intercept.  

## Model
In this problem, we are going to inherit logitic model of problem1b, which uses Newton's method to fit.

# 问题 (d)
Now we only train on y-labels. and test on true labels


## Validation set
Upon train on y-labels, by chance, I found that accuracy for training set and validation set was both 0.5.  

