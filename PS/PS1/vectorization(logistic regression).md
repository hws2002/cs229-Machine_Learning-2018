# Different implementations of logistic regression and its dis/advantages (python) (vectorization)
This document explains about the things that I have learned from solving problem1b and problem2d.  

## Circumstance
The task was to implement logistic regression with Newton's methond in python.  
And to make myself understand the code even after some amount of time, I tried to write code in a way that it updates each elements of gradient and hesse matrix. But it was rather worse choice.  
This mistake was due to my lack of fundamentals in python, computer system.  
Let's look into four different implementations of logistic regression and compare those.  

## Codes
First one (worst)
```python
self.theta = np.zeros(x.shape[1]) # start from 0 vector
        newtheta = np.zeros(x.shape[1]) # (3,)
        gradient = np.zeros(x.shape[1]) #(3,)
        hesse = np.zeros((x.shape[1], x.shape[1])) #(3,3)
        
        # # loop until convergence
        iter = 0
        while (iter < self.max_iter):
            iter += 1
            # step1 : calculate gradient
            for i in range(self.theta.shape[0]): # 3
                for j in range(x.shape[0]): # 800
                    gradient[i] += (y[j] - 1 / (1 + np.exp(-np.dot(self.theta, x[j])))) * x[j][i]
                gradient[i] = -1/x.shape[0] * gradient[i]
                
            # step2 : calculate hesse matrix
            for i in range(hesse.shape[0]):
                for j in range(hesse.shape[1]):
                    for k in range(x.shape[0]):
                        hesse[i][j] += (1 / (1 + np.exp(-np.dot(self.theta, x[k])))) * (1 - 1 / (1 + np.exp(-np.dot(self.theta,x[k])))) * x[k][i] * x[k][j]
                    hesse[i][j] = 1/x.shape[0] * hesse[i][j]
            
            # step3 : update theta
            if (np.linalg.det(hesse) == 0):
                raise ValueError('Hesse matrix is singular')
            
            self.theta = newtheta
            newtheta = newtheta - np.dot(np.linalg.inv(hesse), gradient)
            if( np.linalg.norm(newtheta - self.theta) < self.eps):
                self.theta = newtheta
                break

```

Second one (better than the first one, but still needs improvement)
```python
        self.theta = np.zeros(x.shape[1]) # start from 0 vector
        newtheta = np.zeros(x.shape[1]) # (3,)
        gradient = np.zeros(x.shape[1]) #(3,)
        hesse = np.zeros((x.shape[1], x.shape[1])) #(3,3)
        
        # # loop until convergence
        iter = 0
        while (iter < self.max_iter):
            iter += 1
            # step1 : calculate gradient
            for i in range(self.theta.shape[0]): # 3
                for j in range(x.shape[0]): # 800
                    gradient[i] += (-1/x.shape[0]) * (y[j] - 1 / (1 + np.exp(-np.dot(self.theta, x[j])))) * x[j][i]
                # gradient[i] = -1/x.shape[0] * gradient[i]
                
            # step2 : calculate hesse matrix
            for i in range(hesse.shape[0]):
                for j in range(hesse.shape[1]):
                    for k in range(x.shape[0]):
                        hesse[i][j] += (1/x.shape[0]) * (1 / (1 + np.exp(-np.dot(self.theta, x[k])))) * (1 - 1 / (1 + np.exp(-np.dot(self.theta,x[k])))) * x[k][i] * x[k][j]
                    # hesse[i][j] = 1/x.shape[0] * hesse[i][j]
            
            # step3 : update theta
            if (np.linalg.det(hesse) == 0):
                raise ValueError('Hesse matrix is singular')
            
            self.theta = newtheta
            newtheta = newtheta - np.dot(np.linalg.inv(hesse), gradient)
            if( np.linalg.norm(newtheta - self.theta) < self.eps):
                self.theta = newtheta
                break
```

### Vectorization

```python
        m,d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d)
            
        while True:
            theta = np.copy(self.theta) # deep copy 
            h_x = 1/(1+np.exp(-1 * x @ self.theta ))
            gradient_J_theta = (-1/m) * np.transpose(x) @ (y - h_x)
            H = (1/m)*np.transpose(x)@ np.diag( (h_x*(1-h_x)).flatten() ) @ x
            self.theta = self.theta -  np.linalg.inv(H)@gradient_J_theta
            
            # terminate condition
            if  np.linalg.norm(self.theta-theta,ord=1) < self.eps:
                break
        

```

Forth one (best)
```python
# Init theta
        m, n = x.shape
        self.theta = np.zeros(n)
        # Newton's method
        while True:
            # Save old theta
            theta_old = np.copy(self.theta)
            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m

            # Updata theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)

            # End training
            if np.linalg.norm(self.theta-theta_old, ord=1) < self.eps:
                break
```

### Advantages and Disadvantages
the following is the answer from Chat-GPT4 under I asked to compare the second and the forth code snippet.


> The two code snippets you provided are indeed implementations of a logistic regression model using Newton's method, but there are some subtle differences that could lead to different results.

1. **Gradient and Hessian Calculation**:
    - In the first snippet, the gradient and Hessian matrix (Hesse matrix) are calculated using nested loops. This can introduce numerical inaccuracies, especially if there's an error in the indexing or the summation process. It also tends to be less efficient compared to vectorized operations.
    - The second snippet uses vectorized operations for calculating the gradient and the Hessian. This is generally more efficient and less prone to errors due to indexing. The use of vectorized operations can also lead to different numerical results due to the way floating-point arithmetic is handled in batches versus individual operations.

2. **Theta Update Mechanism**:
    - In the first code, there's a potential issue with how `theta` is updated. It assigns `self.theta` to `newtheta`, then updates `newtheta`, but `self.theta` is not updated with the new values of `newtheta` until the next iteration. This could lead to a lag in the update mechanism.
    - In the second snippet, `self.theta` is directly updated after the gradient and Hessian are calculated, which is a more straightforward approach.

3. **Stopping Criteria**:
    - Both snippets use a norm-based stopping criterion, but the norm used is different. The first uses the default norm (`np.linalg.norm`, which is L2 by default), and the second explicitly uses the L1 norm (`np.linalg.norm(..., ord=1)`). The choice of norm can affect when the loop stops, potentially leading to different results if the algorithm is near convergence.

4. **Numerical Stability**:
    - The way Hessian inversion and matrix multiplication are performed can also introduce numerical differences. The first snippet might be more prone to numerical instability due to the explicit loop-based calculation of the Hessian matrix.

5. **Initialization and Intermediate Variable Handling**:
    - The handling of intermediate variables like `gradient` and `Hesse` matrix in the first snippet might introduce additional scope for error or variance.

These factors combined can lead to different results even though the overarching methodology (Newton's method for logistic regression) is the same. In practice, the second approach is generally preferred due to its efficiency and reduced risk of numerical errors.


## Results
