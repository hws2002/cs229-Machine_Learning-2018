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
Now y-labels affect training example(training session).  

## Model
At this point, I found that my code isn't robust enough.  
As I run the code, I got these errors  
```bash
~/src/p01b_logreg.py:78: RuntimeWarning: overflow encountered in exp
  gradient[i] += (y[j] - 1 / (1 + np.exp(-np.dot(self.theta, x[j])))) * x[j][i]
~/src/p01b_logreg.py:85: RuntimeWarning: overflow encountered in exp
  hesse[i][j] += (1 / (1 + np.exp(-np.dot(self.theta, x[k])))) * (1 - 1 / (1 + np.exp(-np.dot(self.theta,x[k])))) * x[k][i] * x[k][j]
```
Original code
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
So changed corresponding to the following  

```bash
            # step1 : calculate gradient
            for i in range(self.theta.shape[0]): # 3
                for j in range(x.shape[0]): # 800
                    gradient[i] += (-1/x.shape[0]) * (y[j] - 1 / (1 + np.exp(-np.dot(self.theta, x[j])))) * x[j][i]
                # gradient[i] = -1/x.shape[0] * gradient[i]


            for i in range(hesse.shape[0]):
                for j in range(hesse.shape[1]):
                    for k in range(x.shape[0]):
                        hesse[i][j] += (1/x.shape[0]) * (1 / (1 + np.exp(-np.dot(self.theta, x[k])))) * (1 - 1 / (1 + np.exp(-np.dot(self.theta,x[k])))) * x[k][i] * x[k][j]
                    # hesse[i][j] = 1/x.shape[0] * hesse[i][j]
```
Although this code precisely illustrates the update of each element both in gradient and hesse matrix(at least to me ^^), for the sake of code robustness and stability, it would be better to use following style of code.
```bash
# Init theta
        m, n = x.shape
        self.theta = np.zeros(n)
        iter = 0
        # Newton's method
        while True:
            # Save old theta
            theta_old = np.copy(self.theta)
            iter+=1
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
unless the number itself has overflow, this won't incur unnecessary overflow during the update.  

```python
self.theta = np.zeros(x.shape[1]) # start from 0 vector
        newtheta = np.zeros(x.shape[1]) # (3,)
        gradient = np.zeros(x.shape[1]) #(3,)
        hesse = np.zeros((x.shape[1], x.shape[1])) #(3,3)
        
        # # loop until convergence
        iter = 0
        while (iter < self.max_iter):
            iter += 1
            print(iter)
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