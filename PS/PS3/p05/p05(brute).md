# (a)

## Source Image
`peppers-small.tiff` is of shape (128, 128, 3)  
To get the (r,g,b) at pixel position (i,j) : A[i,j,:] or A[i,j,]  


## Centroids $\mu_i$
$\mu_1, \mu_2, ..., \mu_{16} \in \R^3$  
where each elements representing red, green, blue intensity.  
They are updated by the following fomula  
$\mu_j = {****\sum_{i=1}^n 1{c^{(i)} = j}x^{(j)} \over \sum_{i=1}^n1{c^{(i)} = j}}$

## $C^{(i)}$

$C^{(i)} = \argmin_j||x^{(i)} - \mu_j||_2^2$

## Break condition
I ran the code for 50 and 100 iterations, and due to the fact that k-means sometimes end up with local minima, repeated for 5 times each.  
It roughly took 1 and a half minutes to finish compression when threshold was 50 iterations, and 4 minutes for 100.  

# (b) Compression Factor 
In the original image, we needed 24bits to represent a pixel.  
But now, as the number of colors that pixels can take is only 16, we only need 4bits.  
In this perspective, we can say that the image are compressed by factor of 6.


# Appendix
I tried 500 iterations, and it was not too different from 50 or 100 iterations.  
