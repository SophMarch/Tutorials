# Linear Discriminant Analysis

## Shortcut

## Logic
Linear Discriminant Analysis (LDA) is a classification algorithm adapted to multiclass problems based on statistical properties of the data. To apply it, we need to assume that each variable has a gaussian distribution and that each input has a same variance. The intuition behind the LDA is to estimate the probability that a new input variable set belong to each class of the problem. Then, the class with the highest probability is the prediction. Also, from the probability field, we have: 

$$P(A\cap B) = P(A)P(B|A)= P(B)P(A|B)$$

Indeed, $P(A\cap B)$ is the probability that $A$ occurs times the probability that $B$ happens while $A$ and inversely. We then retrieve the Bayes Theorem:

$$P(A|B)= \frac{P(A)P(B|A)}{P(B)}$$

With $A$ the probability that the output variable $Y$ belong to the class $y$ and $B$ the probability that the input variable $X$ equals $x$, we have:

$$P(Y=y|X=x)= \frac{P(Y=y)P(X=x|Y=y)}{P(X=x  )}$$

## Python Tutorial 
