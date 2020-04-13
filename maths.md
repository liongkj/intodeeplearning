### normalize an array

- normalize a vector
  ∥v∥=√a^2+b^2
- i think is just make vector smaller

### tf.math.exp

exponential:

- tf.math.exp([2])
- e ^ 2 == 7.xxx

### np.random.choice

numpy.random.choice(a, size=None, replace=True, p=None)
Generates a random sample from a given 1-D array

- a : 1-D array-like or int
  If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)

- size : int or tuple of ints, optional
  Output shape. If the given shape is, e.g., (m, n, k), then m _ n _ k samples are drawn. Default is None, in which case a single value is returned.

- replace : boolean, optional
  Whether the sample is with or without replacement

- p : 1-D array-like, optional
  The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

### log

usually just used for making graphs smoother, easier to analyze

### logsigma (don't know yet)

## reduce

### reduce_sum

    - just break down and sum all of n
    - if axis = 0, sum each row
    - if axis = 1, sum each column

### reduce_mean

    - just break down and calculate mean all of n
    - if axis = 0, mean for each row
    - if axis = 1, mean for each column

# Symbols

∑ = sum of

standard deviation:

1. how spread out numbers are

2.

variance a^2:

1. sum of mean / n
2.

# probability

        q (z | x)

- can be state as probability of z event occuring in x condition

# Linear regression

- aka Least square sum

- normally to find line of best fit
