# clustering

## k-mean clustering

### goal of k-mean clustering

divide a set of N element $\lbrace X_{1},X_{2}...X_{N}\rbrace$ in $K$ groups $u_{1},u_{2},...,u_{k}$ that satisfies:

$$
\begin{aligned}
    Minimize \quad J = \sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk}D(n,k)\cr
    r_{nk} =
    \begin{cases}
        1 & X_{n} \in u_{k}\cr
        0 & X_{n} \notin u_{k}\cr
    \end{cases}\cr
    D(n,k) \text{ denote the distance between }X_{n} \text{ and } u_{k}\cr
\end{aligned}
$$

### process of k-mean clustering

### running time of k-mean clustering

## hierarchical clustering

### process of hierarchical clustering

- **INI**: Assign each data point into its own group(It's one of instances)
- **ITERATION**: look for the two closest groups and merge them into one group
- **TERMINATION**: Stop when all the data points are merged into a single cluster

### Distance Measure of hierarchical clustering

- Distance between data points a and b: $d(a,b)$
- Group A and B:
  - single

### running time of hierarchical clustering
