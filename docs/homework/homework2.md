# homework1

## 2.1

### 基于协方差矩阵的特征值分解

$$
\begin{aligned}
    x_{i} = \frac{1}{N}\sum_{j=1}^{N}x_{j}\cr
    X = [ x_{1},...,x_{N}](n \times N)\cr
    C = \frac{1}{n} XX^{T}(n\times n)\cr
    Cv_{i} = \lambda_{i} v_{i}\cr
    w = \arg\max (\lambda_{}) \,v\cr
\end{aligned}
$$

### 基于SVD分解协方差矩阵实现PCA算法

$$
\begin{aligned}
    C = U\Sigma V^{T}\cr
    w = \max(U[i])\cr
\end{aligned}
$$

w 即是第一主成分

- 特征值分解实现方法简单，适合小规模数据分析
- 一些SVD算法不用计算协方差矩阵，效率高

## 2.2

