# adaptive learning

## feature

Points come one by one. The clusters update each time a point arrive. So it can cluster dynamic datas.

## CL (Competitive learning)

### process of CL

- **INI** : Initialize $k$ clusters
- **ITERATION** : Each time a point ($x_{t}$) arrive, it choose the nearest group $m_{j}$ and update it.

$$
\begin{aligned}
    c = arg \text{ } min_{j}\varepsilon_{t}(\theta)\cr
    \varepsilon_{t}(\theta) = D(t,j)\cr
    p_{j,t} =
    \begin{cases}
        1 & j=c\cr
        0 & j\neq c\cr
    \end{cases}\cr
    m_{j}^{new} = m_{j}^{old} + \eta p_{j,t}(x_{t}-m_{j}^{old})\cr
\end{aligned}
$$

## FSCL (Frequency sensitive competitive learning)

## RPCL (Rival penalized competitive learning)
