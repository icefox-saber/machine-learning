import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Step 1: 数据生成过程
def generate_data(N=100, m=3, n=10, sigma_e=0.1, mu=0):
    """
    Generate data for Factor Analysis.
    N: number of samples
    m: latent dimension
    n: observed dimension
    sigma_e: standard deviation of noise
    mu: mean of Gaussian noise
    """
    # 生成A: 随机生成观测矩阵
    A = np.random.randn(n, m)
    
    # 生成潜在变量y: 维度为(m, N)
    y = np.random.randn(m, N)
    
    # 生成噪声e: 维度为(n, N)
    e = np.random.normal(mu, sigma_e, (n, N))
    
    # 生成观测数据x: x = Ay + e
    X = A @ y + e
    return X.T  # 返回N个样本 (N, n)

# Step 2: 使用EM算法的Factor Analysis模型
def run_fa(X, max_components):
    """
    Run Factor Analysis for different latent dimensions.
    X: input data, shape (N, n)
    max_components: maximum latent dimension to test
    """
    aic_list = []
    bic_list = []
    N, n = X.shape

    for m in range(1, max_components + 1):
        # Fit Factor Analysis model
        fa = FactorAnalysis(n_components=m, max_iter=500)
        fa.fit(X)
        
        # Log-likelihood
        log_likelihood = np.sum(fa.score_samples(X))
        
        # 计算AIC和BIC
        d_m = m * n + m + n  # 参数数量: loadings + latent means + noise variance
        aic = log_likelihood - d_m
        bic = log_likelihood - 0.5 * np.log(N) * d_m
        
        aic_list.append(aic)
        bic_list.append(bic)

    return aic_list, bic_list

# Step 3: 重复实验，观察AIC和BIC的表现
def experiment():
    # 参数设置
    N = 50  # 样本数量
    n = 20  # 观测变量的维度
    max_m = 18  # 测试的最大潜在维度
    
    m_true_list = [15]  # 真实的潜在维度列表
    
    plt.figure(figsize=(12, 8))

    for i, m_true in enumerate(m_true_list, 1):
        # 生成数据
        X = generate_data(N=N, m=m_true, n=n)
        
        # 运行Factor Analysis并计算AIC/BIC
        aic, bic = run_fa(X, max_m)
        
        # 绘制AIC和BIC结果
        plt.subplot(2, 2, i)
        plt.plot(range(1, max_m + 1), aic, label='AIC', marker='o')
        plt.plot(range(1, max_m + 1), bic, label='BIC', marker='s')
        plt.axvline(m_true, color='r', linestyle='--', label=f'True m={m_true}')
        plt.xlabel('Number of latent factors (m)')
        plt.ylabel('Score')
        plt.title(f'AIC and BIC for True m={m_true}')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment()
