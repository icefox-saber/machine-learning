import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV


# 定义 BIC 和 AIC 评分函数
def gmm_bic_score(estimator, X):
    return -estimator.bic(X)


def gmm_aic_score(estimator, X):
    return -estimator.aic(X)


# 设置参数网格
param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}


# 实验次数
n_experiments = 10


# 簇数量
n_components = 3


# 样本数量
n_samples = 600


# 维度范围
dimensions_list = range(2, 6)


# 初始化准确度列表
bic_accuracy = np.zeros(len(dimensions_list))
aic_accuracy = np.zeros(len(dimensions_list))
vbem_accuracy = np.zeros(len(dimensions_list))


# 进行实验
for i, dimension in enumerate(dimensions_list):
    for _ in range(n_experiments):
        # 数据生成
        np.random.seed(_)
        C = np.random.rand(dimension, dimension)
        X = np.concatenate([np.dot(np.random.randn(n_samples, dimension), C) + np.array([2 * j] * dimension) for j in range(n_components)])


        # 使用 GridSearchCV 找到 BIC 和 AIC 最优模型
        bic_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
        aic_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_aic_score)
        bic_search.fit(X)
        aic_search.fit(X)


        # 获取 BIC 和 AIC 的最佳簇数
        bic_best_n_components = bic_search.best_params_["n_components"]
        aic_best_n_components = aic_search.best_params_["n_components"]


        # 使用 Variational Bayesian EM (VBEM) 自动确定簇的数量
        vbem_model = BayesianGaussianMixture(n_components=10, covariance_type='full', random_state=0)
        vbem_model.fit(X)
        vbem_n_components = np.sum(vbem_model.weights_ > 1e-2)  # 只考虑权重较大的簇


        # 统计准确度
        bic_accuracy[i] += np.abs(bic_best_n_components - n_components) < 1
        aic_accuracy[i] += np.abs(aic_best_n_components - n_components) < 1
        vbem_accuracy[i] += np.abs(vbem_n_components - n_components) < 1


# 计算平均准确度
bic_accuracy /= n_experiments
aic_accuracy /= n_experiments
vbem_accuracy /= n_experiments


# 绘制对比图
methods = ['BIC', 'AIC', 'VBEM']
accuracy_list = [bic_accuracy, aic_accuracy, vbem_accuracy]


plt.figure(figsize=(10, 6))
for i, (method, accuracy) in enumerate(zip(methods, accuracy_list)):
    plt.plot(dimensions_list, accuracy, label=method, marker='o')


plt.xlabel("dimensions")
plt.ylabel("accuracy")
plt.title("accuracy-dimensions")
plt.legend()
plt.show()