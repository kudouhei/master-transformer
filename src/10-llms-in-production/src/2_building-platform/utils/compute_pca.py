import numpy as np

from scipy import linalg

# compute the principal components of the data
def compute_pca(data, n_components=2):
    """
    Input:
        data: numpy array of shape (n_samples, n_features)
        n_components: number of principal components to compute
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    # m, n = data.shape

    # mean of each feature
    # 每列特征减去均值，使数据均值为零（PCA的前提条件）: 数据平移到原点附近，方便计算
    data -= data.mean(axis=0)

    # compute the covariance matrix
    # 计算协方差矩阵
    R = np.cov(data, rowvar=False)

    # compute eigenvectors and eigenvalues of the covariance matrix
    # 计算协方差矩阵的特征向量和特征值
    evals, evecs = linalg.eigh(R)

    # sort eigenvalues in descending order
    # 按降序排序特征值
    idx = np.argsort(evals)[::-1]
    
    # choose the first n eigenvectors, n is the number of components
    # 选择前n个特征向量，n是主成分的数量
    evecs = evecs[:, idx]

    # sort eigenvalues in descending order
    # 按降序排序特征值
    evals = evals[idx]

    # select the first n eigenvectors, n is the number of components
    # 选择前n个特征向量，n是主成分的数量
    evecs = evecs[:, :n_components]

    # transform the data using the eigenvectors
    # 使用特征向量变换数据
    X_reduced = np.dot(evecs.T, data.T).T

    # 返回降维后的数据和原始数据
    return X_reduced, evecs, evals
    