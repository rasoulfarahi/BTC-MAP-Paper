from sklearn.decomposition import PCA

def apply_pca(X):
    # n_components=0.95 ensures that 95% of the variance is preserved
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    return X_reduced
