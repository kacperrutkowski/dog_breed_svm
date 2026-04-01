from sklearn.decomposition import PCA

def do_PCA(X, n_components : int):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca