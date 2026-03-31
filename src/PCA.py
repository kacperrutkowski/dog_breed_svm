from preprocessing import preprocess_data
from sklearn.decomposition import PCA

X, y = preprocess_data(path_to_data = "data/Images", img_size = 64, n_class = 10, n_samples_in_class = 100)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

