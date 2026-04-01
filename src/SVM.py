from preprocessing import preprocess_data
from PCA import do_PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



X, y = preprocess_data(path_to_data = "data/Images", img_size = 64, n_class = 2, n_samples_in_class = 100)

X_pca = do_PCA(X, n_components= 200)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size = 0.2, random_state=42
)
print(y_train.shape)

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

print("Accuracy: ", model.score(X_test, y_test))
