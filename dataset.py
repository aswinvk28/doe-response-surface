from mlxtend.data import loadlocal_mnist

def get_XY(images_path, labels_path):
    X, y = loadlocal_mnist(
    images_path=images_path, 
    labels_path=labels_path)

    return X, y

def get_mnist_digits(X, y, num, idx):
    return X[y==num][idx].reshape(28,28)

def get_emnist_digits(X, y, idx):
    return X[y==idx].reshape(28,28)

