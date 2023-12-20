import time
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import decomposition
from csv import writer


models = [
    ("Nearest Centroid", NearestCentroid()),
    ("kNN (k=3)", KNeighborsClassifier(n_neighbors=3)),
    ("kNN (k=7)", KNeighborsClassifier(n_neighbors=7)),
    ("Naive Bayes (Gaussian)", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("RF (5)", RandomForestClassifier(n_estimators=5, n_jobs=8)),
    ("RF (50)", RandomForestClassifier(n_estimators=50, n_jobs=8)),
    ("RF (500)", RandomForestClassifier(n_estimators=500, n_jobs=8)),
    ("RF (1000)", RandomForestClassifier(n_estimators=1000, n_jobs=8)),
    ("linearSVM (C=0.01)", LinearSVC(C=0.01)),
    ("linearSVM (C=0.1)", LinearSVC(C=0.1)),
    ("linearSVM (C=1.0)", LinearSVC(C=1.0)),
    ("linearSVM (C=10.0)", LinearSVC(C=10.0)),
]
csv_writer = None

def run(x_train, y_train, x_test, y_test, clf):
    s = time.time()
    clf.fit(x_train, y_train)
    e_train = time.time() - s
    s = time.time()
    score = clf.score(x_test, y_test)
    e_test = time.time() - s
    print("score = %0.4f (time, train=%8.3f, test=%8.3f)" % (score, e_train, e_test))
    return score, e_train, e_test


def train(x_train, y_train, x_test, y_test, experiment):    
    for model_name, model in models:
        print(f"{model_name + ':':^20}", end="")
        model_results = run(x_train, y_train, x_test, y_test, model)
        row = [model_name, experiment, *model_results]
        csv_writer.writerow(row)


def main():
    global csv_writer
    x_train = np.load("data/mnist/mnist_train_vectors.npy").astype("float64")
    y_train = np.load("data/mnist/mnist_train_labels.npy")
    x_test = np.load("data/mnist/mnist_test_vectors.npy").astype("float64")
    y_test = np.load("data/mnist/mnist_test_labels.npy")

    with open("data/mnist/mnist_experiments.csv", "w", newline="") as csvfile:
        csv_header = ["model", "run", "score", "runtime train", "runtime test"]
        csv_writer = writer(csvfile, delimiter=",")
        csv_writer.writerow(csv_header)

        print("Models trained on raw [0,255] images:")
        train(x_train, y_train, x_test, y_test, "raw [0, 255]")

        print("Models trained on raw [0,1) images:")
        train(x_train / 256.0, y_train, x_test / 256.0, y_test, "raw [0, 1)")

        m = x_train.mean(axis=0)
        s = x_train.std(axis=0) + 1e-8
        x_ntrain = (x_train - m) / s
        x_ntest = (x_test - m) / s

        print("Models trained on normalized images:")
        train(x_ntrain, y_train, x_ntest, y_test, "normalized")

        pca = decomposition.PCA(n_components=15)
        pca.fit(x_ntrain)
        x_ptrain = pca.transform(x_ntrain)
        x_ptest = pca.transform(x_ntest)

        print("Models trained on first 15 PCA components of normalized images:")
        train(x_ptrain, y_train, x_ptest, y_test, "PCA (15)")


main()
