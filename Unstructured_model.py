from preprocessing import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


def calculate_accuracy(predictions, real):
    return np.average(predictions == real)

def logistic_regression(data):
    test_set_x, test_set_y, train_set_x, train_set_y = data
    lr = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=30000)
    lr.fit(train_set_x, train_set_y)
    pred = lr.predict(test_set_x)
    acc = calculate_accuracy(pred, test_set_y)
    print("Logistic Regression accuracy:",round(acc, 3))

def naive_bayes(data):
    test_set_x, test_set_y, train_set_x, train_set_y = data
    nb = GaussianNB()
    nb.fit(train_set_x, train_set_y)
    pred = nb.predict(test_set_x)
    acc = calculate_accuracy(pred, test_set_y)
    print("Naive Bayes accuracy:",round(acc, 3))

def one_vs_rest(model, data, name='model'):
    testSet_x, testSet_y, trainSet_x, trainSet_y = data
    ovr = OneVsRestClassifier(model)
    ovr.fit(trainSet_x, trainSet_y)
    pred = ovr.predict(testSet_x)
    acc = calculate_accuracy(pred, testSet_y)
    print("One vs rest",name,' accuracy:',round(acc, 3))

if __name__ == '__main__':
    data = prepare_train_test()
    test_x, test_y, train_x, train_y = data
    val_x, testSet_x, val_y, testSet_y = train_test_split(test_x, test_y, test_size=2/3)
    data = testSet_x, testSet_y, train_x, train_y
    naive_bayes(data)
    logistic_regression(data)
    one_vs_rest(Perceptron(max_iter=600000), data, 'perceptron')
    one_vs_rest(LogisticRegression(max_iter=600000), data, 'logistic regression')
    one_vs_rest(SVC(max_iter=600000, kernel="linear"), data, 'linear svm')
    one_vs_rest(SVC(max_iter=600000, kernel="poly"), data, 'poly svm')

