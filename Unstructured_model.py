from preprocessing import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
#import graphviz
#from graphviz import Source
import pydot
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = base_agg()
    test_val_set_x, test_val_set_y, train_set_x, train_set_y = data  # for validation
    val_x, test_set_x, val_y, test_set_y = train_test_split(test_val_set_x,test_val_set_y,test_size=2/3, random_state=57)
    data = test_set_x, test_set_y, train_set_x, train_set_y