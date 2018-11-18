from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle
from sklearn.svm import SVR


def regression(X,Y):
    
    # regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    # regr = linear_model.BayesianRidge()

    svr = SVR(kernel='rbf', C=100, gamma=0.00001)
    # svr = SVR(kernel='linear', C=1)
    # svr = SVR(kernel='poly', C=1, degree=2)  #slow!

    num_test=1000
#     num_train=total_rows-num_test
    num_train=len(Y)-num_test

    X_train=X[:num_train, :]
    y_train=Y[:num_train]
    X_test=X[num_train:, :]
    y_test=Y[num_train:]
    index_test=[i for i in range(num_test)]
    

    svr.fit(X_train, y_train)

    y_pred=svr.predict(X_test)
    return index_test, y_test, y_pred
    



if __name__ == "__main__":
    start=time.time()
    x_file = open("x.p",'r')
    y_file = open("y.p",'r')
    X=pickle.load(x_file)
    Y=pickle.load(y_file)

    test_index, test_y, pred_y=regression(X,Y)  
    me = (np.abs(test_y - pred_y)).mean()
    print('me: ',me)
    plt.scatter(test_index, test_y,s=8)
    plt.scatter(test_index, pred_y,s=4)
    plt.savefig('SVR.png')
    plt.show()

    end=time.time()
    print('Time: ', start-end)