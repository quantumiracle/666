from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle


def regression(X,Y):
    
    # regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    regr = linear_model.BayesianRidge()

    num_test=1000
#     num_train=total_rows-num_test
    num_train=len(Y)-num_test

    X_train=X[:num_train, :]
    y_train=Y[:num_train]
    X_test=X[num_train:, :]
    y_test=Y[num_train:]
    print(np.shape(y_test))
    index_test=[i for i in range(num_test)]
    

    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    return index_test, y_test, y_pred
    

    

if __name__ == "__main__":
    start=time.time()
    # x_file = open("x.p",'r')
    # y_file = open("y.p",'r')
    # X=pickle.load(x_file)
    # Y=pickle.load(y_file)
    x_file=pd.read_csv("input.csv",header = None)
    y_file=pd.read_csv("output.csv",header = None)
    X=x_file.values.T
    Y=y_file.values
    print(np.shape(X), np.shape(Y))

    test_index, test_y, pred_y=regression(X,Y)  
    mse = (np.abs(test_y - pred_y)).mean()
    print('me: ',me)
    plt.scatter(test_index, test_y,s=8)
    plt.scatter(test_index, pred_y,s=4)
    plt.savefig('LR.png')
    plt.show()

    end=time.time()
    print('Time: ', start-end)