from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


def regression(X,Y,X_):
    
    # regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    regr = linear_model.BayesianRidge()
    # regr=LogisticRegression()  #slow!

    regr = SVR(kernel='rbf', C=10, gamma=0.00001)
    # regr = SVR(kernel='linear', C=1)
    # regr = SVR(kernel='poly', C=1, degree=2)  #slow!

    regr.fit(X, Y)
    index_test=[i for i in range(len(X_))]

    # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
    #                  normalize=False)
    y_pred=regr.predict(X_)
    return index_test, y_test, y_pred
    

    
    

# print(regr.coef_)
# np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)

if __name__ == "__main__":
    start=time.time()
    # x_file = open("x.p",'r')
    # y_file = open("y.p",'r')
    # X=pickle.load(x_file)
    # Y=pickle.load(y_file)
    x_file1=pd.read_csv("input_train_0.7.csv",header = None)
    y_file=pd.read_csv("output.csv",header = None)
    x_train=x_file1.values[1:]
    Y=y_file.values
    x_file2=pd.read_csv("input_test_0.7.csv",header = None)
    x_test=x_file2.values[1:]
    y_train=Y[:len(x_train)]
    y_test=Y[len(x_train):]

    print(np.shape(x_train), np.shape(y_train))
    print(np.shape(x_test),np.shape(y_test))

    test_index, test_y, pred_y=regression(x_train,y_train,x_test) 
    end=time.time()
    me = (np.abs(test_y - pred_y)).mean()
    print('me: ',me)

    '''plot'''
    plot_len=500
    plt.scatter(test_index[:plot_len], test_y[:plot_len],s=8, label='label')
    plt.scatter(test_index[:plot_len], pred_y[:plot_len],s=4,label='prediction',alpha=0.8)
    plt.xlabel('Samples')
    plt.ylabel('Value')
    leg = plt.legend(loc=1)

    plt.savefig('./images/LR.png')
    plt.show()

    
    print('Time: ', end-start)