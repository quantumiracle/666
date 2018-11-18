from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression


def regression(X,Y):
    
    regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    # regr = linear_model.BayesianRidge()
    # regr=LogisticRegression()  #slow!
    
#     raw_data='Road-Accident.csv'
#     df = pd.read_csv('Road-Accident.csv',encoding='utf-8')
    num_test=1000
#     num_train=total_rows-num_test
    num_train=len(Y)-num_test

    X_train=X[:num_train, :]
    y_train=Y[:num_train]

    '''add noise, increase dataset'''
    X_train0=X[:num_train, :]
    y_train0=Y[:num_train]
    for i in range(10):
        X_train=np.concatenate((X_train,X_train0+10*np.random.rand(np.shape(X_train0)[0],np.shape(X_train0)[1])))
        y_train=np.concatenate((y_train,y_train0))
    X_test=X[num_train:, :]
    y_test=Y[num_train:]

    # X_test=X[:1000, :]
    # y_test=Y[:1000]
    index_test=[i for i in range(num_test)]
    
    regr.fit(X_train, y_train)

    y_pred=regr.predict(X_test)
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
    plt.scatter(test_index, test_y,s=8, label='label')
    plt.scatter(test_index, pred_y,s=4,label='prediction')
    leg = plt.legend(loc=1)

    plt.savefig('LR.png')
    plt.show()

    end=time.time()
    print('Time: ', start-end)