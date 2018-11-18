from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


def regression(X,Y):
    '''model selection'''
    # regr = linear_model.LinearRegression()
    # regr = linear_model.Lasso()
    # regr = linear_model.Ridge(alpha=.1)
    # regr = linear_model.BayesianRidge()
    # regr=LogisticRegression()  #slow!

    # regr = SVR(kernel='rbf', C=100, gamma=0.00001)
    regr = SVR(kernel='linear', C=1)
    # regr = SVR(kernel='poly', C=1, degree=2)  #slow!
    
#     raw_data='Road-Accident.csv'
#     df = pd.read_csv('Road-Accident.csv',encoding='utf-8')
    num_test=3000
#     num_train=total_rows-num_test
    num_train=len(Y)-num_test

    X_train=X[:num_train, :]
    y_train=Y[:num_train]

    '''add noise, increase dataset'''
    # X_train0=X[:num_train, :]
    # y_train0=Y[:num_train]
    # for i in range(10):
    #     X_train=np.concatenate((X_train,X_train0+10*np.random.rand(np.shape(X_train0)[0],np.shape(X_train0)[1])))
    #     y_train=np.concatenate((y_train,y_train0))
    
    
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

    plt.savefig('./3_fea_images/LR.png')
    plt.show()

    
    print('Time: ', end-start)