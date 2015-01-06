'''
Created on Nov 15, 2014

@author: nandini1986
'''

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax
from numpy import transpose
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron


if __name__ == '__main__':

# Part 1 - Open Dataset    
    f1 = open("chessboard.csv","rb")
    data = f1.read()
    content = data.split('\n')
    boo = []
    dataList = []
    for c in content:
        boo.append(c.split('\r'))
    boo2 = boo[0]
    for element in boo2:    
        data = element.split(',')
        if data[0] == 'A':
            continue
        dataList.append(data)
        
#Part 2 - Plot scatter plot
    A = []
    B = []
    label = []
    A0 = []
    A1 = []
    B0 = []
    B1 = []
    AB = []

     
    for i in xrange(len(dataList)):
        A.append(float(dataList[i][0]))
        B.append(float(dataList[i][1]))
        label.append(dataList[i][2])
    
    for i in xrange(len(label)):
        AB.append([float(A[i]),float(B[i])])
        if label[i] == '1':
            A1.append(float(A[i]))
            B1.append(float(B[i]))
        elif label[i] == '0':
            A0.append(float(A[i]))
            B0.append(float(B[i]))

    plot0 = plt.scatter(A0, B0, marker='o', color = 'red')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.title("Scatter Plot")
    plt.xlabel('A')
    plt.ylabel('B')
    plt.show()
        
    
# Part 3a - Stratified Sampling + Different SVMs
    X_train_fold = []
    y_train_fold = []
    X_test_fold = []
    y_test_fold = []
    n = 5
    Z_test = []
    X_train, X_test, y_train, y_test = train_test_split(AB, label, test_size=0.4, random_state=42)
    skf = StratifiedKFold(y_train, n_folds=n)
 
# Linear SVM
    print "\nLinear SVM"
    print "------------"
    CC = [1, 10, 50, 100, 10**4]
    for c in CC:
        score = 0
        for train, test in skf:
            for i in xrange(len(train)):
                X_train_fold.append(X_train[i])
                y_train_fold.append(y_train[i])
            for i in xrange(len(test)):
                X_test_fold.append(X_train[i])
                y_test_fold.append(y_train[i])
            clf = svm.SVC(kernel = 'linear', C = c).fit(X_train_fold,y_train_fold)
            score = score + clf.score(X_test_fold, y_test_fold)
        del X_train_fold[:]
        del y_train_fold[:]
        del X_test_fold[:]
        del y_test_fold[:]
        print "Average score for C = ",c,"=",(score/float(n))*100,"%"
 
   
#Plotting Linear SVM for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = svm.SVC(kernel = 'linear', C = 10).fit(X_train_fold,y_train_fold)
       
    score = clf.score(X_test, y_test)
    print "Score on test data with C = 10 is ",score*100,"%"
           
    Xplot = []
    Yplot = []
    Zplot = []
                   
    clf = svm.SVC(kernel = 'linear', C = 10).fit(AB,label)      
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),np.arange(0, 4.2, 0.2))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
    plot0 = plt.scatter(A0, B0, marker='o', color = 'green')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title("Linear kernel")
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()
         

 
 
#   Polynomial SVM
    print "\nPolynomial SVM"
    print "------------"
     
    CC = [10**-2, 1, 10**2, 10**6]
    D = [2, 3, 4, 5, 6]
     
    for c in CC:
        for d in D: 
            score = 0
            #print len(skf)
            #print skf
            for train, test in skf:
                for i in xrange(len(train)):
                    X_train_fold.append(X_train[i])
                    y_train_fold.append(y_train[i])
                for i in xrange(len(test)):
                    X_test_fold.append(X_train[i])
                    y_test_fold.append(y_train[i])
                clf = svm.SVC(kernel = 'poly', C = c, degree = d).fit(X_train_fold,y_train_fold)
                score = score + clf.score(X_test_fold, y_test_fold)
            del X_train_fold[:]
            del y_train_fold[:]
            del X_test_fold[:]
            del y_test_fold[:]
            print "Average score for C = ",c,"& d = ",d," is",(score/float(n))
  
# Plotting Polynomial SVM for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = clf = svm.SVC(kernel = 'poly', C = 1, degree = 4).fit(X_train_fold,y_train_fold)
        
    score = clf.score(X_test, y_test)
    print "Score on test data with C = 1 and degree = 4 is ",score*100,"%"
    
    clf = svm.SVC(kernel = 'poly', C = 1, degree = 4).fit(AB,label)      
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),np.arange(0, 4.2, 0.2))
    plt.title("Polynomial kernel")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
    plot0 = plt.scatter(A0, B0, marker='o', color = 'green')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()


 
# RBF SVM
    print "\nRBF SVM"
    print "------------"
    CC = [10**-2, 1, 100, 10**6]
    gamma = [0.001, 0.01, 0.1, 1]
    for c in CC:
        for g in gamma:
            score = 0
            for train, test in skf:
                for i in xrange(len(train)):
                    X_train_fold.append(X_train[i])
                    y_train_fold.append(y_train[i])
                for i in xrange(len(test)):
                    X_test_fold.append(X_train[i])
                    y_test_fold.append(y_train[i])
                clf = svm.SVC(kernel = 'rbf', C = c, gamma = g).fit(X_train_fold,y_train_fold)
                score = score + clf.score(X_test_fold, y_test_fold)
            del X_train_fold[:]
            del y_train_fold[:]
            del X_test_fold[:]
            del y_test_fold[:]
            print "Average score for C = ",c,"& gamma = ",g," is",(score/float(n)*100),"%"
      
# Plotting RBF for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = svm.SVC(kernel = 'rbf', C = 100, gamma = 1).fit(X_train_fold,y_train_fold)
        
    score = clf.score(X_test, y_test)
    print "Score on test data with C = 100 and gamma = 1 is ",score*100,"%"
       
    clf = svm.SVC(kernel = 'rbf', C = 100, gamma = 1).fit(AB,label)      
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),
                     np.arange(0, 4.2, 0.2))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
    plot0 = plt.scatter(A0, B0, marker='o', color = 'red')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.title("RBF kernel")
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()
        
   


# Random Forest Classifier
    print "\nRandom Forest Classifier"
    print "---------------------"
    N = [10, 50, 80, 100, 120, 500, 1000, 10**4]
    for param in N:
            score = 0
            for train, test in skf:
                for i in xrange(len(train)):
                    X_train_fold.append(X_train[i])
                    y_train_fold.append(y_train[i])
                for i in xrange(len(test)):
                    X_test_fold.append(X_train[i])
                    y_test_fold.append(y_train[i])
                clf = RandomForestClassifier(n_estimators=param).fit(X_train_fold,y_train_fold)
                score = score + clf.score(X_test_fold, y_test_fold)
            del X_train_fold[:]
            del y_train_fold[:]
            del X_test_fold[:]
            del y_test_fold[:]
            print "Average score for n = ",param,"is", (score/float(n)*100),"%"
  
#Plotting Random Forest Classifier for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = RandomForestClassifier(n_estimators=100).fit(X_train_fold,y_train_fold)
     
    score = clf.score(X_test, y_test)
    print "Score on test data with n_estimators = 100 is ",float(score)*100.0,"%"
   
    clf = RandomForestClassifier(n_estimators=100).fit(AB,label)      
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),np.arange(0, 4.2, 0.2))
    #plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
    plot0 = plt.scatter(A0, B0, marker='o', color = 'green')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.title("Random Forest Classifier")
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()

 
 
Logistic Regression
    print "\nLogistic Regression"
    print "-----------------------"
    CC = [1, 100, 1000, 10**4, 10**6, 10**9]
    for c in CC:
            score = 0
            for train, test in skf:
                for i in xrange(len(train)):
                    X_train_fold.append(X_train[i])
                    y_train_fold.append(y_train[i])
                for i in xrange(len(test)):
                    X_test_fold.append(X_train[i])
                    y_test_fold.append(y_train[i])
                logreg = linear_model.LogisticRegression(C=c).fit(X_train_fold,y_train_fold)
                score = score + logreg.score(X_test_fold, y_test_fold)
            del X_train_fold[:]
            del y_train_fold[:]
            del X_test_fold[:]
            del y_test_fold[:]
            print "Average score for C = ",c,"is", (score/float(n)*100),"%"
  


#Plotting Logistic Regression for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = LogisticRegression(C=1).fit(X_train_fold,y_train_fold)
       
    score = clf.score(X_test, y_test)
    print "Score on test data with C = 1 is ",float(score)*100.0,"%"
     
     
    clf = LogisticRegression(C=50).fit(AB,label) 
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),np.arange(0, 4.2, 0.2))
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
 
    plot0 = plt.scatter(A0, B0, marker='o', color = 'green')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.title("Logistic Regression")
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()

          
# Perceptron
    print "\nPerceptron"
    print "----------------"
    N = [2, 3, 4, 5, 20, 50, 1000]
    for param in N:
        score = 0
        for train, test in skf:
            for i in xrange(len(train)):
                X_train_fold.append(X_train[i])
                y_train_fold.append(y_train[i])
            for i in xrange(len(test)):
                X_test_fold.append(X_train[i])
                y_test_fold.append(y_train[i])
            percept = linear_model.Perceptron(n_iter = param, class_weight="auto").fit(X_train_fold,y_train_fold)
            score = score + percept.score(X_test_fold, y_test_fold)
        del X_train_fold[:]
        del y_train_fold[:]
        del X_test_fold[:]
        del y_test_fold[:]
        print "Average score for n_iter = ",param,"is", (score/float(n)*100),"%"


# Plotting Perceptron for now with the best params
    for train, test in skf:
        for i in xrange(len(train)):
            X_train_fold.append(X_train[i])
            y_train_fold.append(y_train[i])
        for i in xrange(len(test)):
            X_test_fold.append(X_train[i])
            y_test_fold.append(y_train[i])
    clf = Perceptron(n_iter = 3).fit(X_train_fold,y_train_fold)
        
    score = clf.score(X_test, y_test)
    print "Score on test data with n_iter = 3 is ",float(score)*100.0,"%"
         
    clf = Perceptron(n_iter = 3, class_weight="auto").fit(AB,label) 
    Xplot = []
    Yplot = []
    Xplot, Yplot = np.meshgrid(np.arange(0, 4.2, 0.2),np.arange(0, 4.2, 0.2))
    plot0 = plt.scatter(A0, B0, marker='o', color = 'green')
    plot1 = plt.scatter(A1,B1, marker = '+', color = 'black')
    plt.title("Perceptron")
    plt.legend((plot0, plot1), ('label 0', 'label 1'), scatterpoints = 1)
    plt.xlabel('A')
    plt.ylabel('B')
    #plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[Xplot.ravel(), Yplot.ravel()])
    Z = Z.reshape(Xplot.shape)
    plt.contourf(Xplot, Yplot, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()