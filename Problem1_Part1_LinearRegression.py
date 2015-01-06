'''
Created on Nov 11, 2014

@author: nandini1986
'''


import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax

if __name__ == '__main__':
    
    np.matrix('1 2; 3 4')
    
    x = []
    y = []
    i = 0
    f1 = open('girls_train.csv','rb')
    
#1.1a
    data = f1.read()
    idata = data.split('\n')
    for i in xrange(len(idata)):
        element = idata[i].split(",")
        x.append(float(element[0]))
        temp = element[1]
        if i == len(idata):
            y.append(float(element[1]))
        else:
            y.append(float(temp[:-1]))
    print "x = ",x
    print "y = ",y
    print "\n"


#1.1b
        
    plt.scatter(x,y)
    plt.title('Scatter plot')
    plt.xlabel('age')
    plt.ylabel('height')
    plt.show()
    
#1.2a
    alpha = 0.05
    betaZero = 0
    betaOne = 0
    beta0 = []
    beta1 = []
    n = len(x)
    iterations = 1500

    xZero = 1
    for i in xrange(iterations):
        summation1 = 0
        summation2 = 0
        for j in xrange(n):
            summation1 = summation1+(betaZero+betaOne*x[j]-y[j])*float(xZero)
            summation2 = summation2+(betaZero+betaOne*x[j]-y[j])*float(x[j])
        betaZero = betaZero - (alpha/float(n))*float(summation1)
        betaOne = betaOne - (alpha/float(n))*float(summation2)
    print "Beta Zero = ",betaZero
    print "Beta One = ",betaOne
    print "\n"
 
      
#1.2b
    fx = []
    temp = 0.0
    tempsum = 0.0
    for i in xrange(n):
        fx.append(float(betaZero+(betaOne*float(x[i]))))
        #print betaZero+betaOne*x[i]
        temp = float(y[i]) - float(fx[i])
        temp = float(temp*temp)
        tempsum = tempsum + temp
    R = float((1/(2*float(n))))*(tempsum)
    print "R (training) = ",R
    print "\n"
    

#1.3a
    plt.scatter(x,y)
    plt.plot(x, fx)
    plt.title('Scatter plot with Regression Line')
    plt.xlabel('age')
    plt.ylabel('height')
    plt.show()
    
#1.3c
    costfn = []
    numberOfIter = 25
    beta0 = np.linspace(0.0, 0.25, num = numberOfIter)
    beta1 = np.linspace(0.0, 0.07, num = numberOfIter)
    beta00 = []
    beta11 = []

    for i in xrange(numberOfIter):
        for k in xrange(numberOfIter):
            del fx[:]
            temp = 0.0
            tempsum = 0.0
            beta00.append(beta0[i])
            beta11.append(beta1[k])            
            for j in xrange(n):
                fx.append(float(beta0[i]+(float(beta1[k])*x[j])))
                temp = float(y[j]) - float(fx[j])
                temp = float(temp)*float(temp)
                tempsum = tempsum + temp
            costfn.append((float(1/(2*float(n))))*(tempsum))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.plot(beta00, beta11, costfn, linestyle = "none", marker = "o", mfc = "none", markeredgecolor = "red")
    ax.set_title("Cost Function")
    ax.set_xlabel("Beta Zero")
    ax.set_ylabel("Beta One")
    ax.set_zlabel("Cost")
    plt.show()


#1.4a
    age = 4.5
    height = float(betaZero) + float(betaOne*age)
    print "Girl of age",age,"yrs must have height",height,"m per our predicted model.\n"
    
#1.4b
    f2 = open('girls_test.csv','rb')
    data = f2.read()
    del x[:]
    del y[:]
    del fx[:]
    #print data
    idata = data.split('\n')
    for i in xrange(len(idata)):
        idata[i] = idata[i].rstrip('\r')
        element = idata[i].split(',')
        x.append(float(element[0]))
        y.append(float(element[1]))
        
    temp = 0.0
    tempsum = 0.0
    R = 0.0
    n = len(x)
    for i in xrange(len(x)):
        fx.append(float(betaZero+(betaOne*float(x[i]))))
        temp = float(y[i]) - float(fx[i])
        temp = float(temp*temp)
        tempsum = tempsum + temp
    R = float((1/(2*float(n))))*(tempsum)
    print "R (test) = ",R
    print "\n"
    
