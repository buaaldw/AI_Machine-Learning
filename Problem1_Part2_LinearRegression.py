'''
Created on Nov 15, 2014

@author: nandini1986
'''

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax
from scipy.stats.morestats import ansari
from numpy import transpose


if __name__ == '__main__':
    x1 = []
    x2 = []
    y = []
    x1Scaled = []
    x2Scaled = []
    
# 2.1 - Data separation & normalization    
    f1 = open('girls_age_weight_height_2_8.csv','rb')
    data = f1.read()
    idata = data.split('\n')
    for i in xrange(len(idata)):
        idata[i] = idata[i].rstrip('\r')
        element = idata[i].split(',')
        x1.append(float(element[0]))
        x2.append(float(element[1]))
        y.append(float(element[2]))
    
    meanX1 = np.mean(x1)
    meanX2 = np.mean(x2)
    
    stdX1 = np.std(x1)
    stdX2 = np.std(x2)
    
    print "Mean age = ",meanX1
    print "Mean weight = ", meanX2
    print "Std dev age = ",stdX1
    print "Std dev weight = ",stdX2
    
    n = len(x1)
    for i in xrange(n):
        x1Scaled.append(float((x1[i]-meanX1)/stdX1))
        x2Scaled.append(float((x2[i]-meanX2)/stdX2))
    print x1Scaled
    print x2Scaled
    
# 2.2 - Gradient Descent
    alpha = 0.05
    betaZero = 0.0
    betaOne = 0.0
    betaTwo = 0.0
    iterations = 1500    
    
    xZero = 1
    for i in xrange(iterations):
        summation1 = 0
        summation2 = 0
        summation3 = 0
        for j in xrange(n):
            summation1 = summation1+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*xZero
            summation2 = summation2+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*x1Scaled[j]
            summation3 = summation3+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*x2Scaled[j]
        betaZero = betaZero - (alpha/float(n))*summation1
        betaOne = betaOne - (alpha/float(n))*summation2
        betaTwo = betaTwo - (alpha/float(n))*summation3
    print "\nBeta Zero = ",betaZero
    print "Beta One = ",betaOne
    print "Beta Two = ",betaTwo
    
    fx = []
    temp = 0.0
    tempsum = 0.0
    for i in xrange(n):
        fx.append(float(float(betaZero)+(betaOne*x1Scaled[i])+betaTwo*x2Scaled[i]))
        temp = float(y[i]) - float(fx[i])
        temp = float(temp)*float(temp)
        tempsum = float(tempsum) + float(temp)
    R = float((1/(2*float(n))))*(float(tempsum))
    print "R (training) = ",R
    print "\n"

#2.3a - Plotting Risk function for different learning rates
    alphas = [0.005, 0.001, 0.05, 0.1, 0.5, 1.0]
    numbOfIter = 50
    costfn = [[],[],[],[],[],[]]
    fx = 0.0
    finalR =[]
    R_final=[]
    
    for i in xrange(len(alphas)):
        betaZero = 0.0
        betaOne = 0.0
        betaTwo = 0.0
        xZero = 1.0

        for j in xrange(numbOfIter):
            summation1 = 0.0
            summation2 = 0.0
            summation3 = 0.0
            for k in xrange(n):
                summation1 = float(summation1+float((betaZero+float(betaOne*x1Scaled[k])+float(betaTwo*x2Scaled[k])-y[k])*xZero))
                summation2 = float(summation2+float((betaZero+float(betaOne*x1Scaled[k])+float(betaTwo*x2Scaled[k])-y[k])*x1Scaled[k]))
                summation3 = float(summation3+float((betaZero+float(betaOne*x1Scaled[k])+float(betaTwo*x2Scaled[k])-y[k])*x2Scaled[k]))
            
            betaZero = float(betaZero) - (float(alphas[i])/float(n))*float(summation1)
            betaOne = betaOne - (float(alphas[i])/float(n))*float(summation2)
            betaTwo = betaTwo - (float(alphas[i])/float(n))*float(summation3)
            temp = 0.0
            tempsum = 0.0
            for k in xrange(n):
                fx = float(betaZero+(betaOne*x1Scaled[k])+betaTwo*x2Scaled[k])
                temp = float(y[k] - fx)
                temp = float(temp*temp)
                tempsum = float(tempsum + temp)
                temp = (1/(2*float(n)))*(tempsum)
            costfn[i].append(float(temp))

    
    x = []
    a = [0,0,0,0,0,0]
    for i in xrange(numbOfIter):
        x.append(i)
        
    for i in xrange(len(alphas)):
        plt.plot(x, costfn[i])
        a[i], = plt.plot(x,costfn[i])
    plt.legend(a,["Alpha = 0.005","Alpha = 0.001","Alpha = 0.05","Alpha = 0.1","Alpha = 0.5","Alpha = 1.0"])
    plt.title('Risk function for different learning rates')
    plt.xlabel('iterations')
    plt.ylabel('Risk')
    plt.show()
    

# 2.3c - Gradient descent for best alpha
    alpha = 1.0
    betaZero = 0.0
    betaOne = 0.0
    betaTwo = 0.0
    iterations = 1500    
    
    xZero = 1
    for i in xrange(iterations):
        summation1 = 0
        summation2 = 0
        summation3 = 0
        for j in xrange(n):
            summation1 = summation1+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*xZero
            summation2 = summation2+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*x1Scaled[j]
            summation3 = summation3+(betaZero+betaOne*x1Scaled[j]+betaTwo*x2Scaled[j]-y[j])*x2Scaled[j]
        betaZero = betaZero - (alpha/float(n))*summation1
        betaOne = betaOne - (alpha/float(n))*summation2
        betaTwo = betaTwo - (alpha/float(n))*summation3
    print "Best Alpha = ",alpha
    print "\nBeta Zero = ",betaZero
    print "Beta One = ",betaOne
    print "Beta Two = ",betaTwo

#2.3d - Prediction
    age = 5
    weight = 20
    ageScaled = float((age-meanX1))/float(stdX1)
    weightScaled = float((weight - meanX2))/float(stdX2)
    Ans = betaZero + float(betaOne*ageScaled) + float(betaTwo*weightScaled)
    print "\nUsing Gradient Descent:"
    print "A "+str(age)+"-year old girl weighing "+str(weight)+"kilos should have height "+str(Ans)+"m.\n"
    
#2.4a - Normal Equation
    print "\nNormal Equation:"
    feature = []
    yList = []
    for i in xrange(n):
        feature.append([1, x1[i], x2[i]])
        yList.append(y[i])
    featureMatrix = np.matrix(feature)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    featureT = transpose(featureMatrix)
    temp = featureT.dot(featureMatrix)
    temp = temp.getI()
    temp = temp.dot(featureT)
    beta = temp.dot(yMatrix)
    print "Beta Matrix = \n",beta

#2.4b - Prediction
    featureNormalEquation = [1, 5, 20]
    featureNormalEquation = np.matrix(featureNormalEquation)
    AnsNormalEquation = featureNormalEquation.dot(beta)
    ht = float(AnsNormalEquation[0][0])
    print "\nUsing Normal Equation:"
    print "A "+str(age)+"-year old girl weighing "+str(weight)+"kilos should have height "+str(ht)+"m.\n"