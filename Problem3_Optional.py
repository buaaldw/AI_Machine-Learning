'''
Created on Nov 16, 2014

@author: nandini1986
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import transpose

def degree0(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        ageList.append([1])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta



def degree1(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        ageList.append([1, age[i]])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta

def degree2(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        x = age[i]*age[i]
        ageList.append([1, age[i], x])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta


def degree3(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        ageList.append([1, age[i], age[i]**2, age[i]**3])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta


def degree4(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        ageList.append([1, age[i], age[i]**2, age[i]**3, age[i]**4])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta


def degree5(age, height):
    del ageList[:]
    del yList[:]
    for i in xrange(len(age)):
        ageList.append([1, age[i],age[i]**2, age[i]**3, age[i]**4, age[i]**5])
        yList.append(height[i])
    ageMatrix = np.matrix(ageList)
    yMatrix = np.matrix(yList)
    yMatrix = transpose(yMatrix)
    ageMatrixT = transpose(ageMatrix)
    temp = ageMatrixT.dot(ageMatrix)
    temp = temp.getI()
    temp = temp.dot(ageMatrixT)
    beta = temp.dot(yMatrix)
    return beta



if __name__ == '__main__':
    f1 = open("girls_2_20_train.csv")
    data = f1.read()

    age = [2,4,8,9,11,13,15,16,17,18]
    height = [80,93,94,116,135,167.08,174,176,177,178]
    
    ageVal = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20]
    heightVal = [83,81,95,93,90,94,100,114,125,138,140,169,175,176,178,176,180,174,165,176]
    
    n = len(age)
    nVal = len(ageVal)
    ageList = []
    ageMatrix = []
    yList = []
    yMatrix = []

# Part 1 - Print betas

    beta = degree0(age, height)
    print "\nBeta Matrix degree 0 = \n",beta
    beta = degree1(age, height)
    print "\nBeta Matrix degree 1 = \n",beta 
    beta = degree2(age, height)
    print "\nBeta Matrix degree 2 = \n",beta 
    beta = degree3(age, height)
    print "\nBeta Matrix degree 3 = \n",beta 
    beta = degree4(age, height)
    print "\nBeta Matrix degree 4 = \n",beta 
    beta = degree5(age, height)
    print "\nBeta Matrix degree 5 = \n",beta 
    
# Part 2 - Plotting regression lines and points

    plt.scatter(age, height, marker = 'o')
    plots = [0,0,0,0,0,0]
    fx = []
    R = []
    tempsum = 0
    beta = degree0(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0]))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[0], = (plt.plot(age, fx))        
    
    tempsum = 0
    del fx[:]
    beta = degree1(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0])+float(beta[1]*float(age[a])))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[1], = (plt.plot(age, fx))
    
    del fx[:]
    tempsum = 0
    beta = degree2(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0])+(float(beta[1])*float(age[a]))+(float(beta[2])*float(age[a]**2)))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[2], = (plt.plot(age, fx))
    
    tempsum = 0
    del fx[:]
    beta = degree3(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0])+(float(beta[1])*float(age[a]))+(float(beta[2])*float(age[a]**2))+(float(beta[3])*float(age[a]**3)))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[3], = (plt.plot(age, fx))
    
    tempsum = 0
    del fx[:]
    beta = degree4(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0])+(float(beta[1])*float(age[a]))+(float(beta[2])*float(age[a]**2))+(float(beta[3])*float(age[a]**3))+(float(beta[4])*float(age[a]**4)))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[4], = (plt.plot(age, fx))

    tempsum = 0
    del fx[:]
    beta = degree5(age, height)
    for a in xrange(len(age)):
        fx.append(float(beta[0])+(float(beta[1])*float(age[a]))+(float(beta[2])*float(age[a]**2))+(float(beta[3])*float(age[a]**3))+(float(beta[4])*float(age[a]**4))+(float(beta[5])*float(age[a]**5)))
        temp = float(height[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    R.append((float(1/(2*float(n))))*(tempsum))
    plots[5], = (plt.plot(age, fx))    
    
    print "\nR = ",R
    plt.title("Regression Functions")
    plt.xlabel('age')
    plt.ylabel('height')
    plt.legend(plots,["Degree = 0","Degree = 1","Degree = 2","Degree = 3", "Degree = 4", "Degree = 5"], loc = 2)
    plt.show()
    

# Part 3 and 4

    del fx[:]
    Rval = []
    beta = degree0(age, height)
    tempsum = 0
    for a in xrange(nVal):
        fx.append(float(beta[0]))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))       
    
    tempsum = 0
    del fx[:]
    beta = degree1(age, height)
    for a in xrange(nVal):
        fx.append(float(beta[0])+float(beta[1]*float(ageVal[a])))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))
    
    del fx[:]
    tempsum = 0
    beta = degree2(age, height)
    for a in xrange(nVal):
        fx.append(float(beta[0])+(float(beta[1])*float(ageVal[a]))+(float(beta[2])*float(ageVal[a]**2)))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))
    
    tempsum = 0
    del fx[:]
    beta = degree3(age, height)
    for a in xrange(nVal):
        fx.append(float(beta[0])+(float(beta[1])*float(ageVal[a]))+(float(beta[2])*float(ageVal[a]**2))+(float(beta[3])*float(ageVal[a]**3)))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))
    
    tempsum = 0
    del fx[:]
    beta = degree4(age, height)
    for a in xrange(nVal):
        fx.append(float(beta[0])+(float(beta[1])*float(ageVal[a]))+(float(beta[2])*float(ageVal[a]**2))+(float(beta[3])*float(ageVal[a]**3))+(float(beta[4])*float(ageVal[a]**4)))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))

    tempsum = 0
    del fx[:]
    beta = degree5(age, height)
    for a in xrange(nVal):
        fx.append(float(beta[0])+(float(beta[1])*float(ageVal[a]))+(float(beta[2])*float(ageVal[a]**2))+(float(beta[3])*float(ageVal[a]**3))+(float(beta[4])*float(ageVal[a]**4))+(float(beta[5])*float(ageVal[a]**5)))
        temp = float(heightVal[a]) - float(fx[a])
        temp = float(temp)*float(temp)
        tempsum = tempsum + temp
    Rval.append((float(1/(2*float(nVal))))*(tempsum))   
    
    plots2 = [0,0]
    degrees = [0,1,2,3,4,5]
    print "\nR Validation = ",Rval
    
    plots2[0], = plt.plot(R, degrees)
    plots2[1], = plt.plot(Rval,degrees)
    plt.title("Training & Validation Errors")
    plt.ylabel('degree')
    plt.xlabel('mean square error')
    plt.legend(plots2,["R training", "R validation"])
    plt.show()
   
    
    
    