# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:16:49 2017

@author: Nat
"""
import numpy as np
import scipy
import os
os.chdir('D:/projects/partsQuality/data')
import sys
sys.path.append('../python/')
from sklearn.externals import joblib
figureDir = '../figure/'
from matplotlib import pyplot as plt
import seaborn as sns

class BayesChangeDetection(object):
    '''
    class to store default values and intermediate results 
        to avoid repeating calculation of e.g. determinants
    '''
    
    def __init__(self, y):
        self.numZero = np.power(10, -7.0)
        self.defaultSpanOfS = np.arange(-1, 1.1, step = 0.1)
        self.defaultSpanOfBeta = np.arange(0, 1, step = 0.1)
        self.defaultSpanOfSigma = np.arange(-2, 2.5, step = 0.5)
        self.y = self.vecReshape(y)
        self.n = self.y.shape[0]
        self.f = np.zeros([self.n, 4])
        self.om = np.zeros([self.n, self.n])
        self.listOfBasics = []
    
    # https://arxiv.org/pdf/1104.3448.pdf
    def xiMinus(self, theta, t, mode = 'constant'):
        '''
        y = beta0 * xiMinus(mode = constant) + beta1 * xiMinus(mode = linear) + beta2 * xiPlus(mode = linear) + beta3 * xiPlus(mode = constant) + zie
        first part of the signal hocky stick (in case of one change point)
        if mode == 'constant': (denoted phi in the paper) step change with levels beta0 and beta3
        if mode == 'linear': mixed linear with slopes beta1 and beta2
        '''
        if mode == 'constant':
            if t <= theta:
                return 1
            else:
                return 0
        if mode == 'linear':
            if t <= theta:
                return theta - t
            else:
                return 0
    
    def xiPlus(self, theta, t, mode = 'constant'):
        '''
        y = beta0 * xiMinus(mode = constant) + beta1 * xiMinus(mode = linear) + beta2 * xiPlus(mode = linear) + beta3 * xiPlus(mode = constant) + zie
        second part of the signal hocky stick (in case of one change point)
        if mode == 'constant': (denoted phi in the paper) step change with levels beta0 and beta3
        if mode == 'linear': mixed linear with slopes beta1 and beta2    
        '''
        if mode == 'constant':
            if t <= theta:
                return 0
            else:
                return 1
        if mode == 'linear':
            if t <= theta:
                return 0
            else:
                return theta - t

    def fTheta(self, theta):
        '''
        calculates F as in y = Fbeta + xi
        theta: change time point to be tested
        n: number of data points in the time window to be tested
    
        returns a matrix of n x 4
        '''
        f = np.zeros([self.n, 4])
        for j in np.arange(0, self.n):
            f[j][0] = self.xiMinus(theta, j, 'constant')
            f[j][1] = self.xiMinus(theta, j, 'linear')
            f[j][2] = self.xiPlus(theta, j, 'linear')
            f[j][3] = self.xiPlus(theta, j, 'constant')
        self.f = f.copy()
    
    def omegaTheta(self, theta, s):
        '''
        covariance matrix of the signal y.
        assuming all observation points being independent, use kronecker delta.
        s: slopes of the hocky stick of deviation STD(zie(t)), s[0] and s[3] always equal zero
            (no noise slopes estimated for levels).
        returns a matrix of n x n
        '''
        om = np.zeros([self.n,self.n])
        for j in np.arange(0, self.n):
            om[j][j] = np.power(1 + s[1] * self.xiMinus(theta, j, 'linear') + s[2] * self.xiPlus(theta, j, 'linear'), 2)      
            if om[j][j] == 0: 
                om[j][j] = self.numZero
        self.om = om.copy()
    
    def inverseFunc(self, mat):
        '''
        calculate the inverse of a matrix.
        if matrix is singular, then use SVD instead of solve.
        in general be ware of matrix inverses:
            https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/
        '''
        if np.linalg.cond(mat) < 1 / np.finfo(mat.dtype).eps:
            return np.linalg.solve(mat, np.eye(mat.shape[1], dtype = float))
        else:
            U, S, V = scipy.linalg.svd(mat)
            D = np.diag(S)
            tmp = np.matmul(V, np.linalg.inv(D))
            return np.matmul(tmp, np.transpose(U))
    
    def vecReshape(self, vec):
        try:
            vec.shape[1]
        except IndexError:
            vec = vec.reshape(len(vec), 1)
        return vec
    
    def calculateBasics(self, theta, s):
        '''
        search by theta and s if the calculation has been done before
        if not, calculate the matrices and store in a list as object
        '''
        flag = False
        for obj in self.listOfBasics:
            if np.allclose(obj.name, (theta, s[1], s[2])):               
                self.omDet = obj.omDet
                self.omInv = obj.omInv.copy()
                self.fTomInv = obj.fTomInv.copy()
                self.fTomInvf = obj.fTomInvf.copy()
                self.fTomInvfDet = obj.fTomInvfDet
                self.omDetfTomInvfDet = obj.omDetfTomInvfDet
                
                flag = True
                break
        
        if flag is False:
            self.omDet = np.linalg.det(self.om)
            self.omInv = self.inverseFunc(self.om)
            self.fTomInv = np.matmul(np.transpose(self.f), self.omInv)
            self.fTomInvf = np.matmul(self.fTomInv, self.f)
            _, logdet = np.linalg.slogdet(self.fTomInvf)
            self.fTomInvfDet = np.exp(logdet)
            self.omDetfTomInvfDet = self.omDet * self.fTomInvfDet
            obj = Basics(theta, s, self.omDet, self.omInv, self.fTomInv, 
                         self.fTomInvf, self.fTomInvfDet, self.omDetfTomInvfDet)
            self.listOfBasics = self.listOfBasics + [obj]
            
class Basics(object):
    def __init__(self, theta, s, omDet, omInv, fTomInv, fTomInvf, 
                 fTomInvfDet, omDetfTomInvfDet):
        self.name = (theta, s[1], s[2])
        self.omDet = omDet
        self.omInv = omInv.copy()
        self.fTomInv = fTomInv.copy()
        self.fTomInvf = fTomInvf.copy()
        self.fTomInvfDet = fTomInvfDet
        self.omDetfTomInvfDet = omDetfTomInvfDet
            
def betaStar(BayesChangeDetectionClass):
    '''
    calculate best linear unbiased predictor
    returns a matrix of 3 x 1
    '''
    y = BayesChangeDetectionClass.y.copy()
    # inverse of omega is inverse of a diagnal matrix
    fTomInv = BayesChangeDetectionClass.fTomInv.copy()
    tmp = BayesChangeDetectionClass.fTomInvf.copy()
    # instead of inversing tmp, solve the following equation numerically:
    # tmp * betastar = fTomInv * y
    target = np.matmul(fTomInv, y)
    betastar = np.linalg.solve(tmp, target)
    return betastar

def residuumSq(BayesChangeDetectionClass, betastar):
    '''
    calculate residuum R^2 as distance between vectors y and F * betastar
        both belongs to the same distribution (Mahalanobis distance)
    returns a single value
    '''
    y = BayesChangeDetectionClass.y.copy()
    f = BayesChangeDetectionClass.f.copy()
    omInv = BayesChangeDetectionClass.omInv.copy()
    
    fBetastar = np.matmul(f, betastar)
    ytoFBetastar = np.subtract(y, fBetastar)
    ytoFBetastarT = np.transpose(ytoFBetastar)
    tmp = np.matmul(ytoFBetastarT, omInv)
    result = np.matmul(tmp, ytoFBetastar)
    return result.tolist()[0][0]

def sigmaHat(residuumSq, n):
    '''
    calculates estimator of scale parameter sigma, or intrinsic fluctuation.
    returns a single value.
    '''
    return np.sqrt(residuumSq / (n + 1))


def probSigmaThetaS(BayesChangeDetectionClass, rsq, sigma):
    '''
    p(sigma, theta, s | y) = integral(p(beta, sigma, theta, s | y) dbeta 
    not normalized.
    
    since the integral over theta and s cannot be carried out analytically, 
        we will use a numeric integral to find probability density distribution 
        of sigma given y. 
    
    sigma is the itrinsic variability of the signal.
    
    returns a single value with given paramters.
    '''
    n = BayesChangeDetectionClass.n
    product = BayesChangeDetectionClass.omDetfTomInvfDet
    numZero = BayesChangeDetectionClass.numZero

    if product <= numZero:
        return 0
    
    power = - rsq / (2 * np.power(sigma, 2))
    e = np.exp(power)
    numerator = np.power(sigma, (1-n)) * e
    
    denominator = np.sqrt(product)
    return numerator / denominator

def probBetaThetaS(BayesChangeDetectionClass, beta):
    '''
    p(beta, theta, s | y) = integral(p(beta, sigma, theta, s | y) dsigma
    not normalized.
    
    since the integral over theta and s cannot be carried out analytically, 
        we will use a numeric integral to find probability density distribution 
        of beta given y. 
    
    beta are the levels (in step change) and slopes (in 
        continuous linear trend change) of signal y 
        before and after change point theta.
    
    returns a single value with given paramters.    
    '''
    beta = np.array(beta)
    omDet = BayesChangeDetectionClass.omDet
    numZero = BayesChangeDetectionClass.numZero
    y = BayesChangeDetectionClass.y.copy()
    beta = BayesChangeDetectionClass.vecReshape(beta)
    f = BayesChangeDetectionClass.f.copy()
    omInv = BayesChangeDetectionClass.omInv.copy()
    n = BayesChangeDetectionClass.n
    
    denominator = np.sqrt(omDet)
    if denominator <= numZero:
        return 0
    
    fBeta = np.matmul(f, beta)
    ytofBeta = np.subtract(y, fBeta)
    tmp = np.matmul(np.transpose(ytofBeta), omInv)
    numerator = np.matmul(tmp, ytofBeta)[0][0]
    
    result = np.power(numerator, -(n / 2.0)) / denominator
    return result

def probThetaS(BayesChangeDetectionClass, rsq):
    '''
    p(theta, s | y) = integral(p(beta, sigma, theta, s | y) dsigma dbeta
    not normalized.
    
    since the integral over theta and s cannot be carried out analytically, 
        we will use a numeric integral to find probability density distribution 
        of theta given y, as well as the probability density distribution of s 
        given y.
    
    theta is the change point in time. s is the noise slopes before and after 
        the change point theta.

    returns a single value with given paramters.
    '''
    numZero = BayesChangeDetectionClass.numZero
    n = BayesChangeDetectionClass.n
    product = BayesChangeDetectionClass.omDetfTomInvfDet
    if product <= numZero:
        return 0
    
    denominator = np.sqrt(product)
    numerator = np.power(rsq, -(n - 2.0) / 2.0)
        
    result = numerator / denominator
    return result

def normalized(BayesChangeDetectionClass, mat):
    '''
    returns normalized probability density in the n-dimensional parameter space
    '''
    total = np.sum(mat)
#    if total <= BayesChangeDetectionClass.numZero:
#        return np.zeros(mat.shape)
    return mat.copy() / total

def getThetaRange(n):
    return np.arange(1.0, n, step = 1.0)

def probTheta(BayesChangeDetectionClass):
    '''
    marginalization of s to get probability density of theta from p(theta, s | y)
    normalized.

    returns a vector of shape (n,) containing discrete probability of the ith 
        time point being a change point theta, i belongs to [0, n)
    '''
    n = BayesChangeDetectionClass.n
    integralRangeS = BayesChangeDetectionClass.defaultSpanOfS
    integralRangeTheta = getThetaRange(n)
    
    result = np.zeros(len(integralRangeTheta))
    for theta in integralRangeTheta:
        
        integralS = 0.0
        BayesChangeDetectionClass.fTheta(theta)
        for s1 in integralRangeS:
            for s2 in integralRangeS:
                s = [0, s1, s2, 0]
                BayesChangeDetectionClass.omegaTheta(theta, s)
                BayesChangeDetectionClass.calculateBasics(theta, s)
                try:
                    betastar = betaStar(BayesChangeDetectionClass)
                except np.linalg.LinAlgError:
                    continue
                    
                rsq = residuumSq(BayesChangeDetectionClass, betastar)
    
                tmpIntegral = probThetaS(BayesChangeDetectionClass, rsq)
                if tmpIntegral < 0:
#                    if tmpIntegral > BayesChangeDetectionClass.numZero:
#                        print('warning: integral < 0: theta: %f, s1: %f, s2: %f, integral: %s.' %(theta, s1, s2, np.float64(tmpIntegral).astype(str)))
                    tmpIntegral = 0
                    
                integralS = integralS + tmpIntegral

        result[list(integralRangeTheta).index(theta)] = integralS
    return normalized(BayesChangeDetectionClass, result)

def probS(BayesChangeDetectionClass):
    '''
    marginalization of theta to get probability density of s from p(theta, s | y).
        since s0 := 0, we only need to find p(s1, s2 | y).
    normalized.
    
    returns a spanOfS x spanOfS matrix containing the probability of having 
        noise slopes (s1, s2) as one value in the matrix
    '''
    n = BayesChangeDetectionClass.n
    integralRangeS = BayesChangeDetectionClass.defaultSpanOfS
    integralRangeTheta = getThetaRange(n)
    
    # 2-dimensional probability density
    result = np.zeros([len(integralRangeS), len(integralRangeS)])
    for s1 in integralRangeS:
        for s2 in integralRangeS:
        
            integralTheta = 0
            s = [0, s1, s2, 0]
            for theta in integralRangeTheta:
                BayesChangeDetectionClass.fTheta(theta)
                BayesChangeDetectionClass.omegaTheta(theta, s)
                BayesChangeDetectionClass.calculateBasics(theta, s)
                
                betastar = betaStar(BayesChangeDetectionClass)
                rsq = residuumSq(BayesChangeDetectionClass, betastar)
                
                integralTheta = integralTheta + probThetaS(BayesChangeDetectionClass, rsq)    

            result[s1][s2] = integralTheta
    return normalized(BayesChangeDetectionClass, result)

def probBeta(BayesChangeDetectionClass):
    '''
    marginalization of theta and s to get probability density of beta 
        from p(beta, theta, s | y).
    normalized.
    
    returns a 4-dimensional matrix of shape (spanOfBetaConst, spanOfBetaLinear, spanOfBetaLinear, spanOfBetaConst).
        the matrix contains probability of having the combination of 
        signal level before changepoint / slope before changepoint / slope after changepoint / level after changepoint 
        (beta0, beta1, beta2, beta3).
    '''
    n = BayesChangeDetectionClass.n
    integralRangeBetaConst = BayesChangeDetectionClass.defaultSpanOfBeta
    print(integralRangeBetaConst) 
    
    integralRangeBetaLinear = np.array([0.0])
        
    integralRangeS = BayesChangeDetectionClass.defaultSpanOfS
    integralRangeTheta = getThetaRange(n)   
    
    result = np.zeros([len(integralRangeBetaConst), 
                       len(integralRangeBetaLinear), 
                       len(integralRangeBetaLinear),
                       len(integralRangeBetaConst)])
    for beta0 in integralRangeBetaConst:
        for beta1 in integralRangeBetaLinear:
            for beta2 in integralRangeBetaLinear:
                for beta3 in integralRangeBetaConst:
                
                    integralThetaS = 0
                    beta = [beta0, beta1, beta2, beta3]
                    
                    for s1 in integralRangeS:
                        for s2 in integralRangeS:
                            s = [0, s1, s2, 0]
                            for theta in integralRangeTheta:
                                BayesChangeDetectionClass.fTheta(theta)
                                BayesChangeDetectionClass.omegaTheta(theta, s)
                                BayesChangeDetectionClass.calculateBasics(theta, s)
                                
                                integralThetaS = integralThetaS + probBetaThetaS(BayesChangeDetectionClass, beta)
                    
                    b0 = list(integralRangeBetaConst).index(beta0)
                    b1 = list(integralRangeBetaLinear).index(beta1)
                    b2 = list(integralRangeBetaLinear).index(beta2)
                    b3 = list(integralRangeBetaConst).index(beta3)
                    print('b0 %d, b1 %d, b2 %d, b3 %d.' %(b0, b1, b2, b3))
                    result[b0][b1][b2][b3] = integralThetaS
    return normalized(BayesChangeDetectionClass, result)

def probSigma(BayesChangeDetectionClass):
    '''
    marginalization of theta and s to get probability density of sigma 
        from p(sigma, theta, s | y).
    normalized.
    
    returns a vector of shape(spanOfSigma, ), containing probability of having
        a certain sigma value.
    '''
    n = BayesChangeDetectionClass.n
    integralRangeSigma = BayesChangeDetectionClass.defaultSpanOfSigma
    integralRangeS = BayesChangeDetectionClass.defaultSpanOfS
    integralRangeTheta = getThetaRange(n)   

    result = np.zeros(len(integralRangeSigma))

    for sigma in integralRangeSigma:
        
        integralThetaS = 0        
        for s1 in integralRangeS:
            for s2 in integralRangeS:
                s = [0, s1, s2]
                for theta in integralRangeTheta:
                    BayesChangeDetectionClass.fTheta(theta)
                    BayesChangeDetectionClass.omegaTheta(theta, s)
                    BayesChangeDetectionClass.calculateBasics(theta, s)
                    
                    betastar = betaStar(BayesChangeDetectionClass)
                    rsq = residuumSq(BayesChangeDetectionClass, betastar)                    
                    
                    integralThetaS = integralThetaS + probSigmaThetaS(BayesChangeDetectionClass, rsq, sigma)
        result[sigma] = integralThetaS
    return normalized(BayesChangeDetectionClass, result)
    
def probThetaWrapper(signal, title = 'testLabel', LSL = [], HSL = [], window = 50, interval = 20):

    yAll = np.array(signal, float)
    avg = yAll.mean()
    yAll = np.subtract(yAll, avg) / avg
    threshold = 0.5 * min([np.abs(avg - LSL.mean()), np.abs(avg - HSL.mean())])
    
    resultAll = []
    alarm = []
    idx = 0
    
    # keep one class for record of basic matrix calculation results
    y = yAll[: window]
    bayesClass = BayesChangeDetection(y)
    for i in np.arange(0, len(yAll) - window, step = interval):
        y = yAll[i: i + window]
        bayesClass.y = bayesClass.vecReshape(y)
    
        result = probTheta(bayesClass)
        theta = np.argmax(result)
        beta0 = np.mean(y[0:theta]) * avg + avg
        beta3 = np.mean(y[theta:]) * avg + avg
        
        resultAll = resultAll + [result]
        if np.abs(beta0 - beta3) >= threshold:
            alarm[idx] = i + theta + 1
            idx += 1
        
        yplot = np.array(signal)[i: i + window]
        plotTitle = title + '_' + str(i) + '-' + str(i + window)
        myQualityPlot(yplot, _LSL = LSL[i: i + window], _HSL = HSL[i: i + window], title = plotTitle, 
                 changePoint = theta, trendBeforeChange = np.ones(theta) * beta0, 
                 trendAfterChange = np.ones(len(yplot) - theta) * beta3, 
                 secondAxis = [0] + list(result))
        
        if i == 0:
            joblib.dump(bayesClass.listOfBasics, 'listOfBasics_' + str(window) +'.pkl')
    
    return result, alarm

def myQualityPlot(_yplot, _x = None, LSL = [], HSL = [], title = None, 
             changePoint = None, trendBeforeChange = [], trendAfterChange = [], 
             secondAxis = []):
    if _x is None:
        x = np.arange(len(_yplot))
    else:
        x = np.array(_x)
    yplot = np.array(_yplot)

    fig, ax1 = plt.subplots()
    ax1.set_title(title)

    ax1.plot(x, yplot, 'blue')
    if len(LSL) > 0:
        ax1.plot(x, LSL, '-', color = 'red')
    if len(HSL) > 0:
        ax1.plot(x, HSL, '-', color = 'red')
    
    if changePoint is not None and len(trendBeforeChange) > 0:
        ax1.plot(x[0: changePoint], trendBeforeChange, 'green')
    if changePoint is not None and len(trendAfterChange) > 0:   
        ax1.plot(x[changePoint:], trendAfterChange, 'green')
    
    if len(secondAxis) > 0:
        ax2 = ax1.twinx()
        ax2.plot(x, secondAxis, 'orange')

    plt.savefig(figureDir + title + 'qualityPlot.png')
    plt.close()
    
def myDistrPlot(_y, LSL = [], HSL = [], title = None, sigma = 3):
    if len(LSL) > 0:
        LSLvalue = LSL.tolist()[0]
    else:
        LSLvalue = sys.maxsize
    if len(HSL) > 0:
        HSLvalue = HSL.tolist()[0]
    else: 
        HSLvalue = - sys.maxsize
    
    yplot = np.array(_y)
    stdev = np.std(yplot)
    
    stdevLower = np.mean(yplot) - sigma * stdev
    stdevHigher = np.mean(yplot) + sigma * stdev
    
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    
    dp = sns.distplot(yplot, color = 'darkred')
    axes = dp.axes
    # get xlim
    data_x, data_y = dp.lines[0].get_data()
    xmin = np.min(data_x)
    xmax = np.max(data_x)
    # set minimum distance to border
    spacer = np.min([10, 0.1 * (xmax - xmin)])
    axes.set_xlim(min([xmin, LSLvalue - spacer, stdevLower - spacer]), 
                  max([xmax, HSLvalue + spacer, stdevHigher - spacer]))
    
    # get ylim
    ymax = np.max(data_y)
    vLine = [0, ymax]
    if len(LSL) > 0:
        # draw vertical line of LSL
        LSLx = np.ones(2) * LSLvalue
        ax1.plot(LSLx, vLine, color = 'orange')
    if len(HSL) > 0:
        # draw vertical line of HSL
        HSLx = np.ones(2) * HSLvalue
        ax1.plot(HSLx, vLine, color = 'orange')
    # draw vertical line of sigma * standard deviation
    stdLx = np.ones(2) * stdevLower
    stdHx = np.ones(2) * stdevHigher
    ax1.plot(stdLx, vLine, color = 'darkred')
    ax1.plot(stdHx, vLine, color = 'darkred')
    
    plt.savefig(figureDir + title + '_distribution.png')
    plt.close()
    
    return data_x, data_y
