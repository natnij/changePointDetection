# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:27:20 2017
"""
import numpy as np
import pandas as pd
import scipy
from datetime import datetime
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression as lr
from matplotlib import pyplot as plt
from multiprocessing import Pool
import gc

import warnings
warnings.filterwarnings("ignore")

class BayesChangeDetection(object):
    '''
    class to store default values and intermediate results
        to avoid repeating calculation of e.g. determinants
    '''
    window = 100
    numZero = np.power(10, -100.0)
    minDataSize = 100
    topItems = 2 # defines which items are to be tested for change point
    scale = np.power(10, np.ceil(np.log10(window + 1)))
    defaultSpanOfS = np.arange(-3.0 * scale,  3.1 * scale, step = 0.3 * scale)
    defaultSpanOfBeta = np.arange(-3.0 * scale, 3.1 * scale, step = 0.3 * scale)
    defaultSpanOfSigma = np.arange(-2, 2.5, step = 0.5)
    figureDir = '../figure/'
    dataDir = '../data/'
    n = 0
    alarm = pd.DataFrame()
    result = {}
    levelThres = 50
    slopeThres = 1
    mcmcIter = 10
    ignoredDataPoints = 20
    drawChartFromPython = True

    def __init__(self):
        self.setGlobals()
        self.pool = Pool()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, ls):
        if isinstance(ls, (list, pd.core.series.Series, np.array)) and len(ls) > 0:
            self._y = self.vecReshape(ls)
            if self.n != self._y.shape[0]:
                self.n = self._y.shape[0]
                self.f = np.zeros([self.n, 4])
                self.om = np.zeros([self.n, self.n])
                self.scale = np.power(10, np.ceil(np.log10(self.n + 1)))
                self.defaultSpanOfS = np.arange(-3.0 * self.scale,  3.1 * self.scale, step = 0.3 * self.scale)
                self.defaultSpanOfBeta = np.arange(-3.0 * self.scale, 3.1 * self.scale, step = 0.3 * self.scale)
        else:
            print('input format not allowed. returning empty list.')
            self._y = []
            self.n = 0

    @y.deleter
    def y(self):
        self._y = []
        self.n = 0
        self.f = []
        self.om = []
        self.omDet = 0
        self.omInv = []
        self.fTomInv = []
        self.fTomInvf = []
        self.fTomInvfDet = 0
        self.omDetfTomInvfDet = 0
        self.betastarValue = 0
        self.rsq = 0
        self.sigmaHat = 0

    def setGlobals(self):
        try:
            self.listOfBasics = joblib.load(self.dataDir + 'newListOfBasics_' + str(self.window) + '.pkl')
        except FileNotFoundError:
            self.listOfBasics = {}

        result_simpleAvg = pd.read_csv(self.dataDir + 'items.csv')
        result_simpleAvg = result_simpleAvg[result_simpleAvg['count'] >= self.minDataSize]
        tmp = result_simpleAvg[['feature','value', 'target']]
        tmp = tmp.iloc[0:self.topItems,:]
        testFeature = list(set(tmp['feature']))
        df = pd.read_csv(self.dataDir + 'df.csv')

        df2 = pd.DataFrame()
        for i in np.arange(tmp.shape[0]):
            feature = tmp['feature'].tolist()[i]
            value = tmp['value'].tolist()[i]
            target = tmp['target'].tolist()[i]
            selected = df.loc[df[feature] == value, :]
            selected['targetValue'] = selected[target]
            selected[feature] = selected[feature] + target
            otherCol = [x for x in testFeature if x != feature]
            selected[otherCol] = None
            df2 = pd.concat([df2, selected], axis = 0)

        df2 = df2.loc[:, ['targetValue'] + testFeature]
        onehot = pd.get_dummies(df2)

        self.col = []
        self.dictOfData = {}
        for col in onehot.columns[1:]:
            tmp = np.array(onehot[col], dtype = bool)
            result = onehot[tmp]['targetValue']

            if len(result) >= self.window * 2:
                m = int(len(result) / self.window)
                idx = np.arange(-self.window, 0, 1) * m
            else:
                idx = np.arange(-(np.min([self.window, len(result)])), 0, 1)
            self.dictOfData[col] = result.iloc[idx]
            self.col.append(col)

        del onehot, df, df2, selected, tmp, result_simpleAvg
        gc.collect()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    # https://arxiv.org/pdf/1104.3448.pdf
    def xiMinus(self, theta, t, mode = 'constant'):
        '''
        y = beta0 * xiMinus(mode = constant) + beta1 * xiMinus(mode = linear) + beta2 * xiPlus(mode = linear) + beta3 * xiPlus(mode = constant) + zie
        first part of the signal hocky stick (in case of one change point)
        if mode == 'constant': (denoted phi in the paper) step change with levels beta0 and beta3
        if mode == 'linear': mixed linear with slopes beta1 and beta2

        to avoid overflow in determinant calculation, xiMinus and xiPlus are scaled down, beta and s are scaled up.
        '''
        scale = self.scale
        if mode == 'constant':
            if t <= theta:
                return 1.0 / scale
            else:
                return 0.0
        if mode == 'linear':
            if t <= theta:
                return 1.0 / scale * (theta - t)
            else:
                return 0.0

    def xiPlus(self, theta, t, mode = 'constant'):
        '''
        y = beta0 * xiMinus(mode = constant) + beta1 * xiMinus(mode = linear) + beta2 * xiPlus(mode = linear) + beta3 * xiPlus(mode = constant) + zie
        second part of the signal hocky stick (in case of one change point)
        if mode == 'constant': (denoted phi in the paper) step change with levels beta0 and beta3
        if mode == 'linear': mixed linear with slopes beta1 and beta2

        to avoid overflow in determinant calculation, xiMinus and xiPlus are scaled down, beta and s are scaled up.
        '''
        scale = self.scale
        if mode == 'constant':
            if t <= theta:
                return 0.0
            else:
                return 1.0 / scale
        if mode == 'linear':
            if t <= theta:
                return 0.0
            else:
                return 1.0 / scale * (theta - t)

    def fTheta(self, theta):
        '''
        calculates F as in y = Fbeta + xi
        theta: change time point to be tested
        n: number of data points in the time window to be tested

        returns a matrix of n x 4
        '''
        for j in np.arange(0, self.n):
            self.f[j][0] = self.xiMinus(theta, j, 'constant')
            self.f[j][1] = self.xiMinus(theta, j, 'linear')
            self.f[j][2] = self.xiPlus(theta, j, 'linear')
            self.f[j][3] = self.xiPlus(theta, j, 'constant')

    def omegaTheta(self, theta, s):
        '''
        covariance matrix of the signal y.
        assuming all observation points being independent, use kronecker delta.
        s: slopes of the hocky stick of deviation STD(zie(t)), s[0] and s[3] always equal zero
            (no noise slopes estimated for levels).
        returns a matrix of n x n
        '''
        for j in np.arange(0, self.n):
            self.om[j][j] = (np.power(1 + s[1]
            	* self.xiMinus(theta, j, 'linear')
            	+ s[2] * self.xiPlus(theta, j, 'linear'), 2) )
            if self.om[j][j] == 0:
                self.om[j][j] = self.numZero

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
            vec = vec.reshape(-1, 1)
        return vec

    def calculateBasics(self, theta, s):
        '''
        search by theta and s if the calculation has been done before
        if not, calculate the matrices and store in a list as object
        '''

        try:
            obj = self.listOfBasics[(theta, s[1], s[2])]
            self.omDet = obj[0]
            self.omInv = self.inverseFunc(self.om)
            self.fTomInv = np.matmul(np.transpose(self.f), self.omInv)
            self.fTomInvf = np.matmul(self.fTomInv, self.f)
            self.fTomInvfDet = obj[1]
            self.omDetfTomInvfDet = obj[2]
        except:
            _, omLogdet = np.linalg.slogdet(self.om)
            self.omDet = np.exp(omLogdet)
            self.omInv = self.inverseFunc(self.om)
            self.fTomInv = np.matmul(np.transpose(self.f), self.omInv)
            self.fTomInvf = np.matmul(self.fTomInv, self.f)
            _, logdet = np.linalg.slogdet(self.fTomInvf)
            self.fTomInvfDet = np.exp(logdet)
            self.omDetfTomInvfDet = self.omDet * self.fTomInvfDet
            obj = [self.omDet, self.fTomInvfDet, self.omDetfTomInvfDet]
            self.listOfBasics[(theta, s[1], s[2])] = obj.copy()

    def betaStar(self):
        '''
        calculate best linear unbiased predictor
        returns a matrix of 3 x 1
        '''
        tmp = self.fTomInvf
        # instead of inversing tmp, solve the following equation numerically:
        # tmp * betastar = fTomInv * y
        target = np.matmul(self.fTomInv, self._y)
        self.betastarValue = np.linalg.solve(tmp, target)

    def residuumSq(self):
        '''
        calculate residuum R^2 as distance between vectors y and F * betastar
            both belongs to the same distribution (Mahalanobis distance)
        returns a single value
        '''

        fBetastar = np.matmul(self.f, self.betastarValue)
        ytoFBetastar = np.subtract(self._y, fBetastar)
        ytoFBetastarT = np.transpose(ytoFBetastar)
        tmp = np.matmul(ytoFBetastarT, self.omInv)
        result = np.matmul(tmp, ytoFBetastar)
        self.rsq = result.tolist()[0][0]

    def sigmaHat(self):
        '''
        calculates estimator of scale parameter sigma, or intrinsic fluctuation.
        returns a single value.
        '''
        self.sigmaHat = np.sqrt(self.rsq / (self.n + 1))


    def probSigmaThetaS(self, sigma):
        '''
        p(sigma, theta, s | y) = integral(p(beta, sigma, theta, s | y) dbeta
        not normalized.

        since the integral over theta and s cannot be carried out analytically,
            we will use a numeric integral to find probability density distribution
            of sigma given y.

        sigma is the itrinsic variability of the signal.

        returns a single value with given paramters.
        '''
        product = self.omDetfTomInvfDet

        if product <= self.numZero:
            return 0

        power = - self.rsq / (2 * np.power(sigma, 2))
        e = np.exp(power)
        numerator = np.power(sigma, (1-self.n)) * e

        denominator = np.sqrt(product)
        return numerator / denominator

    def probBetaThetaS(self, beta):
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
        beta = self.vecReshape(np.array(beta))

        denominator = np.sqrt(self.omDet)
        if denominator <= self.numZero:
            return 0

        fBeta = np.matmul(self.f, beta)
        ytofBeta = np.subtract(self._y, fBeta)
        tmp = np.matmul(np.transpose(ytofBeta), self.omInv)
        numerator = np.matmul(tmp, ytofBeta)[0][0]

        result = np.power(numerator, -(self.n / 2.0)) / denominator
        return result

    def probThetaS(self):
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
        product = self.omDetfTomInvfDet
#        if product <= self.numZero:
#            return 0

        denominator = np.sqrt(product)
        numerator = np.power(self.rsq, -(self.n - 2.0) / 2.0)

        result = numerator / denominator
        return result

    def normalized(self, mat):
        '''
        returns normalized probability density in the n-dimensional parameter space
        '''
        total = np.sum(mat)
        return mat.copy() / total

    def getThetaRange(self):
        return np.arange(1.0, self.n, step = 1.0)

    def probS(self, integralRangeS1, integralRangeS2, theta):

        if len(integralRangeS1) == 1:
            result = np.zeros(len(integralRangeS2))
        else:
            result = np.zeros(len(integralRangeS1))
        self.fTheta(theta)
        for s1 in integralRangeS1:
            for s2 in integralRangeS2:
                s = [0, s1, s2, 0]
                self.omegaTheta(theta, s)
                self.calculateBasics(theta, s)
                try:
                    self.betaStar()
                except np.linalg.LinAlgError:
                    continue
                self.residuumSq()

                tmpIntegral = self.probThetaS()
                if tmpIntegral < 0:
                    tmpIntegral = 0

                if len(integralRangeS1) == 1:
                    result[list(integralRangeS2).index(s2)] = tmpIntegral
                else:
                    result[list(integralRangeS1).index(s1)] = tmpIntegral

        return self.normalized(result)

    def probTheta(self, s1, s2):

        integralRangeTheta = self.getThetaRange()

        result = np.zeros(len(integralRangeTheta))
        for theta in integralRangeTheta:
            self.fTheta(theta)
            s = [0, s1, s2, 0]
            self.omegaTheta(theta, s)
            self.calculateBasics(theta, s)
            try:
                self.betaStar()
            except np.linalg.LinAlgError:
                continue
            self.residuumSq()

            tmpIntegral = self.probThetaS()
            if tmpIntegral < 0:
                tmpIntegral = 0

            result[list(integralRangeTheta).index(theta)] = tmpIntegral

        if self.ignoredDataPoints > 0:
            result[:self.ignoredDataPoints] = 0
            result[-self.ignoredDataPoints:] = 0

        return self.normalized(result)

    def initializeMCMC(self):
        initS1 = np.median(self.defaultSpanOfS)
        initS2 = np.median(self.defaultSpanOfS)
        return initS1, initS2

    def mcmc(self):
        s1, s2 = self.initializeMCMC()
        thetaRange = self.getThetaRange()
        s1Range = self.defaultSpanOfS
        s2Range = self.defaultSpanOfS

        thetaTbl = pd.DataFrame()
        s1Tbl = pd.DataFrame()
        s2Tbl = pd.DataFrame()
        for i in np.arange(self.mcmcIter):
            thetaProb = self.probTheta(s1, s2)
            theta = np.random.choice(thetaRange, size = 1, p = thetaProb)[0]

            s1Prob = self.probS(s1Range, [s2], theta)
            s1 = np.random.choice(s1Range, size = 1, p = s1Prob)[0]

            s2Prob = self.probS([s1], s2Range, theta)
            s2 = np.random.choice(s2Range, size = 1, p = s2Prob)[0]

            thetaTbl = pd.concat([thetaTbl, pd.DataFrame(thetaProb, columns = [i])], axis = 1)
            s1Tbl = pd.concat([s1Tbl, pd.DataFrame(s1Prob, columns = [i])], axis = 1)
            s2Tbl = pd.concat([s2Tbl, pd.DataFrame(s2Prob, columns = [i])], axis = 1)

        return thetaTbl, s1Tbl, s2Tbl

    def probThetaWorkerProcess(self, col):
        try:
            ls = self.dictOfData[col]
        except KeyError:
            print('%s: not found in data.' %col)
            return

        if len(ls) < self.minDataSize:
            print('%s: not enough data points.' %col)
            return

        mean = np.mean(ls)
        std = np.std(ls)

        # normalize
        self.y = (ls - mean) / std

        print('%s start time: %s.' %(str(col), datetime.now()))
        thetaTbl, s1Tbl, s2Tbl = self.mcmc()
        print('%s end time: %s.' %(str(col), datetime.now()))

        tmp = thetaTbl.iloc[:, -1]

        # de-normalize
        self.y = ls

        theta = np.argmax(tmp)

        if theta > 0:
            beta0 = np.mean(self.y[:theta])
        else:
            beta0 = self.y[0].tolist()[0]

        if theta < len(tmp) - 1:
            beta3 = np.mean(self.y[theta:])
        else:
            beta3 = self.y[-1].tolist()[0]

        flag = False
        chartData = pd.DataFrame()

        if beta3 * beta0 < 0:
            # flipped mean, check for manual adjustments to machine
            afterChange = np.ones(self.n - theta) * beta3
            flag = True
        elif np.abs(beta3) - np.abs(beta0) >= self.levelThres:
            # jump in level is greater than defined threshold
            afterChange = np.ones(self.n - theta) * beta3
            flag = True
        elif np.abs(beta3) - np.abs(beta0) >= 0:
            # jump in level is not greater than level threshold, however there is increase in bias
            # check for trend
            try:
                ols = lr()
                x = np.arange(self.n - theta)
                ols.fit(x.reshape(-1,1), self.y[theta:].reshape(-1,1))
                intercept = ols.intercept_[0]
                slope = ols.coef_[0][0]
                if (beta3 > 0 and slope >= self.slopeThres) or (beta3 < 0 and slope <= -self.slopeThres):
                    afterChange = np.arange(self.n - theta) * slope + intercept
                    flag = True
            except:
                pass
        else:
            afterChange = np.ones(self.n - theta) * beta3
            flag = True

        if flag:
            if self.drawChartFromPython:
                myQualityPlot(self.y, savePath = self.figureDir, title = col,
                         changePoint = theta, trendBeforeChange = np.ones(theta) * beta0,
                         trendAfterChange = afterChange,
                         secondAxis = [0] + list(tmp))

            chartData = pd.DataFrame(self.y, columns = ['sampleDataPoints'])
            chartData['changePointProbability'] = [0] + list(tmp)
            chartData['mean'] = np.hstack([np.ones(theta) * beta0, afterChange])
            chartData['index'] = col
        else:
            chartData = pd.DataFrame(self.y, columns = ['sampleDataPoints'])
            chartData['changePointProbability'] = 0
            chartData['mean'] = np.mean(self.y)
            chartData['index'] = col

        return [col, tmp, chartData]

    def probThetaParentProcess(self):
        result = self.pool.map(self.probThetaWorkerProcess, self.col)
        keys = [x[0] for x in result]
        values = [x[1] for x in result]
        chart = [x[2] for x in result]
        self.result = dict(zip(keys, values))
        self.chart = pd.concat(chart)
        self.chart = self.chart[~self.chart['index'].isnull()]

def myQualityPlot(_yplot, savePath, _x = None, LSL = [], HSL = [], title = None,
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

    plt.savefig(savePath + title + 'qualityPlot.png')
    plt.close()


#%%
if __name__ == '__main__':
    bayesClass = BayesChangeDetection()

    print('pooling: %s.' % datetime.now())
    bayesClass.probThetaParentProcess()
    bayesClass.chart.to_csv(bayesClass.dataDir + 'bayes_chart.csv')




