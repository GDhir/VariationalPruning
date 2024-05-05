import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import csv
from sklearn.preprocessing import normalize
from functools import partial
import sys
import csv
from mnist import MNIST
import random

def runRVM(plotFolderName):

    sigmaVal = 0.3
    sigmaSq = sigmaVal ** 2
    betaVal = 1 / ( sigmaSq )
    N = 100
    M = 100
    xVals = np.linspace( 0, 2 * np.pi, N )
    epsVals = np.random.randn( N ) * sigmaVal    
    tVals = np.sin( xVals ) + epsVals

    basisIndexVals = np.random.randint( 0, N, M )
    completePhiDesign = np.ones( [N, M] )
    nIncluded = 0
    sigmaValBasis = 16 * sigmaSq

    for idx in range(M):
        centerPoint = basisIndexVals[idx]
        completePhiDesign[:, idx] = np.exp( -np.power( ( xVals -  xVals[ centerPoint ] ), 2 ) / 2 / sigmaValBasis )

    np.random.seed()
    nIter = 1000

    indexVals = set()

    alphaVals = np.ones(M) * np.inf
    curIndex = np.random.randint(0, M)
    curPhiDesign = np.ones( [N, 1] )

    phiVal = completePhiDesign[ :, curIndex ]
    phiNormSq = np.power( np.linalg.norm( phiVal ), 2 )
    alphaVals[ curIndex ] = phiNormSq / ( np.power( np.linalg.norm( np.dot(phiVal, tVals) ), 2 ) / phiNormSq - sigmaSq )
    indexVals.add( curIndex )

    curCov = 1 / ( alphaVals[ curIndex ] + ( np.dot( phiVal, phiVal ) * betaVal ) )
    mVal = betaVal * curCov * np.dot( phiVal, tVals )

    sVals = np.zeros( M )
    qVals = np.zeros( M )

    QVals = np.zeros( M )
    SVals = np.zeros( M )

    for idx in range(M):

        postVal = phiVal * ( curCov * np.dot( phiVal, tVals ) )

        QVals[idx] = betaVal * np.dot( completePhiDesign[ :, idx ], tVals ) - ( betaVal ** 2 ) *\
            np.dot( completePhiDesign[:, idx], postVal)

        postVal = phiVal * ( curCov * np.dot( phiVal, completePhiDesign[:, idx] ) )

        SVals[idx] = betaVal * np.dot( completePhiDesign[ :, idx ], completePhiDesign[:, idx] ) - ( betaVal ** 2 ) *\
            np.dot( completePhiDesign[:, idx], postVal)

        if idx in indexVals:
            qVals[idx] = alphaVals[idx] * QVals[idx] / ( alphaVals[idx] - SVals[idx] )
            sVals[idx] = alphaVals[idx] * SVals[idx] / ( alphaVals[idx] - SVals[idx] )

        else:
            qVals[idx] = QVals[idx]
            sVals[idx] = SVals[idx]

    for iter in range(nIter):

        curIndex = np.random.randint(0, M)

        thetaVal = qVals[curIndex] ** 2 - sVals[curIndex]

        if thetaVal > 0 and curIndex in indexVals:
            alphaVals[curIndex] = ( sVals[curIndex] ** 2 )/( qVals[curIndex] ** 2 - sVals[curIndex] )
        
        elif thetaVal > 0:
            alphaVals[curIndex] = ( sVals[curIndex] ** 2 )/( qVals[curIndex] ** 2 - sVals[curIndex] )
            indexVals.add( curIndex )

        elif curIndex in indexVals:
            indexVals.remove( curIndex )
            alphaVals[curIndex] = np.inf

        curValidIndexes = sorted( list( indexVals ) )
        curLen = len(curValidIndexes)
        A = np.zeros( [curLen, curLen] )
        curDesignMat = np.zeros( [N, curLen] )

        for idx, curValidIdx in enumerate(curValidIndexes):

            A[idx, idx] = alphaVals[curValidIdx]
            curDesignMat[ :, idx ] = completePhiDesign[ :, curValidIdx ]

        ## Update Q, S
        curCov = np.linalg.inv( A + betaVal * ( np.transpose( curDesignMat ) @ curDesignMat ) )
        meanVal = betaVal * ( curCov @ ( np.transpose( curDesignMat ) @ tVals ) )

        ## Update q, s
        alphaCovSum = 0
        curIdx = 0
        for idx in range(M):

            postVal = curDesignMat @ ( curCov @ ( np.transpose(curDesignMat) @ tVals ) )

            QVals[idx] = betaVal * np.dot( completePhiDesign[ :, idx ], tVals ) - ( betaVal ** 2 ) *\
                np.dot( completePhiDesign[:, idx], postVal)

            postVal = curDesignMat @ ( curCov @ ( np.transpose(curDesignMat) @ completePhiDesign[:, idx] ) )

            SVals[idx] = betaVal * np.dot( completePhiDesign[ :, idx ], completePhiDesign[:, idx] ) - ( betaVal ** 2 ) *\
                np.dot( completePhiDesign[:, idx], postVal)

            if idx in indexVals:
                qVals[idx] = alphaVals[idx] * QVals[idx] / ( alphaVals[idx] - SVals[idx] )
                sVals[idx] = alphaVals[idx] * SVals[idx] / ( alphaVals[idx] - SVals[idx] )

            else:
                qVals[idx] = QVals[idx]
                sVals[idx] = SVals[idx]

            if idx in indexVals:
                alphaCovSum = alphaCovSum + alphaVals[idx] * curCov[ curIdx, curIdx ]
                curIdx = curIdx + 1

        yVals = curDesignMat @ meanVal

        print( indexVals )
        sigmaSq = ( np.linalg.norm( tVals - yVals ) ** 2 ) / ( N - M + alphaCovSum )
        betaVal = 1 / sigmaSq

    yVals = curDesignMat @ meanVal

    curValidIndexes = sorted( list( indexVals ) )
    curXVals = np.zeros( len(curValidIndexes) )
    curTVals = np.zeros( len(curValidIndexes) )

    for idx, curValidIdx in enumerate(curValidIndexes):
        curXVals[ idx ] = xVals[ curValidIdx ]
        curTVals[ idx ] = tVals[ curValidIdx ]

    plt.plot( xVals, tVals, "o" )
    plt.plot( curXVals, curTVals, "x" )
    plt.plot( xVals, yVals, )

    plt.xlabel('X') 
    plt.ylabel('$sin(x) + \epsilon$') 
    # plt.title('Trajectory') 

    plt.tight_layout()
    plt.grid(linestyle='dotted')

    plotFileName = plotFolderName + "RVM_sinxGaussian.eps"
    plt.savefig( plotFileName, format = "eps" )

    plt.show()


if __name__ == "__main__":

    dataFolderName = "/home/gaurav/CS6190/Project/data/data/samples/"
    plotFolderName = "/home/gaurav/CS6190/Project/Code/Plots/"

    runRVM( plotFolderName )