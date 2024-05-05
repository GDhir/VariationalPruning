import torch
from torch import nn
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
from dataclasses import dataclass

@dataclass
class KL:
    accumulated_kl_div = 0

class MyDataset( torch.utils.data.Dataset ):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class LinearVariationalDropoutBayes(nn.Module):
    def __init__(self, in_features, out_features, parent, batch_size):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        # self.accumulated_kl_div = parent

        self.muW = nn.Parameter(torch.randn(in_features, out_features))
        self.sigmaW = nn.Parameter(torch.ones(in_features, out_features))

        self.muZ = nn.Parameter( torch.randn(in_features) )
        self.sigmaZ = nn.Parameter( torch.ones(in_features) )

    def forward(self, input, bayes = True ):

        nSamples = input.shape[0]
        device = "cuda"

        if bayes:

            epsVal = torch.randn( [nSamples, self.in_features], device = device )
            zVals = self.muZ + self.sigmaZ * epsVal
            hiddenVals = input * zVals
            hiddenMean = hiddenVals @ self.muW

            vH =  torch.pow( hiddenVals, 2 ) @ self.sigmaW

            epsVal2 = torch.randn_like( vH )

            finalVal = hiddenMean + torch.sqrt( vH ) * epsVal2

            # self.accumulated_kl_div += self.kl_divergence( self.muZ, self.sigmaZ, self.muW, self.sigmaW )

            return finalVal
        
        else:

            return input @ self.muW

        #####Uncomment for Vanilla NN SGD
        # weight = self.mu

        ######## Bayes NN with Gaussian Prior
        # weight = self.mu + epsVal * torch.sqrt( torch.log( 1 + torch.exp( self.rho ) ) )

        # return input @ weight
    
    def forwardWithPosteriorMean( self, input ):

        return (input @ self.mu)
    
    def kl_divergence(self):

        k1 = 0.63576
        k2 = 1.87320
        k3 = 1.48695

        logalphaI = torch.log( torch.pow( self.sigmaZ, 2) ) - torch.log( torch.pow( self.muZ, 2) )
        sfplus = torch.nn.Softplus()

        klDiv1 = -( k1 * torch.sigmoid( k2 + k3 * logalphaI ) - 0.5 * ( sfplus( -logalphaI ) ) - k1 )

        klDiv1 = klDiv1.sum()

        klDiv2 = 0.5 * ( -torch.log( self.sigmaW ) + self.sigmaW + torch.pow( self.muW, 2 ) - 1 )

        klDiv2 = klDiv2.sum()

        return klDiv1 + klDiv2

class BayesNet(nn.Module):

    def __init__(self, num_hidden_layers, numNodes, numFeatures, batchSize = 128):

        super().__init__()
        self.accumulated_kl_div = 0

        self.inputLayer = LinearVariationalDropoutBayes( numFeatures, numNodes, self, batchSize)

        self.linears = nn.ModuleList(
            [LinearVariationalDropoutBayes( numNodes, numNodes, self, batchSize ) for _ in range(num_hidden_layers)])
        
        self.inputBatchNorm = nn.BatchNorm1d(numNodes)

        self.batchNormLayers = nn.ModuleList(
            [nn.BatchNorm1d(numNodes) for _ in range(num_hidden_layers)]
        )

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        })

        self.final = LinearVariationalDropoutBayes( numNodes, 10, self.accumulated_kl_div, batchSize)
        self.finalSoftmax = nn.Softmax(1)

    def forward( self, x, act, bayes = True ):

        if bayes:
            x = self.inputLayer(x, bayes)
            self.accumulated_kl_div = self.accumulated_kl_div + self.inputLayer.kl_divergence()

            x = self.activations[act](x)
            # x = self.inputBatchNorm(x)

            for idx, linear in enumerate(self.linears):
                x = linear(x, bayes)
                self.accumulated_kl_div = self.accumulated_kl_div + linear.kl_divergence()

                x = self.activations[act](x)
                # x = self.batchNormLayers[idx](x)

            x = self.final(x, bayes)
            self.accumulated_kl_div = self.accumulated_kl_div + self.final.kl_divergence()
            x = self.finalSoftmax(x)

        else:
            x = self.inputLayer(x, bayes)
            x = self.activations[act](x)
            # x = self.inputBatchNorm(x)

            for idx, linear in enumerate(self.linears):
                x = linear(x, bayes)
                x = self.activations[act](x)
                # x = self.batchNormLayers[idx](x)

            x = self.final(x, bayes)
            x = self.finalSoftmax(x)

        return x
    
    def forwardWithPosteriorMean( self, x, act ):

        x = self.inputLayer.forwardWithPosteriorMean(x)
        x = self.activations[act](x)

        for idx, linear in enumerate(self.linears):
            x = linear.forwardWithPosteriorMean(x)
            x = self.batchNormLayers[idx](x)
            x = self.activations[act](x)

        x = self.final.forwardWithPosteriorMean(x)
        x = self.finalSoftmax(x)

        return x
    
    # @property
    # def accumulated_kl_div(self):
    #     return self.kl_loss.accumulated_kl_div
    
    def reset_kl_div(self):
        self.accumulated_kl_div = 0

def computeLogLikelihood( pred, labels ):

    criterion = torch.nn.CrossEntropyLoss( reduction = "mean" )
    loss = -criterion( pred, labels )

    return loss

def computePredAccuracy( net, featureVecs, labels, act, bayes = True ):

    nSamples = featureVecs.shape[0]
    nIterations = 10
    acc = 0

    for idx in range(nIterations):

        curPred = net.forward( featureVecs, act, bayes )
        # curPred = torch.softmax( curPred, 1 )

        maxTensor = torch.max( curPred, 1 )
        maxClass = maxTensor.indices

        for nIdx in range(nSamples):

            acc = acc + int( labels[nIdx] == maxClass[nIdx] )

    return acc / nIterations / nSamples

def computePredLikelihood( net, featureVecs, labels, act, bayes = True ):

    nIterations = 10
    likelihood = 0

    for idx in range(nIterations):

        curPred = net.forward( featureVecs, act, bayes )
        lossFunc = torch.nn.CrossEntropyLoss( reduction = "mean" )
        likelihood = likelihood - lossFunc( curPred, labels )

    return likelihood / nIterations

def customLoss( net, pred, labels, nBatches, bayes = True ):

    # Loss for Log Uniform Prior with Bayes NN
    
    if bayes:
        kl = net.accumulated_kl_div
        net.reset_kl_div()

        criterion = torch.nn.CrossEntropyLoss( reduction = "sum" )
        loss = nBatches * criterion( pred, labels ) + kl

        loss = loss + 0.5 * torch.sum( net.inputLayer.sigmaZ - 0.5 ) +\
        0.5 * torch.sum( net.inputLayer.sigmaW - 0.5 )
    
        for layerVal in net.linears:

            loss = loss + 0.5 * torch.sum( layerVal.sigmaZ - 0.5 ) +\
            0.5 * torch.sum( layerVal.sigmaW - 0.5 )

        loss = loss + 0.5 * torch.sum( net.final.sigmaZ - 0.5 ) +\
            0.5 * torch.sum( net.final.sigmaW - 0.5 )
    

    else:
    ####### Vanilla NN SGD
        criterion = torch.nn.CrossEntropyLoss( reduction = "sum" )
        loss = nBatches * criterion( pred, labels )

    return loss

def trainLoop( featureVecs, labels, net, lossFcn, optimizer, act, nBatches, bayes = True ):

    nTrain = featureVecs.shape[0]
    nFeatures = featureVecs.shape[1]

    pred = net( featureVecs, act, bayes )

    loss = lossFcn( net, pred, labels, nBatches, bayes )
    # criterion = torch.nn.BCEWithLogitsLoss( reduction = "sum" )
    # loss = criterion( pred[ :, 0 ], labels )
    # print( net.inputLayer.mu )
    # print( torch.mean( torch.abs( torch.sigmoid( pred[:,0] ) - labels ) ) )
    # print(loss)

    # net.zero_grad()
    loss.backward()
    # print(net.inputLayer.mu)
    optimizer.step()
    optimizer.zero_grad()

def plotHistories( plotFolderName, trAveLikeVals, teAveLikeVals, teAccVals, lr, numNodes, actVal, filenamePrefix = "" ):

    plt.figure() 
 
    # Plot the trajectory 
    plt.plot( trAveLikeVals, "-o" ) 

    plt.xlabel('Epoch') 
    plt.ylabel('Average Training Likelihood') 
    # plt.title('Trajectory') 

    plt.tight_layout()
    plt.grid(linestyle='dotted')

    plotFileName = plotFolderName + filenamePrefix + "VDropout_TrainingAverageLikelihood_act=" + actVal +\
        "_numNodes=" + str(numNodes) + "_lr=" + str(lr) + ".eps"
    plt.savefig( plotFileName, format = "eps" )

    plt.figure() 
 
    # Plot the trajectory 
    plt.plot( teAveLikeVals, "-o" ) 

    plt.xlabel('Epoch') 
    plt.ylabel('Average Test Likelihood') 
    # plt.title('Trajectory') 

    plt.tight_layout()
    plt.grid(linestyle='dotted')

    plotFileName = plotFolderName + filenamePrefix + "VDropout_TestAverageLikelihood_act=" + actVal +\
        "_numNodes=" + str(numNodes) + "_lr=" + str(lr) + ".eps"
    plt.savefig( plotFileName, format = "eps" )

    plt.figure() 
 
    # Plot the trajectory 
    plt.plot( teAccVals, "-o" ) 

    plt.xlabel('Epoch') 
    plt.ylabel('Average Test Error') 
    # plt.title('Trajectory') 

    plt.tight_layout()
    plt.grid(linestyle='dotted')

    plotFileName = plotFolderName + filenamePrefix + "VDropout_AverageTestError_act=" + actVal +\
        "_numNodes=" + str(numNodes) + "_lr=" + str(lr) + ".eps"
    plt.savefig( plotFileName, format = "eps" )

    plt.show()

def testVanillaNNPostTuningCUDA( trainFeatureVecs, trainLabels, testFeatureVecs, testLabels, dataFolderName ):

    # Train on the complete training data containing 60k images post hyperparameter tuning


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    nSamples = trainFeatureVecs.shape[0]
    nFeatures = trainFeatureVecs.shape[1]
    batchSize = 128
    nBatches = int( nSamples / batchSize )
    nTest = testFeatureVecs.shape[0]

    trainFeatureVecs = torch.tensor( trainFeatureVecs, dtype = torch.float32, device = device )
    trainLabels = torch.tensor( trainLabels, dtype = torch.long, device = device )

    testFeatureVecs = torch.tensor( testFeatureVecs, dtype = torch.float32, device = device )
    testLabels = torch.tensor( testLabels, dtype = torch.float32, device = device )

    traindata = MyDataset(trainFeatureVecs, trainLabels)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size = batchSize, shuffle = True )

    # Create the network (from previous section) and optimizer
    numHiddenLayers = 2

    actVals = ["relu"]
    numNodeVals = [400]

    dictVals = dict()

    epochs = 1000
    lr = 1e-5
    trAveLikeVals = np.zeros( [epochs] )
    teAveLikeVals = np.zeros( [epochs] )
    teAccVals = np.zeros( [epochs] )

    for numNodes in numNodeVals:
        for act in actVals:

            net = BayesNet( numHiddenLayers, numNodes, nFeatures )
            net.to( device )
            optimizer = torch.optim.Adam( net.parameters(), lr = lr)
            lossFcn = customLoss

            # Run a sample training loop that "teaches" the network
            # to output the constant zero function

            for epoch in range(epochs):

                for i, data in enumerate(trainloader, 0):

                    featureBatch, labelBatch = data    

                    trainLoop( featureBatch,\
                            labelBatch, net, lossFcn,\
                            optimizer, act, nBatches )


                # trainPredMean = net.forwardWithPosteriorMean( trainFeatureVecs, act )

                # trainAveLikelihood = computeLogLikelihood( trainPredMean,\
                                    # trainLabels )

                # testPredMean = net.forwardWithPosteriorMean( testFeatureVecs, act )
                # testAveLikelihood = computeLogLikelihood( testPredMean, testLabels )
                testAveAccuracy = computePredAccuracy( net, testFeatureVecs, testLabels, act )

                testAveError = ( 1 - testAveAccuracy ) * 100

                # trAveLikeVals[epoch] = trainAveLikelihood
                # teAveLikeVals[epoch] = testAveLikelihood
                teAccVals[epoch] = testAveError

                print( epoch, testAveError )

            keyVal = "numNodes= " + str(numNodes) + " activation= " + act + " batchSize= " + str(batchSize)

            # trainAcc = computePredAccuracy( net, trainFeatureVecs, trainLabels, act )
            # print( trainAcc )
            trainAcc = 0

            testAcc = computePredAccuracy( net, testFeatureVecs, testLabels, act )
            print( testAcc )

            valueStr = "TrainAccuracy= " + str(trainAcc) + " TestAcc= " + str(testAcc) 
            dictVals[keyVal] = valueStr

            del net

            filenamePrefix = "PostTuningPlot_VanillaNN_PR4c"
            plotHistories( plotFolderName, trAveLikeVals, teAveLikeVals, teAccVals, lr, numNodes, act, filenamePrefix )

    textFileName = dataFolderName + "PostTuningVanillaNN_PR4c_DictOutput.csv"

    with open( textFileName, "a" ) as textHandle:

        writerHandle = csv.writer( textHandle )

        # loop over dictionary keys and values
        for key, val in dictVals.items():

            # write every key and value to file
            writerHandle.writerow([key, val])


def testBayesNNPostTuningCUDA( trainFeatureVecs, trainLabels, testFeatureVecs, testLabels, dataFolderName ):

    # Train on the complete training data containing 60k images post hyperparameter tuning


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    nSamples = trainFeatureVecs.shape[0]
    nFeatures = trainFeatureVecs.shape[1]
    batchSize = 128
    nBatches = int( nSamples / batchSize )
    nTest = testFeatureVecs.shape[0]

    trainFeatureVecs = torch.tensor( trainFeatureVecs, dtype = torch.float32, device = device )
    trainLabels = torch.tensor( trainLabels, dtype = torch.long, device = device )

    testFeatureVecs = torch.tensor( testFeatureVecs, dtype = torch.float32, device = device )
    testLabels = torch.tensor( testLabels, dtype = torch.float32, device = device )

    traindata = MyDataset(trainFeatureVecs, trainLabels)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size = batchSize, shuffle = True )

    # Create the network (from previous section) and optimizer
    numHiddenLayers = 2

    actVals = ["relu"]
    numNodeVals = [400]

    dictVals = dict()

    epochs = 1000
    lr = 1e-4
    trAveLikeVals = np.zeros( [epochs] )
    teAveLikeVals = np.zeros( [epochs] )
    teAccVals = np.zeros( [epochs] )

    for numNodes in numNodeVals:
        for act in actVals:

            net = BayesNet( numHiddenLayers, numNodes, nFeatures, batchSize )
            net.to( device )
            optimizer = torch.optim.Adam( net.parameters(), lr = lr )
            lossFcn = customLoss

            # Run a sample training loop that "teaches" the network
            # to output the constant zero function

            for epoch in range(epochs):

                if epoch < 100:
                    bayes = False
                else:
                    bayes = True

                for i, data in enumerate(trainloader, 0):

                    featureBatch, labelBatch = data    

                    trainLoop( featureBatch,\
                            labelBatch, net, lossFcn,\
                            optimizer, act, nBatches, bayes )

                # trainPredMean = net.forwardWithPosteriorMean( trainFeatureVecs, act )

                # trainAveLikelihood = computeLogLikelihood( trainPredMean,\
                                    # trainLabels )

                # testPredMean = net.forwardWithPosteriorMean( testFeatureVecs, act )
                # testAveLikelihood = computeLogLikelihood( testPredMean, testLabels )
                testAveAccuracy = computePredAccuracy( net, testFeatureVecs, testLabels, act, bayes )

                testAveError = ( 1 - testAveAccuracy ) * 100

                # trAveLikeVals[epoch] = trainAveLikelihood
                # teAveLikeVals[epoch] = testAveLikelihood
                teAccVals[epoch] = testAveError

                print( epoch, testAveError )

            keyVal = "numNodes= " + str(numNodes) + " activation= " + act + " batchSize= " + str(batchSize)

            # trainAcc = computePredAccuracy( net, trainFeatureVecs, trainLabels, act )
            # print( trainAcc )
            trainAcc = 0

            testAcc = computePredAccuracy( net, testFeatureVecs, testLabels, act, bayes )
            print( testAcc )

            valueStr = "TrainAccuracy= " + str(trainAcc) + " TestAcc= " + str(testAcc) 
            dictVals[keyVal] = valueStr

            del net

            filenamePrefix = "PostTuningPlot"
            plotHistories( plotFolderName, trAveLikeVals, teAveLikeVals, teAccVals, lr, numNodes, act, filenamePrefix )

    textFileName = dataFolderName + "PostTuning_VDropout_DictOutput.csv"

    with open( textFileName, "a" ) as textHandle:

        writerHandle = csv.writer( textHandle )

        # loop over dictionary keys and values
        for key, val in dictVals.items():

            # write every key and value to file
            writerHandle.writerow([key, val])


def tuneHyperParametersCUDA( allFeatureVecs, allLabels, dataFolderName ):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    nSamples = allFeatureVecs.shape[0]
    nFeatures = allFeatureVecs.shape[1]
    nTrain = int( 50000 )
    batchSize = 256
    nBatches = int( nTrain / batchSize )
    nTest = nSamples - nTrain

    allFeatureVecs = torch.tensor( allFeatureVecs, dtype = torch.float32, device = device )
    allLabels = torch.tensor( allLabels, dtype = torch.long, device = device )

    # finalTestFeatureVecs = torch.tensor( finalTestFeatureVecs, dtype = torch.float32, device = device )
    # finalTestLabels = torch.tensor( finalTestLabels, dtype = torch.float32, device = device )

    allIdxVals = torch.randint( 0, nSamples, (nSamples,) )
    trainIdxVals = allIdxVals[ 0:nTrain ]
    testIdxVals = allIdxVals[ nTrain:nSamples ]

    trainFeatureVecs = allFeatureVecs[ trainIdxVals, : ]
    trainLabels = allLabels[ trainIdxVals ]

    testFeatureVecs = allFeatureVecs[ testIdxVals, : ]
    testLabels = allLabels[ testIdxVals ]

    traindata = MyDataset(trainFeatureVecs, trainLabels)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size = batchSize, shuffle = True )

    # Create the network (from previous section) and optimizer
    numHiddenLayers = 2

    actVals = ["relu", "tanh"]
    numNodeVals = [400, 800, 1200]

    dictVals = dict()

    epochs = 1000
    lr = 1e-3
    trAveLikeVals = np.zeros( [epochs] )
    teAveLikeVals = np.zeros( [epochs] )
    teAccVals = np.zeros( [epochs] )

    for numNodes in numNodeVals:
        for act in actVals:

            net = BayesNet( numHiddenLayers, numNodes, nFeatures )
            net.to( device )
            optimizer = torch.optim.Adam( net.parameters(), lr = lr )
            lossFcn = customLoss

            # Run a sample training loop that "teaches" the network
            # to output the constant zero function

            for epoch in range(epochs):

                for i, data in enumerate(trainloader, 0):

                    featureBatch, labelBatch = data    

                    trainLoop( featureBatch,\
                            labelBatch, net, lossFcn,\
                            optimizer, act, nBatches )


                # trainPredMean = net.forwardWithPosteriorMean( trainFeatureVecs, act )

                # trainAveLikelihood = computeLogLikelihood( trainPredMean,\
                                    # trainLabels )

                # testPredMean = net.forwardWithPosteriorMean( testFeatureVecs, act )
                # testAveLikelihood = computeLogLikelihood( testPredMean, testLabels )
                testAveAccuracy = computePredAccuracy( net, testFeatureVecs, testLabels, act )

                testAveError = ( 1 - testAveAccuracy ) * 100

                # trAveLikeVals[epoch] = trainAveLikelihood
                # teAveLikeVals[epoch] = testAveLikelihood
                teAccVals[epoch] = testAveError

                print( epoch, testAveError )

            keyVal = "numNodes= " + str(numNodes) + " activation= " + act

            # trainAcc = computePredAccuracy( net, trainFeatureVecs, trainLabels, act )
            # print( trainAcc )
            trainAcc = 0

            testAcc = computePredAccuracy( net, testFeatureVecs, testLabels, act )
            print( testAcc )

            valueStr = "TrainAccuracy= " + str(trainAcc) + " TestAcc= " + str(testAcc) 
            dictVals[keyVal] = valueStr

            del net

            plotHistories( plotFolderName, trAveLikeVals, teAveLikeVals, teAccVals, lr, numNodes, act )

    textFileName = dataFolderName + "PR4a_DictOutput.csv"

    with open( textFileName, "a" ) as textHandle:

        writerHandle = csv.writer( textHandle )

        # loop over dictionary keys and values
        for key, val in dictVals.items():

            # write every key and value to file
            writerHandle.writerow([key, val])

    
if __name__ == "__main__":


    mndata = MNIST( dataFolderName )

    trainFeatureVec, trainLabels = mndata.load_training()
    testFeatureVec, testLabels = mndata.load_testing()

    trainFeatureVec = np.array( trainFeatureVec )
    trainFeatureVec = trainFeatureVec / 255
    trainLabels = np.array( trainLabels )

    testFeatureVec = np.array( testFeatureVec )
    testFeatureVec = testFeatureVec / 255
    testLabels = np.array( testLabels )

    # tuneHyperParametersCUDA( trainFeatureVec, trainLabels, textFolderName )
    testBayesNNPostTuningCUDA( trainFeatureVec, trainLabels, testFeatureVec, testLabels, textFolderName )

    # testVanillaNNPostTuningCUDA( trainFeatureVec, trainLabels, testFeatureVec, testLabels, dataFolderName )

    # # Plot Image
    # index = random.randrange(0, len(images))  # choose an index ;-)
    
    # image = np.asarray(images[20]).squeeze()
    # image = np.reshape( image, [28, 28] )
    # plt.imshow(image)
    # plt.show()
    # print(mndata.display(images[index]))