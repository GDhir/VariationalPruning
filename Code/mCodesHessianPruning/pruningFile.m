clc; clear;

%% Get fully trained network and compute prediction accuracy
[net, inputTrain, responseTrain] = getTrainedXORNetwork();
YPred = predict(net, inputTrain);
predictedVals = getOutputFromRegression( YPred );
% YTest = imdsTest(:, nlabels + 1);
accuracy = sum(predictedVals == responseTrain) / numel(responseTrain);
disp( accuracy )

%% Get DLNet representation, dlarray for response and training data
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, 'regressionoutput');
dlnet = dlnetwork(lgraph);

% One hot encoding for classification tasks
% finalResponseTrain = dlarray( double( onehotencode( responseTrain, 2 ) ) )';

% Regression Task exact output
finalResponseTrain = dlarray( double( responseTrain ) )';

dlDataTrain =  dlarray( double( inputTrain ), "BC" );
nSamples = size( dlDataTrain, 2 );

oneSampleTrain = dlDataTrain( :, 1 );
oneResponseTrain = finalResponseTrain( :, 1 );

%% Get Network Gradient and Hessian/Hessian Inverse
[nParams, gradientArr] = getLinearizedNetworkGradient( dlnet, oneSampleTrain, oneResponseTrain );

gradientSz = size( gradientArr, 1 );
alpha = 1e-4;

% Hessian Inverse
HsInv = eye( gradientSz) / alpha;

% Hessian
% HsVal = eye( gradientSz) * alpha;

for idx = 1:nSamples

    oneSampleTrain = dlDataTrain( :, idx );
    oneResponseTrain = finalResponseTrain( :, idx );
    
    [~, gradientArr] = getLinearizedNetworkGradient( dlnet, oneSampleTrain, oneResponseTrain );

    % Calculation of Hessian
    % HsVal = HsVal + ( gradientArr * gradientArr' ) / nSamples;

    % Direct Calculation of Hessian Inverse using Expectation Approximation
    HsInv = HsInv - ( HsInv * gradientArr * gradientArr' * HsInv ) /...
        ( nSamples + gradientArr' * HsInv * gradientArr );

end

%% Find weight to prune using saliency score
weightVals = getLinearizedNetworkWeights( dlnet, nParams );

% Direct Calculation of Inverse using Hessian
% HsInv2 = inv( extractdata( HsVal ) );

[minWeightIdx, minLossInc] = getWeightToPrune( weightVals, HsInv );

disp(minWeightIdx)
disp(minLossInc)

