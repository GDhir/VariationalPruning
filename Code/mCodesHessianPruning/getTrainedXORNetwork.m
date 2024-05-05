function [net, inputTrain, responseTrain] = getTrainedXORNetwork()

%% Specify inputs and outputs
inputTrain = [0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1];
% responseTrain = categorical( [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]' ); % desired output
responseTrain = [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]';
% inputVals = [arg1; arg2]';
% output = categorical( [0, 1, 1, 0]' );
numFeatures = 2;

%% Layer Specification and Training
layers = [ ...
    featureInputLayer(numFeatures)
    fullyConnectedLayer(4)
    reluLayer
    % sigmoidLayer
    fullyConnectedLayer(2)
    reluLayer
    fullyConnectedLayer(1)
    % sigmoidLayer
    % softmaxLayer
    % classificationLayer
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'Plots','training-progress', ...
    'Verbose',false);

options.MaxEpochs = 2000;

net = trainNetwork(inputTrain, responseTrain, layers, options);

x0 = 0;
y0 = 0;
width = 6;
height = 5;
currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
% currentfig.Position = [x0 y0 width height];
currentfig.Units = 'inches';
currentfig.PaperPositionMode = 'auto';
% print(currentfig, 'xorAcc.eps', '-depsc2');
savefig(currentfig,'test.fig');

end