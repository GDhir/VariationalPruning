digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

labelVals = imds.Labels;
dataVals = readall(imds);
numFeatures = 28 * 28;
targetSize = [numFeatures, 1];

nlabels = size(labelVals, 1);
numTrain = 0.8 * nlabels;
numTest = nlabels - numTrain;

for idx = 1:nlabels
    dataVals{idx} = reshape(dataVals{idx}, targetSize);
end

tableData = zeros(nlabels, numFeatures);
for idx = 1:nlabels
    tableData(idx, :) = dataVals{idx}';
end

idx = randperm( nlabels );
idxTrain = idx(1 : numTrain);
idxTest = idx(numTrain+1 : end);

% tableData( :, numFeatures + 1 ) = labelVals;
imdsTrain = tableData( idxTrain, : );
responseTrain = labelVals( idxTrain );
imdsTest = tableData( idxTest, : );
responseTest = labelVals( idxTest, :);

%% vals
layers = [ ...
    featureInputLayer(numFeatures,'Normalization', 'zerocenter')
    % featureInputLayer(numFeatures)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(100)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

miniBatchSize = 256;

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Plots','training-progress', ...
    'Verbose',false);

% options = trainingOptions('lbfgs', ...
%     'Plots','training-progress', ...
%     'Verbose',false);

net = trainNetwork(imdsTrain, responseTrain, layers,options);
YPred = classify(net,imdsTest);
% YTest = imdsTest(:, nlabels + 1);

accuracy = sum(YPred == responseTest)/numel(responseTest)