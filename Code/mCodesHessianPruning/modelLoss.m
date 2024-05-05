function [lossVal, gradients, state] = modelLoss(net,X,T)

% Forward data through network.
[Y,state] = forward(net, X);

% Calculate cross-entropy loss.
% loss = crossentropy(Y,T);

% Calculate regression loss
lossVal = mse( Y, T );

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(dlarray( lossVal ), net.Learnables);

end