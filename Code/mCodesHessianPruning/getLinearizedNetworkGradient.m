function [nParams, gradientArr] = getLinearizedNetworkGradient( dlnet, oneSampleTrain, oneResponseTrain )

    [~, gradients] = dlfeval(@modelLoss, dlnet, oneSampleTrain, oneResponseTrain);
    
    gradientArr = [];
    nParams = size( gradients, 1 );
    
    for idx = 1:nParams
    
        newGrads = gradients( idx, 3 ).Value{:};
        szVal = size( newGrads );
        szVal = szVal(1) * szVal(2);
        
        gradientArr = [ gradientArr; reshape( newGrads, [szVal, 1] ) ];
    
    end

end
