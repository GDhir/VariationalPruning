function weightVals = getLinearizedNetworkWeights( dlnet, nParams )

weightVals = [];
for paramIdx = 1:nParams
    
    weights = dlnet.Learnables( paramIdx, 3 ).Value{:};
    szVal = size( weights );
    szVal = szVal(1) * szVal(2);
    weightVals = [ weightVals; reshape( weights, [szVal, 1] ) ];

end

end