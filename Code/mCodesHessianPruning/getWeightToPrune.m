%% Find weight to prune using saliency score
function [minWeightIdx, minLossInc] = getWeightToPrune( weightVals, HsInv )

szW = size(weightVals, 1);
lossIncrease = zeros(size(weightVals));

for idx = 1:szW
    lossIncrease(idx, 1) = ( weightVals(idx) ^ 2 ) / 2 / HsInv( idx, idx );
end

[minLossInc, minWeightIdx] = min( lossIncrease );

end