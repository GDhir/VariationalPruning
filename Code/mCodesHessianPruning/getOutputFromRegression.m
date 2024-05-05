function actualVals = getOutputFromRegression( ypred )

szPred = size( ypred, 1 );
actualVals = zeros( szPred, 1 );

for idx = 1:szPred

    if( ypred( idx ) >= 0.5 )
        actualVals( idx ) = 1;
    else
        actualVals( idx ) = 0;
    end

end

end