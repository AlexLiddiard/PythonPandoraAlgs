############################################## BDT CALCULATOR CONFIGURATION ##################################################

features = [
    {'name': 'RSquaredU', 'algorithmName': 'LinearRegression'},#
    {'name': 'RSquaredV', 'algorithmName': 'LinearRegression'},#
    {'name': 'RSquaredW', 'algorithmName': 'LinearRegression'},#
    {'name': 'BinnedHitStdU', 'algorithmName': 'HitBinning'},#
    {'name': 'BinnedHitStdV', 'algorithmName': 'HitBinning'},#
    {'name': 'BinnedHitStdW', 'algorithmName': 'HitBinning'},#
    {'name': 'RadialBinStdU', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStdV', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStdW', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStd3D', 'algorithmName': 'HitBinning'},
    {'name': 'ChainCountU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainCountV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainCountW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioStdU', 'algorithmName': 'ChainCreation'},#
    {'name': 'ChainRatioStdV', 'algorithmName': 'ChainCreation'},#
    {'name': 'ChainRatioStdW', 'algorithmName': 'ChainCreation'},#
    {'name': 'ChainRSquaredStdU', 'algorithmName': 'ChainCreation'},#
    {'name': 'ChainRSquaredStdV', 'algorithmName': 'ChainCreation'},#
    {'name': 'ChainRSquaredStdW', 'algorithmName': 'ChainCreation'},#
    {'name': 'AngularSpanU', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpanV', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpanW', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpan3D', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan'},
    {'name': 'PcaVarU', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaVarV', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaVarW', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis'},#
    {'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis'},#
    {'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'BraggPeakU', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakV', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakW', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak'},
    {'name': 'Moliere3D', 'algorithmName': 'MoliereRadius'},#
    #{'name': 'minCoordX3D', 'algorithmName': 'GeneralInfo'},
    #{'name': 'minCoordY3D', 'algorithmName': 'GeneralInfo'},
    #{'name': 'minCoordZ3D', 'algorithmName': 'GeneralInfo'},
    #{'name': 'maxCoordX3D', 'algorithmName': 'GeneralInfo'},
    #{'name': 'maxCoordY3D', 'algorithmName': 'GeneralInfo'},
    #{'name': 'maxCoordZ3D', 'algorithmName': 'GeneralInfo'},
]

balanceClasses = False
calculateBDTAll = False
calculateFeatureImportances = False