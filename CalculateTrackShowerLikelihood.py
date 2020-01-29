import pandas as pd
import numpy as np
import math as m
import DataSampler as ds
import LikelihoodCalculator as lc

features = [
    #{'name': 'RSquaredU', 'algorithmName': 'LinearRegression', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredV', 'algorithmName': 'LinearRegression', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredW', 'algorithmName': 'LinearRegression', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'BinnedHitStdU', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdV', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdW', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdU', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdV', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdW', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50),},
    #{'name': 'RadialBinStd3D', 'algorithmName': 'HitBinning', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'ChainCountU', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountV', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountW', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainRatioAvgU', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioAvgV', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioAvgW', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgU', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgV', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgW', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioStdU', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdV', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdW', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdU', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdV', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdW', 'algorithmName': 'ChainCreation', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'AngularSpanU', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanV', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanW', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpan3D', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'PcaVarU', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarV', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarW', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargedHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargedHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargedHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeStdMeanRatio', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeStdMeanRatio', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeStdMeanRatio', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargeStdMeanRatio', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'BraggPeakU', 'algorithmName': 'BraggPeak', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakV', 'algorithmName': 'BraggPeak','pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakW', 'algorithmName': 'BraggPeak','pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'Moliere3D', 'algorithmName': 'MoliereRadius', 'pdfBins': np.linspace(0, 0.15, num=100)},
    #{'name': 'BDTU', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTV', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTW', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDT3D', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTMulti', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
]

if __name__ == "__main__":
    # Get data samples
    ds.GetTrainingPfoData(classes=ds.classes, features=features)
    ds.GetPerfPfoData(classes=ds.classes, features=features)
    print("Training likelihood using the following samples:")
    ds.PrintSampleInput(ds.dfTrainingPfoData)
    
    # Calculate likelihood values
    print("Calculating likelihood values")
    likelihoodValues = lc.CalculateLikelihoodValues(ds.dfTrainingPfoData["track"], ds.dfTrainingPfoData["shower"], ds.dfInputPfoData["all"], ('track', 'shower'), features, ds.performancePreFilters)
    dfLikelihood = pd.DataFrame({"Likelihood": likelihoodValues})

    # Save the results
    ds.SavePfoData(dfLikelihood, "LikelihoodCalculator")
    print("Finished!")