import numpy as np
import pandas as pd
import math as m
import DataSampler as ds
from UpRootFileReader import MicroBooneGeo

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
    ##{'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    ##{'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'BraggPeakU', 'algorithmName': 'ChargeHitBinning', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakV', 'algorithmName': 'BraggPeak','pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakW', 'algorithmName': 'BraggPeak','pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak', 'pdfBins': np.linspace(0, 1, num=100)},
    ##{'name': 'Moliere3D', 'algorithmName': 'MoliereRadius', 'pdfBins': np.linspace(0, 0.15, num=100)},
    {'name': 'BDTU', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTV', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTW', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDT3D', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTMulti', 'BDTMulti', 'algorithmName': 'DecisionTreeCalculator', 'pdfBins': np.linspace(-10, 15, num = 50)},
]

delta = 1e-12
viewsUsed = ds.GetViewsUsed(features)

if __name__ == "__main__":
    # Get data samples
    ds.GetTrainingPfoData(features)
    ds.GetPerfPfoData(features)

    # Calculate priors
    nClass0Prior = len(ds.dfTrainingPfoData["track"]["union"])
    nClass1Prior = len(ds.dfTrainingPfoData["shower"]["union"])
    class1Prior = nClass1Prior / (nClass1Prior + nClass0Prior)
    class0Prior = nClass0Prior / (nClass1Prior + nClass0Prior)

    print("Training likelihood using the following samples:")

    ds.PrintSampleInput(ds.dfTrainingPfoData)
    
    #Calculate histogram bins, obtain probabilities from them
    print("\nPriors: showers %.3f, tracks %.3f" % (class1Prior, class0Prior))
    print("Calculating likelihood values")

    def CalculateLikelihoodValues(dfTrainingPfoData, dfAllPfoData, classNames, features, performancePreFilters):
        probabilities = {
            classNames[0]: {},
            classNames[1]: {}
        }
        for feature in features:
            featureView = ds.GetFeatureView(feature['name'])
            class1Hist, binEdges = np.histogram(dfTrainingPfoData[classNames[1]][featureView][feature['name']], bins=feature['pdfBins'], density=True)
            class0Hist, binEdges = np.histogram(dfTrainingPfoData[classNames[0]][featureView][feature['name']], bins=feature['pdfBins'], density=True)
            class1Hist = np.concatenate(([1], class1Hist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
            class1Hist[class1Hist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
            class0Hist[class0Hist==0] = delta
            class0Hist = np.concatenate(([1], class0Hist, [1]))
            featureValues = dfAllPfoData[feature['name']]
            histIndices = np.digitize(featureValues, feature['pdfBins'])
            if featureView not in probabilities[classNames[0]]:
                probabilities[classNames[0]][featureView] = 1
                probabilities[classNames[1]][featureView] = 1
            probabilities[classNames[0]][featureView] *= class0Hist[histIndices]
            probabilities[classNames[1]][featureView] *= class1Hist[histIndices]

        # Obtain likelihood values, using only probabilities from views that the PFO satisfied the filters.
        pfoCheck = {}
        for key in viewsUsed:
            pfoCheck[key] = dfAllPfoData.eval(performancePreFilters[key])
        p1 = np.ones(len(dfAllPfoData))
        p0 = np.ones(len(dfAllPfoData))
        for key in viewsUsed:
            p1 *= (1 + pfoCheck[key] * (probabilities[classNames[1]][key] - 1))
            p0 *= (1 + pfoCheck[key] * (probabilities[classNames[0]][key] - 1))
        return p1 / (p1 + p0)

    # Save the results
    dfLikelihood = pd.DataFrame({"Likelihood": CalculateLikelihoodValues(ds.dfTrainingPfoData, ds.dfAllPfoData, ('track', 'shower'), features, ds.performancePreFilters)})
    ds.SavePfoData(dfLikelihood, "LikelihoodCalculator")
    print("Finished!")