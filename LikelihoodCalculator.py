import numpy as np
import pandas as pd
import math as m
import DataSampler as ds
from UpRootFileReader import MicroBooneGeo

features = (
    #{'name': 'RSquaredU', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredV', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredW', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'BinnedHitStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdW', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdW', 'pdfBins': np.linspace(0, 12, num=50),},
    #{'name': 'RadialBinStd3D', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'ChainCountU', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountV', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountW', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainRatioAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    ##{'name': 'ChainRSquaredAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    ##{'name': 'ChainRSquaredStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'AngularSpanU', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanV', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanW', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpan3D', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'LongitudinalSpanU', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanV', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanW', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'LongitudinalSpan3D', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'PcaVarU', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarV', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarW', 'pdfBins': np.linspace(0, 10, num=50)},
    ##{'name': 'PcaVar3D', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaRatioU', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioV', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioW', 'pdfBins': np.linspace(0, 0.4, num=50)},
    ##{'name': 'PcaRatio3D', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'ChargedBinnedHitStdU', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdV', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdW', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedStdMeanRatioU', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioV', 'pdfBins': np.linspace(0, 4, num=100)},
    ##{'name': 'ChargedStdMeanRatioW', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'BraggPeakU', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakV', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakW', 'pdfBins': np.linspace(0, 1, num=100)},
    ##{'name': 'BraggPeak3D', 'pdfBins': np.linspace(0, 1, num=100)},
    ##{'name': 'Moliere3D', 'pdfBins': np.linspace(0, 0.15, num=100)},
    {'name': 'BDTU', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTV', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTW', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDT3D', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTMulti', 'pdfBins': np.linspace(-10, 15, num = 50)},
)

delta = 1e-12
viewsUsed = ds.GetViewsUsed(features)

if __name__ == "__main__":
    # Get data samples
    ds.GetTrainingPfoData(viewsUsed=viewsUsed)
    ds.GetPerfPfoData(viewsUsed=viewsUsed)

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
    ds.dfAllPfoData["Likelihood"] = CalculateLikelihoodValues(ds.dfTrainingPfoData, ds.dfAllPfoData, ('track', 'shower'), features, ds.performancePreFilters)
    ds.SavePickleFile()
    print("Finished!")