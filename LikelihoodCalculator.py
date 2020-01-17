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
    ##{'name': 'Moliere3D', 'pdfBins': np.linspace(0, 0.0002, num=100)},
    #{'name': 'BDTU', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTV', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDTW', 'pdfBins': np.linspace(-10, 15, num = 50)},
    #{'name': 'BDT3D', 'pdfBins': np.linspace(-10, 15, num = 50)},
    {'name': 'BDTMulti', 'pdfBins': np.linspace(-10, 15, num = 50)},
)

delta = 1e-12
viewsUsed = ds.GetViewsUsed(features)

if __name__ == "__main__":
    # Get data samples
    ds.GetTrainingPfoData(viewsUsed=viewsUsed)
    ds.GetPerfPfoData(viewsUsed=viewsUsed)

    # Calculate priors
    nTracksPrior = len(ds.dfTrainingPfoData["track"]["union"])
    nShowersPrior = len(ds.dfTrainingPfoData["shower"]["union"])
    showerPrior = nShowersPrior / (nShowersPrior + nTracksPrior)
    trackPrior = nTracksPrior / (nShowersPrior + nTracksPrior)

    print("Training likelihood using the following samples:")
    for view in ds.dfTrainingPfoData["track"]:
        print("%s: %s tracks, %s showers" % (view, len(ds.dfTrainingPfoData["track"][view]), len(ds.dfTrainingPfoData["shower"][view])))
    
    #Calculate histogram bins, obtain probabilities from them
    print("\nPriors: showers %.3f, tracks %.3f" % (showerPrior, trackPrior))
    print("Calculating likelihood values")
    probabilities = {
        "track": {},
        "shower": {}
    }
    for feature in features:
        featureView = ds.GetFeatureView(feature['name'])
        showerHist, binEdges = np.histogram(ds.dfTrainingPfoData["shower"][featureView][feature['name']], bins=feature['pdfBins'], density=True)
        trackHist, binEdges = np.histogram(ds.dfTrainingPfoData["track"][featureView][feature['name']], bins=feature['pdfBins'], density=True)
        showerHist = np.concatenate(([1], showerHist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
        showerHist[showerHist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
        trackHist[trackHist==0] = delta
        trackHist = np.concatenate(([1], trackHist, [1]))
        featureValues = ds.dfAllPfoData[feature['name']]
        histIndices = np.digitize(featureValues, feature['pdfBins'])
        if featureView not in probabilities["track"]:
            probabilities["track"][featureView] = 1
            probabilities["shower"][featureView] = 1
        probabilities["track"][featureView] *= trackHist[histIndices]
        probabilities["shower"][featureView] *= showerHist[histIndices]

    # Obtain likelihood values, using only probabilities from views that the PFO satisfied the filters.
    pfoCheck = {}
    for key in viewsUsed:
        pfoCheck[key] = ds.dfAllPfoData.eval(ds.performancePreFilters[key])
    ps = np.ones(len(ds.dfAllPfoData))
    pt = np.ones(len(ds.dfAllPfoData))
    for key in viewsUsed:
        ps *= (1 + pfoCheck[key] * (probabilities["shower"][key] - 1))
        pt *= (1 + pfoCheck[key] * (probabilities["track"][key] - 1))
    likelihoods = ps / (ps + pt)

    # Save the results
    ds.dfAllPfoData["Likelihood"] = likelihoods
    ds.SavePickleFile()
    print("Finished!")