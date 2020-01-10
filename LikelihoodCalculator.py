import numpy as np
import pandas as pd
import math as m
from PfoGraphicalAnalyser import MicroBooneGeo

myTestArea = "/home/jack/Documents/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'

trainingFraction = 0.5
trainingPreFilters = (
    'purityU>=0.8',
    'purityV>=0.8',
    'purityW>=0.8',
    'completenessU>=0.8',
    'completenessV>=0.8',
    'completenessW>=0.8',
    #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
    #'nHitsU + nHitsV + nHitsW >= 100',
    'nHitsU>=50',
    'nHitsV>=50',
    'nHitsW>=50',
    'absPdgCode != 2112',
    'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
    'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
    'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
    'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
    'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
    'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
)

featurePdfs = (
    #{'name': 'RSquaredU', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredV', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredW', 'bins': np.linspace(0, 1, num=50)},
    {'name': 'BinnedHitStdU', 'bins': np.linspace(0, 12, num=50)},
    {'name': 'BinnedHitStdV', 'bins': np.linspace(0, 12, num=50)},
    {'name': 'BinnedHitStdW', 'bins': np.linspace(0, 12, num=50)},
    #{'name': 'ChainCountU', 'bins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountV', 'bins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountW', 'bins': np.linspace(1, 50, num=50)},
    {'name': 'ChainRatioAvgU', 'bins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgV', 'bins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgW', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgU', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgV', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgW', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioStdU', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdV', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdW', 'bins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdU', 'bins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdV', 'bins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdW', 'bins': np.linspace(0, 0.8, num=50)},
    {'name': 'AngularSpanU', 'bins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpanV', 'bins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpanW', 'bins': np.linspace(0, m.pi, num=50)},
    #{'name': 'LongitudinalSpanU', 'bins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanV', 'bins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanW', 'bins': np.linspace(0, 600, num=50)},
)

delta = 1e-12

'''Separate true tracks from true showers. Then plot histograms for feature
values. Convert these histograms in to PDFs using scipy. Plot overlapping
histograms for track and shower types.'''
# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
nPfoData = len(dfPfoData)

# Get training PFOs.
dfTrainingPfoData = dfPfoData[:m.floor(nPfoData * trainingFraction)]
# Apply training pre-filters.
dfTrainingPfoData = dfTrainingPfoData.query(' and '.join(trainingPreFilters))
dfTrainingShowerData = dfTrainingPfoData.query("isShower==1")
dfTrainingTrackData = dfTrainingPfoData.query("isShower==0")
nTrainingPfoData = len(dfTrainingPfoData)
nTrainingShowerData = len(dfTrainingShowerData)
nTrainingTrackData = len(dfTrainingTrackData)
print("Training likelihood using %d tracks and %d showers." % (nTrainingTrackData, nTrainingShowerData))


# Calculate histogram bins, obtain likelihood from them
showerPrior = nTrainingShowerData / nTrainingPfoData
print("Priors: showers %.3f, tracks %.3f" % (showerPrior, 1 - showerPrior))
ptArray = np.repeat(1 - showerPrior, nPfoData)
psArray = np.repeat(showerPrior, nPfoData)
for feature in featurePdfs:
    showerHist, binEdges = np.histogram(dfTrainingShowerData[feature['name']], bins=feature['bins'], density=True)
    trackHist, binEdges = np.histogram(dfTrainingTrackData[feature['name']], bins=feature['bins'], density=True)
    showerHist = np.concatenate(([1], showerHist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
    showerHist[showerHist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
    trackHist[trackHist==0] = delta
    trackHist = np.concatenate(([1], trackHist, [1]))
    featureValues = dfPfoData[feature['name']]
    histIndices = np.digitize(featureValues, feature['bins'])
    ptArray *= trackHist[histIndices]
    psArray *= showerHist[histIndices]
likelihoodArray = psArray / (ptArray + psArray)
dfPfoData["Likelihood"] = likelihoodArray
dfPfoData.to_pickle(outputPickleFile)
print("Finished!")