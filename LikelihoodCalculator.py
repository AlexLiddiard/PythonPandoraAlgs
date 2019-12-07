import numpy as np
import pandas as pd
import math as m
from PfoGraphAnalysis import MicroBooneGeo

myTestArea = "/home/alexliddiard/Desktop/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.bz2'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp(Processed).bz2'

trainingFraction = 0.5
trainingPreFilters = ('purityU>=0.8',
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

featurePdfs = (#{'name': 'F0aU', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F0aV', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F0aW', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F1aU', 'bins': np.linspace(0, 6, num=50)},
               {'name': 'F1aV', 'bins': np.linspace(0, 6, num=50)},
               {'name': 'F1aW', 'bins': np.linspace(0, 6, num=50)},
               #{'name': 'F2aU', 'bins': np.linspace(1, 50, num=50)},
               #{'name': 'F2aV', 'bins': np.linspace(1, 50, num=50)},
               #{'name': 'F2aW', 'bins': np.linspace(1, 50, num=50)},
               {'name': 'F2bU', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F2bV', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F2bW', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2cU', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2cV', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2cW', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2dU', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2dV', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F2dW', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F2eU', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F2eV', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F2eW', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F3aU', 'bins': np.linspace(0, 1.57, num=50)},
               {'name': 'F3aV', 'bins': np.linspace(0, 1.57, num=50)},
               {'name': 'F3aW', 'bins': np.linspace(0, 1.57, num=50)},
               {'name': 'F3bU', 'bins': np.linspace(0, 1000, num=50)},
               {'name': 'F3bV', 'bins': np.linspace(0, 1000, num=50)},
               {'name': 'F3bW', 'bins': np.linspace(0, 1000, num=50)},
)

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
print("Priors: showers %s, tracks %s" % (showerPrior, 1 - showerPrior))
ptArray = np.repeat(1 - showerPrior, nPfoData)
psArray = np.repeat(showerPrior, nPfoData)
for feature in featurePdfs:
    showerHist, binEdges = np.histogram(dfTrainingShowerData[feature['name']], bins=feature['bins'], density=True)
    trackHist, binEdges = np.histogram(dfTrainingTrackData[feature['name']], bins=feature['bins'], density=True)
    showerHist = np.concatenate(([1], showerHist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
    trackHist = np.concatenate(([1], trackHist, [1]))
    featureValues = dfPfoData[feature['name']]
    histIndices = np.digitize(featureValues, feature['bins'])
    ptArray *= trackHist[histIndices]
    psArray *= showerHist[histIndices]
likelihoodArray = psArray / (ptArray + psArray)
dfPfoData["likelihood"] = likelihoodArray
dfPfoData.to_pickle(outputPickleFile)
print("Finished!")