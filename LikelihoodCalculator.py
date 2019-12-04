import numpy as np
import pandas as pd
import math as m
import scipy.stats as st
from tqdm import tqdm

myTestArea = "/home/jack/Documents/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp(Processed).pickle'

trainingFraction = 0.5
trainingPreFilters = ('purityU>=0.8',
              'purityV>=0.8',
              'purityW>=0.8',
              'completenessU>=0.8',
              'completenessV>=0.8',
              'completenessW>=0.8',
              #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
              #'nHitsU + nHitsV + nHitsW >= 100',
              'nHitsU>=10',
              'nHitsV>=10',
              'nHitsW>=10',
              'absPdgCode not in [2112, 14, 12]'
,)

featurePdfs = (#{'name': 'F0aU', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F0aV', 'bins': np.linspace(0, 1, num=50)},
               #{'name': 'F0aW', 'bins': np.linspace(0, 1, num=50)},
               {'name': 'F1aU', 'bins': np.linspace(0, 6, num=50)},
               {'name': 'F1aV', 'bins': np.linspace(0, 6, num=50)},
               {'name': 'F1aW', 'bins': np.linspace(0, 6, num=50)},
               #{'name': 'F2aU', 'bins': np.linspace(0, 30, num=31)},
               #{'name': 'F2aV', 'bins': np.linspace(0, 30, num=31)},
               #{'name': 'F2aW', 'bins': np.linspace(0, 30, num=31)},
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
               #{'name': 'F3aU', 'bins': np.linspace(0, 1.57, num=50)},
               #{'name': 'F3aV', 'bins': np.linspace(0, 1.57, num=50)},
               {'name': 'F3aW', 'bins': np.linspace(0, 1.57, num=50)},
               #{'name': 'F3bU', 'bins': np.linspace(0, 1000, num=100)},
               #{'name': 'F3bV', 'bins': np.linspace(0, 1000, num=100)},
               #{'name': 'F3bW', 'bins': np.linspace(0, 1000, num=100)}
              )


def ShowerLikelihood(featurePdfPairs, featureValues, showerPrior):
    Pt = 1 - showerPrior
    Ps = showerPrior
    for (fS, fT), featureValue in zip(featurePdfPairs, featureValues):
        if featureValue == -1:
            continue
        Pt *= fT(featureValue)
        Ps *= fS(featureValue)
    return Ps / (Pt + Ps)


'''Separate true tracks from true showers. Then plot histograms for feature
values. Convert these histograms in to PDFs using scipy. Plot overlapping
histograms for track and shower types.'''
# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
nPfoData = len(dfPfoData)

# Get training PFOs.
nTrainingPfoData = m.floor(nPfoData * trainingFraction)
dfTrainingPfoData = dfPfoData[:nTrainingPfoData]
# Apply pre-filters.
dfTrainingPfoData = dfTrainingPfoData.query(' and '.join(trainingPreFilters))
nTrainingPfoData = len(dfTrainingPfoData)
dfTrainingShowerData = dfTrainingPfoData.query("isShower==1")
dfTrainingTrackData = dfTrainingPfoData.query("isShower==0")
nTrainingShowerData = len(dfTrainingShowerData)
nTrainingTrackData = len(dfTrainingTrackData)
print("Training likelihood using %d tracks and %d showers." % (nTrainingTrackData, nTrainingShowerData))

# Make feature PDFs
featurePdfPairs = []
for pdf in featurePdfs:
    showerHist = np.histogram(dfTrainingShowerData[pdf['name']], bins=pdf['bins'])
    trackHist = np.histogram(dfTrainingTrackData[pdf['name']], bins=pdf['bins'])
    featurePdfPairs.append((st.rv_histogram(showerHist).pdf, st.rv_histogram(trackHist).pdf))

# Calculate likelihood
featureValuesArray = dfPfoData[(feature['name'] for feature in featurePdfs)].to_numpy()
likelihoodArray = np.zeros(nPfoData)
showerPrior = nTrainingShowerData / nTrainingPfoData
print("\nCalculating likelihoods...")
print("Priors: showers %s, tracks %s" % (showerPrior, 1 - showerPrior))
for i in tqdm(range(0, nPfoData)):
    likelihoodArray[i] = ShowerLikelihood(featurePdfPairs, featureValuesArray[i], showerPrior)
dfPfoData["likelihood"] = likelihoodArray
dfPfoData.to_pickle(outputPickleFile)
print("\nFinished!")