import numpy as np
import pandas as pd
import math as m
import scipy.stats as st
from tqdm import tqdm

myTestArea = "/home/jack/Documents/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp(Processed).pickle'

trainingFraction = 0.5
preFilters = ('purityW>=0.8', 'completenessW>=0.8', 'nHitsW>=10', 'absPdgCode not in [2112, 14, 12]')
featurePdfs = (#{'name': 'F0a', 'bins': np.linspace(0, 1, num=200)},
               {'name': 'F1a', 'bins': np.linspace(0, 6, num=500)},
               #{'name': 'F2a', 'bins': np.linspace(0, 30, num=31)},
               {'name': 'F2b', 'bins': np.linspace(0, 1, num=500)},
               #{'name': 'F2c', 'bins': np.linspace(0, 1, num=200)},
               #{'name': 'F2d', 'bins': np.linspace(0, 1, num=200)},
               {'name': 'F2e', 'bins': np.linspace(0, 1, num=500)},
               {'name': 'F3a', 'bins': np.linspace(0, 1.57, num=100)},
               #{'name': 'F3b', 'bins': np.linspace(0, 1000, num=100)}
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
dfInputPfos = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
dfInputPfos = dfInputPfos.query(' and '.join(preFilters))

# Get training PFOs.
nInputPfoData = len(dfInputPfos)
nTrainingPfoData = m.floor(nInputPfoData * trainingFraction)
dfTrainingPfoData = dfInputPfos[:nTrainingPfoData]
dfTrainingShowerData = dfInputPfos.query("isShower==1")
dfTrainingTrackData = dfInputPfos.query("isShower==0")
nTrainingShowerData = len(dfTrainingShowerData)
nTrainingTrackData = len(dfTrainingTrackData)

# Make feature PDFs
featurePdfPairs = []
for pdf in featurePdfs:
    showerHist = np.histogram(dfTrainingShowerData[pdf['name']], bins=pdf['bins'])
    trackHist = np.histogram(dfTrainingTrackData[pdf['name']], bins=pdf['bins'])
    featurePdfPairs.append((st.rv_histogram(showerHist).pdf, st.rv_histogram(trackHist).pdf))

# Calculate likelihood
featureValuesArray = dfInputPfos[(feature['name'] for feature in featurePdfs)].to_numpy()
likelihoodArray = np.zeros(nInputPfoData)
showerPrior = nTrainingShowerData / nTrainingPfoData
print("\nCalculating likelihoods...")
print("Priors: showers %s, tracks %s" % (showerPrior, 1 - showerPrior))
for i in tqdm(range(0, nInputPfoData)):
    likelihoodArray[i] = ShowerLikelihood(featurePdfPairs, featureValuesArray[i], showerPrior)
dfInputPfos["likelihood"] = likelihoodArray
dfInputPfos.to_pickle(outputPickleFile)
print("\nFinished!")