import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from tqdm import tqdm

inputFile = 'featureData.pickle'
trainingFraction = 0.5
featureList = ({'name': 'F0a', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 40},
               {'name': 'F1a', 'bins': np.linspace(0, 6, num=200), 'graphMaxY': 3},
               {'name': 'F2a', 'bins': np.linspace(0, 30, num=31), 'graphMaxY': 1},
               {'name': 'F2b', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 40},
               {'name': 'F2c', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 40})
likelihoodOptions = {'bins': np.linspace(0, 1, num=200), 'graphMaxY': 200}

#Histogram Creator Programme

def GetFeatureStats(df_shower, df_track, feature):
    fig = plt.figure(figsize=(20,7.5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    shower_filter = df_shower[feature['name']] != -1
    track_filter = df_track[feature['name']] != -1
    showerFeatureData = df_shower[shower_filter][feature['name']]
    trackFeatureData = df_track[track_filter][feature['name']]
    ax1.hist(showerFeatureData, bins=feature['bins'], density=1)
    ax2.hist(trackFeatureData, bins=feature['bins'], density=1)
    ax3.hist(showerFeatureData, bins=feature['bins'], density=1)
    ax3.hist(trackFeatureData, bins=feature['bins'], density=1)

    ax1.set_ylim([0, feature['graphMaxY']])
    ax1.set_title("%s probability density - Showers" % feature['name'])
    ax1.set_xlabel(feature['name'])
    ax1.set_ylabel("Frequency Density")
    ax2.set_ylim([0, feature['graphMaxY']])
    ax2.set_title("%s probability density - Tracks" % feature['name'])
    ax2.set_xlabel(feature['name'])
    ax2.set_ylabel("Frequency Density")
    ax3.set_ylim([0, feature['graphMaxY']])
    ax3.set_title("%s probability density - Showers + Tracks" % feature['name'])
    ax3.set_xlabel(feature['name'])
    ax3.set_ylabel("Frequency Density")
    plt.show()

    shower_hist = np.histogram(showerFeatureData, bins=feature['bins'])
    track_hist = np.histogram(trackFeatureData, bins=feature['bins'])
    fS = st.rv_histogram(shower_hist).pdf
    fT = st.rv_histogram(track_hist).pdf
    return fS, fT


def Likelihood(featurePdfPairs, featureValues):
    Pt = 1
    Ps = 1
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
dfInputPfos = pd.read_pickle(inputFile)
nInputPfos = len(dfInputPfos)
nTrainingPfos = m.floor(nInputPfos * trainingFraction)
dfTrainingPfos = dfInputPfos[:nTrainingPfos]
showerFilter = dfTrainingPfos['pfoTrueType'] == 1
trackFilter = dfTrainingPfos['pfoTrueType'] == 0
dfTrainingShowers = dfTrainingPfos[showerFilter]
dfTrainingTracks = dfTrainingPfos[trackFilter]

# Make histograms, PDFs, and the likelihood function
featurePdfPairs = []
for feature in featureList:
    featurePdfPairs.append(GetFeatureStats(dfTrainingShowers, dfTrainingTracks, feature))


# Evaluate likelihood for all PFOs
featureNames = (feature['name'] for feature in featureList)
featureValuesArray = dfInputPfos[featureNames].to_numpy()
likelihoodArray = np.zeros(nInputPfos)
print("\nCalculating likelihoods...")
for i in tqdm(range(0, nInputPfos)):
    likelihoodArray[i] = Likelihood(featurePdfPairs, featureValuesArray[i])
dfInputPfos["likelihood"] = likelihoodArray
dfInputPfos.to_pickle(inputFile)

dfPerformancePfos = dfInputPfos[nTrainingPfos:]
showerFilter = dfPerformancePfos['pfoTrueType'] == 1
trackFilter = dfPerformancePfos['pfoTrueType'] == 0
likelihoodShowers = dfPerformancePfos[showerFilter]['likelihood']
likelihoodTracks = dfPerformancePfos[trackFilter]['likelihood']

fig = plt.figure(figsize=(20,7.5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.hist(likelihoodShowers, likelihoodOptions['bins'], density=1)
ax2.hist(likelihoodTracks, likelihoodOptions['bins'], density=1)
ax3.hist(likelihoodShowers, likelihoodOptions['bins'], density=1)
ax3.hist(likelihoodTracks, likelihoodOptions['bins'], density=1)
ax1.set_ylim([0, likelihoodOptions['graphMaxY']])
ax1.set_title("Likelihood probability density - Showers")
ax1.set_xlabel("Likelihood")
ax1.set_ylabel("Frequency Density")
ax2.set_ylim([0, likelihoodOptions['graphMaxY']])
ax2.set_title("Likelihood probability density - Tracks")
ax2.set_xlabel("Likelihood")
ax2.set_ylabel("Frequency Density")
ax3.set_ylim([0, likelihoodOptions['graphMaxY']])
ax3.set_title("Likelihood probability density - Showers + Tracks")
ax3.set_xlabel("Likelihood")
ax3.set_ylabel("Frequency Density")
plt.show()

sumCorrectShowers = (likelihoodShowers > 0.5).sum()
sumIncorrectShowers = (likelihoodShowers < 0.5).sum()
sumCorrectTracks = (likelihoodTracks < 0.5).sum()
sumIncorrectTracks = (likelihoodTracks > 0.5).sum()
trackEfficiency = sumCorrectTracks/(sumCorrectTracks+sumIncorrectTracks)
trackPurity = sumCorrectTracks/(sumCorrectTracks + sumIncorrectShowers)
showerEfficiency = sumCorrectShowers/(sumCorrectShowers+sumIncorrectShowers)
showerPurity = sumCorrectShowers/(sumCorrectShowers + sumIncorrectTracks)
print("\nTrack Efficiency %f\n" "Track Purity %f\n" "ShowerEfficiency %f\n" "Shower Purity %f\n" %(trackEfficiency, trackPurity, showerEfficiency, showerPurity) )