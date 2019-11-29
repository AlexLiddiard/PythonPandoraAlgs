import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from tqdm import tqdm
from itertools import count

myTestArea = "/home/jack/Documents/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.pickle'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataNew.pickle'
useExistingLikelihood = False

trainingFraction = 0.5
featureHistograms = ({'name': 'F0a', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 0.1},
                     {'name': 'F1a', 'bins': np.linspace(0, 6, num=200), 'graphMaxY': 0.1},
                     {'name': 'F2a', 'bins': np.linspace(0, 30, num=31), 'graphMaxY': 0.2},
                     {'name': 'F2b', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 0.1},
                     {'name': 'F2c', 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 0.1})
otherHistograms = ({'name': 'likelihood', 'filters': [('isShower==1', 'Showers'), ('isShower==0', 'Tracks')], 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 0.1},
                   {'name': 'likelihood', 'filters': [('absPdgCode==11', 'Electrons/Positrons'), ('absPdgCode==22', 'Photons')], 'bins': np.linspace(0, 1, num=200), 'graphMaxY': 0.1})
purityCompletenessOptions = {'bins': np.linspace(0, 1, num=200)}

# Histogram Creator Programme

def CreateHistogram(df, feature):
    
    df = df.query(feature['name'] + '!=-1')

    fig = plt.figure(figsize=(20, 7.5))
    numberOfSubplots = len(feature['filters'])
    f = []

    for i, (filter, name) in zip(count(1), feature['filters']):
        filteredDf = df.query(filter)

        hist = np.histogram(filteredDf[feature['name']], bins = feature['bins'])

        normedFeatureBinCounts = hist[0]/len(filteredDf)
        binWidth = feature['bins'][1] - feature['bins'][0]
        barPositions = feature['bins'][:-1] + binWidth/2
        
        ax = fig.add_subplot(1, numberOfSubplots, i)
        
        ax.bar(barPositions, normedFeatureBinCounts, binWidth)
        ax.set_ylim([0, feature['graphMaxY']])
        ax.set_title("%s Probability Density - %s" %(feature['name'], name))
        ax.set_xlabel(feature['name'])
        ax.set_ylabel("Probability")
        f.append(st.rv_histogram(hist).pdf)

    plt.show()
    
    return f

def ShowerLikelihood(featurePdfPairs, featureValues, prior):
    Pt = 1 - prior
    Ps = prior
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
nInputPfos = len(dfInputPfos)
nTrainingPfos = m.floor(nInputPfos * trainingFraction)
dfTrainingPfos = dfInputPfos[:nTrainingPfos]
nTrainingShowers = len(dfInputPfos.query("isShower==1"))

# Make histograms, PDFs, and the likelihood function
featurePdfPairs = []
for feature in featureHistograms:
    feature['filters'] = [("isShower==1", "Showers"), ("isShower==0", "Tracks")]
    featurePdfPairs.append(CreateHistogram(dfTrainingPfos, feature))


if 'likelihood' not in dfInputPfos or not useExistingLikelihood:
    # Evaluate likelihood for all PFOs
    featureNames = (feature['name'] for feature in featureHistograms)
    featureValuesArray = dfInputPfos[featureNames].to_numpy()
    likelihoodArray = np.zeros(nInputPfos)
    prior = nTrainingShowers / nTrainingPfos
    print("\nCalculating likelihoods...")
    print("prior = " + str(prior))
    for i in tqdm(range(0, nInputPfos)):
        likelihoodArray[i] = ShowerLikelihood(featurePdfPairs, featureValuesArray[i], prior)
    dfInputPfos["likelihood"] = likelihoodArray
    dfInputPfos.to_pickle(outputPickleFile)


dfPerformancePfos = dfInputPfos[nTrainingPfos:]
for histogram in otherHistograms:
    featurePdfPairs.append(CreateHistogram(dfPerformancePfos, histogram))

likelihoodShowers = dfPerformancePfos.query("isShower==1")['likelihood']
likelihoodTracks = dfPerformancePfos.query("isShower==0")['likelihood']

# Calculating completeness-purity against likelihood cut-off function
def CompletenessPurity(cutOff):
    sumCorrectShowers = (likelihoodShowers > cutOff).sum()
    sumIncorrectShowers = (likelihoodShowers < cutOff).sum()
    sumCorrectTracks = (likelihoodTracks < cutOff).sum()
    sumIncorrectTracks = (likelihoodTracks > cutOff).sum()
    trackEfficiency = sumCorrectTracks/(sumCorrectTracks+sumIncorrectTracks)
    trackPurity = sumCorrectTracks/(sumCorrectTracks + sumIncorrectShowers)
    showerEfficiency = sumCorrectShowers/(sumCorrectShowers+sumIncorrectShowers)
    showerPurity = sumCorrectShowers/(sumCorrectShowers + sumIncorrectTracks)
    return (trackEfficiency, trackPurity, showerEfficiency, showerPurity)

# Printing completeness-purity for likelihood = 0.89
print("\nTrack Efficiency %f\n" "Track Purity %f\n" "ShowerEfficiency %f\n" "Shower Purity %f\n" %CompletenessPurity(0.89))

# Plotting Likelihood against purity and completeness.
trackEfficiencies = []
trackPurities = []
showerEfficiencies = []
showerPurities = []
trackPurityEfficiency = []
showerPurityEfficiency = []

for cutOff in purityCompletenessOptions['bins']:
    CompletenessPurityTuple = CompletenessPurity(cutOff)
    trackEfficiencies.append(CompletenessPurityTuple[0])
    trackPurities.append(CompletenessPurityTuple[1])
    trackPurityEfficiency.append(CompletenessPurityTuple[0]*CompletenessPurityTuple[1])
    showerEfficiencies.append(CompletenessPurityTuple[2])
    showerPurities.append(CompletenessPurityTuple[3])
    showerPurityEfficiency.append(CompletenessPurityTuple[2]*CompletenessPurityTuple[3])

fig = plt.figure(figsize=(20,7.5))
bx1 = fig.add_subplot(1,2,1)
bx2 = fig.add_subplot(1,2,2)

bx1.plot(purityCompletenessOptions['bins'], trackPurities, 'b', purityCompletenessOptions['bins'], trackEfficiencies, 'r', purityCompletenessOptions['bins'], trackPurityEfficiency, 'g')
bx2.plot(purityCompletenessOptions['bins'], showerPurities, 'b', purityCompletenessOptions['bins'], showerEfficiencies, 'r', purityCompletenessOptions['bins'], showerPurityEfficiency, 'g')

bx1.set_ylim([0, 1])
bx1.set_title("%s Purity/Completeness/Product vs Likelihood - Tracks")
bx1.set_xlabel("Likelihood")
bx1.set_ylabel("Purity/Completeness/Product Fraction")

bx2.set_ylim([0, 1])
bx2.set_title("%s Purity/Completeness vs Likelihood - Showers")
bx2.set_xlabel("Likelihood")
bx2.set_ylabel("Purity/Completeness/Product Fraction")

plt.show()
