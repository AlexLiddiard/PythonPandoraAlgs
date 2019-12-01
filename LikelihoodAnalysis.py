import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from HistoSynthesis import CreateHistogram

myTestArea = "/home/alexliddiard/Desktop/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp(Processed).pickle'
trainingFraction = 0.5
featureHistograms = (#{'name': 'F0a', 'bins': np.linspace(0, 1, num=200)},
                     {'name': 'F1a', 'bins': np.linspace(0, 6, num=50)},
                     #{'name': 'F2a', 'bins': np.linspace(0, 30, num=31)},
                     {'name': 'F2b', 'bins': np.linspace(0, 1, num=50)},
                     #{'name': 'F2c', 'bins': np.linspace(0, 1, num=200)},
                     #{'name': 'F2d', 'bins': np.linspace(0, 1, num=200)},
                     {'name': 'F2e', 'bins': np.linspace(0, 1, num=50)},
                     )
likelihoodHistograms = ({'filters': [('isShower==1', 'Showers'), ('isShower==0', 'Tracks')], 'bins': np.linspace(0, 1, num=25)},
                        {'filters': [('absPdgCode==11', 'Electrons/Positrons'), ('absPdgCode==22', 'Photons')], 'bins': np.linspace(0, 1, num=25)},
                        {'filters': [('absPdgCode==2212', 'Protons'), ('absPdgCode==13', 'Muons'), ('absPdgCode==211', 'Charged Pions')], 'bins': np.linspace(0, 1, num=25)})
purityCompletenessGraph = {'bins': np.linspace(0, 1, num=1000)}

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
    return trackEfficiency, trackPurity, showerEfficiency, showerPurity

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)

# Make feature histograms.
for histogram in featureHistograms:
    histogram['filters'] = [("isShower==1", "Showers"), ("isShower==0", "Tracks")]
    CreateHistogram(dfPfoData, histogram)

# Make likelihood histograms.
nPfoData = len(dfPfoData)
nTrainingPfos = m.floor(nPfoData * trainingFraction)
dfPerformancePfoData = dfPfoData[nTrainingPfos:]
for histogram in likelihoodHistograms:
    histogram['name'] = 'likelihood'
    CreateHistogram(dfPerformancePfoData, histogram)

# Plotting Likelihood against purity and completeness.
likelihoodShowers = dfPerformancePfoData.query("isShower==1")['likelihood']
likelihoodTracks = dfPerformancePfoData.query("isShower==0")['likelihood']
trackEfficiencies = []
trackPurities = []
showerEfficiencies = []
showerPurities = []
trackPurityEfficiencies = []
showerPurityEfficiencies = []

bestTrackCutoff = 0
bestShowerCutoff = 0
bestTrackPurityEfficiency = 0
bestShowerPurityEfficiency = 0
for cutOff in purityCompletenessGraph['bins']:
    trackEfficiency, trackPurity, showerEfficiency, showerPurity = CompletenessPurity(cutOff)
    trackEfficiencies.append(trackEfficiency)
    trackPurities.append(trackPurity)
    trackPurityEfficiency = trackEfficiency*trackPurity
    if trackPurityEfficiency > bestTrackPurityEfficiency:
        bestTrackPurityEfficiency = trackPurityEfficiency
        bestTrackCutoff = cutOff
    trackPurityEfficiencies.append(trackPurityEfficiency)

    showerEfficiencies.append(showerEfficiency)
    showerPurities.append(showerPurity)
    showerPurityEfficiency = showerEfficiency*showerPurity
    if showerPurityEfficiency > bestShowerPurityEfficiency:
        bestShowerPurityEfficiency = showerPurityEfficiency
        bestShowerCutoff = cutOff
    showerPurityEfficiencies.append(showerPurityEfficiency)


fig = plt.figure(figsize=(20,7.5))
bx1 = fig.add_subplot(1,2,1)
bx2 = fig.add_subplot(1,2,2)

bx1.plot(purityCompletenessGraph['bins'], trackPurities, 'b', purityCompletenessGraph['bins'], trackEfficiencies, 'r', purityCompletenessGraph['bins'], trackPurityEfficiencies, 'g')
bx2.plot(purityCompletenessGraph['bins'], showerPurities, 'b', purityCompletenessGraph['bins'], showerEfficiencies, 'r', purityCompletenessGraph['bins'], showerPurityEfficiencies, 'g')

bx1.set_ylim([0, 1])
bx1.set_title("Purity/Completeness/Product vs Likelihood - Tracks")
bx1.set_xlabel("Likelihood")
bx1.set_ylabel("Purity/Completeness/Product Fraction")

bx2.set_ylim([0, 1])
bx2.set_title("Purity/Completeness vs Likelihood - Showers")
bx2.set_xlabel("Likelihood")
bx2.set_ylabel("Purity/Completeness/Product Fraction")

plt.show()

# Printing best purity*efficiency for tracks and showers
print("\nBest track purity*efficiency %f, cutoff %f" % (bestTrackPurityEfficiency, bestTrackCutoff))
print("\nTrack Efficiency %f\n" "Track Purity %f\n" "ShowerEfficiency %f\n" "Shower Purity %f\n" % CompletenessPurity(bestTrackCutoff))
print("\nBest shower purity*efficiency %f, cutoff %f" % (bestShowerPurityEfficiency, bestShowerCutoff))
print("\nTrack Efficiency %f\n" "Track Purity %f\n" "ShowerEfficiency %f\n" "Shower Purity %f\n" % CompletenessPurity(bestShowerCutoff))
