import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from HistoSynthesis import CreateHistogram

myTestArea = "/home/alexliddiard/Desktop/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp(Processed).pickle'
trainingFraction = 0.5
featureHistograms = (#{'name': 'F0aU', 'bins': np.linspace(0, 1, num=50)},
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
likelihoodHistograms = ({'filters': [('isShower==1', 'Showers'), ('isShower==0', 'Tracks')], 'bins': np.linspace(0, 1, num=25)},
                        {'filters': [('absPdgCode==11', 'Electrons/Positrons'), ('absPdgCode==22', 'Photons')], 'bins': np.linspace(0, 1, num=25)},
                        {'filters': [('absPdgCode==2212', 'Protons'), ('absPdgCode==13', 'Muons'), ('absPdgCode==211', 'Charged Pions')], 'bins': np.linspace(0, 1, num=25)})
purityCompletenessCutoffGraph = {'bins': np.linspace(0, 1, num=1000)}
purityCompletenessNHitsGraph = {'bins': np.linspace(10, 600, num=30)}

# Calculating completeness-purity against likelihood cut-off function

def Purity(efficiency1, efficiency2, n1, n2):
    Purity1 = 1/(1 + (1 - efficiency2)*n2/(efficiency1*n1))
    return Purity1

def PurityError(efficiency1, efficiencyError1, efficiency2, efficiencyError2, n1, n2):
    purity1 = Purity(efficiency1, efficiency2, n1, n2)
    purityOff1a = Purity(efficiency1 + efficiencyError1, efficiency2, n1, n2)
    purityOff1b = Purity(efficiency1, efficiency2 + efficiencyError2, n1, n2)
    purityErr1a = purityOff1a - purity1
    purityErr1b = purityOff1b - purity1
    purityErr1 = m.sqrt(purityErr1a**2 + purityErr1b**2)
    return purityErr1

def CompletenessPurity(likelihoodTracks, likelihoodShowers, cutOff):
    sumTracks = len(likelihoodTracks)
    sumShowers = len(likelihoodShowers)
    sumCorrectShowers = (likelihoodShowers > cutOff).sum()
    sumIncorrectShowers = (likelihoodShowers < cutOff).sum()
    sumCorrectTracks = (likelihoodTracks < cutOff).sum()
    sumIncorrectTracks = (likelihoodTracks > cutOff).sum()
    trackEfficiency = sumCorrectTracks/(sumCorrectTracks+sumIncorrectTracks)
    trackEfficiencyError = m.sqrt(trackEfficiency * (1 - trackEfficiency) / sumTracks)
    trackPurity = sumCorrectTracks/(sumCorrectTracks + sumIncorrectShowers)
    showerEfficiency = sumCorrectShowers/(sumCorrectShowers+sumIncorrectShowers)
    showerEfficiencyError = m.sqrt(showerEfficiency * (1 - showerEfficiency) / sumShowers)
    showerPurity = sumCorrectShowers/(sumCorrectShowers + sumIncorrectTracks)
    trackPurityError = PurityError(trackEfficiency, trackEfficiencyError, showerEfficiency, showerEfficiencyError, sumTracks, sumShowers)
    showerPurityError = PurityError(showerEfficiency, showerEfficiencyError, trackEfficiency, trackEfficiencyError, sumShowers, sumTracks)
    return trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError

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
for cutOff in purityCompletenessCutoffGraph['bins']:
    trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError = CompletenessPurity(likelihoodTracks, likelihoodShowers, cutOff)
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

# Printing best purity*efficiency for tracks and showers
print("\nBest track purity*efficiency %f, cutoff %f" % (bestTrackPurityEfficiency, bestTrackCutoff))
print("\nTrack Efficiency %f+-%f\n" "Track Purity %f+-%f\n" "ShowerEfficiency %f+-%f\n" "Shower Purity %f+-%f\n" % CompletenessPurity(likelihoodTracks, likelihoodShowers, bestTrackCutoff))
print("\nBest shower purity*efficiency %f, cutoff %f" % (bestShowerPurityEfficiency, bestShowerCutoff))
print("\nTrack Efficiency %f+-%f\n" "Track Purity %f+-%f\n" "ShowerEfficiency %f+-%f\n" "Shower Purity %f+-%f\n" % CompletenessPurity(likelihoodTracks, likelihoodShowers, bestShowerCutoff))

# Show purity/completeness graph
fig = plt.figure(figsize=(20,7.5))
bx1 = fig.add_subplot(1,2,1)
bx2 = fig.add_subplot(1,2,2)

lines = bx1.plot(purityCompletenessCutoffGraph['bins'], trackPurities, 'b', purityCompletenessCutoffGraph['bins'], trackEfficiencies, 'r', purityCompletenessCutoffGraph['bins'], trackPurityEfficiencies, 'g')
bx1.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
lines = bx2.plot(purityCompletenessCutoffGraph['bins'], showerPurities, 'b', purityCompletenessCutoffGraph['bins'], showerEfficiencies, 'r', purityCompletenessCutoffGraph['bins'], showerPurityEfficiencies, 'g')
bx2.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
bx1.axvline(bestTrackCutoff)
bx2.axvline(bestShowerCutoff)
bx1.text(bestTrackCutoff - 0.03, 0.4 ,'Cutoff = %.2f' % bestTrackCutoff, rotation=90, fontsize=12)
bx2.text(bestShowerCutoff - 0.03, 0.4 ,'Cutoff = %.2f' % bestShowerCutoff, rotation=90, fontsize=12)

bx1.set_ylim([0, 1])
bx1.set_title("Purity/Completeness/Product vs Likelihood - Tracks")
bx1.set_xlabel("Likelihood")
bx1.set_ylabel("Fraction")

bx2.set_ylim([0, 1])
bx2.set_title("Purity/Completeness vs Likelihood - Showers")
bx2.set_xlabel("Likelihood")
bx2.set_ylabel("Fraction")

plt.show()

trackEfficiencies = []
trackEfficiencyErrors = []
trackPurities = []
trackPurityErrors = []
trackPurityEfficiencies = []
trackPurityEfficiencyErrors = []
showerEfficiencies = []
showerEfficiencyErrors = []
showerPurities = []
showerPurityErrors = []
showerPurityEfficiencies = []
showerPurityEfficiencyErrors = []

binWidth = purityCompletenessNHitsGraph['bins'][1] - purityCompletenessNHitsGraph['bins'][0]

def PurityEfficiencyError(purity, purityError, efficiency, efficiencyError):
    purityEfficiencyPurityErr = purityError*efficiency
    purityEfficiencyEfficiencyErr = purity*efficiencyError
    error = m.sqrt(purityEfficiencyPurityErr**2 + purityEfficiencyEfficiencyErr**2)
    return error

for nHitsBinMin in purityCompletenessNHitsGraph['bins']:
    likelihoodShowersInBin = dfPerformancePfoData.query("isShower==1 and nHitsW>=@nHitsBinMin and nHitsW <@nHitsBinMin+@binWidth")['likelihood']
    likelihoodTracksInBin = dfPerformancePfoData.query("isShower==0 and nHitsW>=@nHitsBinMin and nHitsW <@nHitsBinMin+@binWidth")['likelihood']
    trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError = CompletenessPurity(likelihoodTracksInBin, likelihoodShowersInBin, bestShowerCutoff)

    trackEfficiencies.append(trackEfficiency)
    trackEfficiencyErrors.append(trackEfficiencyError)
    trackPurities.append(trackPurity)
    trackPurityEfficiency = trackEfficiency*trackPurity
    trackPurityEfficiencies.append(trackPurityEfficiency)
    trackPurityErrors.append(trackPurityError)

    showerEfficiencies.append(showerEfficiency)
    showerEfficiencyErrors.append(showerEfficiencyError)
    showerPurities.append(showerPurity)
    showerPurityEfficiency = showerEfficiency*showerPurity
    showerPurityEfficiencies.append(showerPurityEfficiency)
    showerPurityErrors.append(showerPurityError)

    trackPurityEfficiencyErrors.append(PurityEfficiencyError(trackPurity, trackPurityError, trackEfficiency, trackEfficiencyError))
    showerPurityEfficiencyErrors.append(PurityEfficiencyError(showerPurity, showerPurityError, showerEfficiency, showerEfficiencyError))

fig = plt.figure(figsize=(20,7.5))
bx1 = fig.add_subplot(1,2,1)
bx2 = fig.add_subplot(1,2,2)

bx1.errorbar(purityCompletenessNHitsGraph['bins'], trackPurities, yerr=trackPurityErrors, color='b')
bx1.errorbar(purityCompletenessNHitsGraph['bins'], trackEfficiencies, yerr=trackEfficiencyErrors, color='r')
bx1.errorbar(purityCompletenessNHitsGraph['bins'], trackPurityEfficiencies, yerr=trackPurityEfficiencyErrors, color='g')
bx1.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')

bx2.errorbar(purityCompletenessNHitsGraph['bins'], showerPurities, yerr=showerPurityErrors, color='b')
bx2.errorbar(purityCompletenessNHitsGraph['bins'], showerEfficiencies, yerr=showerEfficiencyErrors, color='r')
bx2.errorbar(purityCompletenessNHitsGraph['bins'], showerPurityEfficiencies, yerr=showerPurityEfficiencyErrors, color='g')
bx2.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')

bx1.set_ylim([0.4, 1])
bx1.set_title("Purity/Completeness/Product vs nHitsW - Tracks")
bx1.set_xlabel("nHitsW")
bx1.set_ylabel("Fraction")

bx2.set_ylim([0.4, 1])
bx2.set_title("Purity/Completeness vs nHitsW - Showers")
bx2.set_xlabel("nHitsW")
bx2.set_ylabel("Fraction")

plt.show()