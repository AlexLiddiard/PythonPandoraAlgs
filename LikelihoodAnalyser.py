import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from PfoGraphicalAnalyser import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire

myTestArea = "/home/tomalex/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.bz2'
trainingFraction = 0.5
performancePreFilters = (
    #'purityU>=0.5',
    #'purityV>=0.5',
    #'purityW>=0.5',
    #'completenessU>=0.5',
    #'completenessV>=0.5',
    #'completenessW>=0.5',
    #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
    #'nHitsU + nHitsV + nHitsW >= 100',
    'nHitsU>=20',
    'nHitsV>=20',
    'nHitsW>=20',
    'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
    'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
    'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
    'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
    'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
    'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
)
likelihoodHistograms = (
    {
        'filters': (
            ('Showers', 'isShower==1', '', True),
            ('Tracks', 'isShower==0', '', True)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log'
    },
    {
        'filters': (
            ('Showers', 'isShower==1', '', True),
            ('Electrons + Positrons', 'absPdgCode==11', 'isShower==1', False),
            ('Photons', 'absPdgCode==22', 'isShower==1', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log'
    },
    {
        'filters': (
            ('Tracks', 'isShower==0', '', True),
            ('Protons', 'absPdgCode==2212', 'isShower==0', False),
            ('Muons', 'absPdgCode==13', 'isShower==0', False),
            ('Charged Pions', 'absPdgCode==211', 'isShower==0', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log'
    },
)

purityCompletenessCutoffGraph = {'bins': np.linspace(0, 1, num=1000)}
purityCompletenessNHitsGraph = {'bins': np.linspace(60, 1400, num=40)}

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
# Apply performance pre-filters
dfPfoData = dfPfoData.query(' and '.join(performancePreFilters))

# Make likelihood histograms.
nPfoData = len(dfPfoData)
nTrainingPfos = m.floor(nPfoData * trainingFraction)
dfPerformancePfoData = dfPfoData[nTrainingPfos:]
for histogram in likelihoodHistograms:
    histogram['name'] = 'Likelihood'
    CreateHistogramWire(dfPerformancePfoData, histogram)

# Plotting Likelihood against purity and completeness.
likelihoodShowers = dfPerformancePfoData.query("isShower==1")['Likelihood']
likelihoodTracks = dfPerformancePfoData.query("isShower==0")['Likelihood']
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
    if trackPurityEfficiency > bestTrackPurityEfficiency and cutOff not in (0, 1):
        bestTrackPurityEfficiency = trackPurityEfficiency
        bestTrackCutoff = cutOff
    trackPurityEfficiencies.append(trackPurityEfficiency)

    showerEfficiencies.append(showerEfficiency)
    showerPurities.append(showerPurity)
    showerPurityEfficiency = showerEfficiency*showerPurity
    if showerPurityEfficiency > bestShowerPurityEfficiency and cutOff not in (0, 1):
        bestShowerPurityEfficiency = showerPurityEfficiency
        bestShowerCutoff = cutOff
    showerPurityEfficiencies.append(showerPurityEfficiency)

# Printing best purity*efficiency for tracks and showers
print("\nBest track purity*efficiency %f, cutoff %f" % (bestTrackPurityEfficiency, bestTrackCutoff))
print("\nTrack Efficiency %f+-%f\n" "Track Purity %f+-%f\n" "Shower Efficiency %f+-%f\n" "Shower Purity %f+-%f\n" % CompletenessPurity(likelihoodTracks, likelihoodShowers, bestTrackCutoff))
print("\nBest shower purity*efficiency %f, cutoff %f" % (bestShowerPurityEfficiency, bestShowerCutoff))
print("\nTrack Efficiency %f+-%f\n" "Track Purity %f+-%f\n" "Shower Efficiency %f+-%f\n" "Shower Purity %f+-%f\n" % CompletenessPurity(likelihoodTracks, likelihoodShowers, bestShowerCutoff))

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
bx1.text(bestTrackCutoff - 0.03, 0.4 ,'Cutoff = %.3f' % bestTrackCutoff, rotation=90, fontsize=12)
bx2.text(bestShowerCutoff - 0.03, 0.4 ,'Cutoff = %.3f' % bestShowerCutoff, rotation=90, fontsize=12)

bx1.set_ylim([0, 1])
bx1.set_title("Purity/Efficiency vs Likelihood Cutoff — Tracks")
bx1.set_xlabel("Likelihood Cutoff")
bx1.set_ylabel("Fraction")

bx2.set_ylim([0, 1])
bx2.set_title("Purity/Efficiency vs Likelihood Cutoff — Showers")
bx2.set_xlabel("Likelihood Cutoff")
bx2.set_ylabel("Fraction")

plt.tight_layout()
plt.savefig("PurityEfficiencyVsLikelihoodCutoff.svg", format='svg', dpi=1200)
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
    likelihoodShowersInBin = dfPerformancePfoData.query("isShower==1 and (nHitsU + nHitsV + nHitsW) >=@nHitsBinMin and (nHitsU + nHitsV + nHitsW) <@nHitsBinMin+@binWidth")['Likelihood']
    likelihoodTracksInBin = dfPerformancePfoData.query("isShower==0 and (nHitsU + nHitsV + nHitsW) >=@nHitsBinMin and (nHitsU + nHitsV + nHitsW) <@nHitsBinMin+@binWidth")['Likelihood']
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

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.errorbar(purityCompletenessNHitsGraph['bins'], trackPurities, yerr=trackPurityErrors, color='b')
ax.errorbar(purityCompletenessNHitsGraph['bins'], trackEfficiencies, yerr=trackEfficiencyErrors, color='r')
ax.errorbar(purityCompletenessNHitsGraph['bins'], trackPurityEfficiencies, yerr=trackPurityEfficiencyErrors, color='g')
ax.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
ax.set_ylim((0.5, 1.01))
#ax.set_title("Purity/Efficiency vs nHits, Cutoff=%.3f, Tracks" % bestShowerCutoff)
ax.set_xlabel("Number of hits")
ax.set_ylabel("Fraction")
plt.tight_layout()
plt.savefig("TrackPurityEfficiencyVsNhits.svg", format='svg', dpi=1200)
plt.show()


fig, ax = plt.subplots(figsize=(10, 7.5))
ax.errorbar(purityCompletenessNHitsGraph['bins'], showerPurities, yerr=showerPurityErrors, color='b')
ax.errorbar(purityCompletenessNHitsGraph['bins'], showerEfficiencies, yerr=showerEfficiencyErrors, color='r')
ax.errorbar(purityCompletenessNHitsGraph['bins'], showerPurityEfficiencies, yerr=showerPurityEfficiencyErrors, color='g')
ax.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
ax.set_ylim((0.5, 1.01))
#ax.set_title("Purity/Efficiency vs nHits, Cutoff=%.3f, Tracks" % bestShowerCutoff)
ax.set_xlabel("Number of hits")
ax.set_ylabel("Fraction")
plt.tight_layout()
plt.savefig("ShowerPurityEfficiencyVsNhits.svg", format='svg', dpi=1200)
plt.show()