import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from PfoGraphicalAnalyser import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire
from itertools import count

myTestArea = "/home/tomalex/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
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
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Showers', 'isShower==1', '', True),
            ('Electrons + Positrons', 'absPdgCode==11', 'isShower==1', False),
            ('Photons', 'absPdgCode==22', 'isShower==1', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Tracks', 'isShower==0', '', True),
            ('Protons', 'absPdgCode==2212', 'isShower==0', False),
            ('Muons', 'absPdgCode==13', 'isShower==0', False),
            ('Charged Pions', 'absPdgCode==211', 'isShower==0', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'track'
    },
)

purityEfficiencyCutoffGraph = {'testValues': np.linspace(0, 1, 1001)}
purityEfficiencyNhitsGraph = {'bins': np.linspace(60, 1400, num=40)}

def Purity(efficiency1, efficiency2, n1, n2):
    Purity1 = 1/(1 + (1 - efficiency2)*n2/(efficiency1*n1))
    return Purity1

# Calculates error on purity1 using efficiency errors
def PurityError(efficiency1, efficiencyError1, efficiency2, efficiencyError2, n1, n2):
    purity1 = Purity(efficiency1, efficiency2, n1, n2)
    purityOff1a = Purity(efficiency1 + efficiencyError1, efficiency2, n1, n2)
    purityOff1b = Purity(efficiency1, efficiency2 + efficiencyError2, n1, n2)
    purityErr1a = purityOff1a - purity1
    purityErr1b = purityOff1b - purity1
    purityErr1 = m.sqrt(purityErr1a**2 + purityErr1b**2)
    return purityErr1

def PurityEfficiencyError(purity, purityError, efficiency, efficiencyError):
    return m.sqrt((purityError*efficiency)**2 + (purity*efficiencyError)**2)

def PurityEfficiency(dfTrackVariable, dfShowerVariable, cutOff, showerCutDirection):
    sumTracks = len(dfTrackVariable)
    sumShowers = len(dfShowerVariable)
    sumCorrectShowers = (dfShowerVariable > cutOff).sum()
    sumIncorrectShowers = (dfShowerVariable < cutOff).sum()
    sumCorrectTracks = (dfTrackVariable < cutOff).sum()
    sumIncorrectTracks = (dfTrackVariable > cutOff).sum()
    if showerCutDirection == "left":
        sumCorrectShowers, sumIncorrectShowers = sumIncorrectShowers, sumCorrectShowers
        sumCorrectTracks, sumIncorrectTracks = sumIncorrectTracks, sumCorrectTracks
    trackEfficiency = sumCorrectTracks/(sumCorrectTracks+sumIncorrectTracks)
    trackPurity = sumCorrectTracks/(sumCorrectTracks + sumIncorrectShowers)
    trackPurityEfficiency = trackPurity * trackEfficiency
    showerEfficiency = sumCorrectShowers/(sumCorrectShowers+sumIncorrectShowers)
    showerPurity = sumCorrectShowers/(sumCorrectShowers + sumIncorrectTracks)
    showerPurityEfficiency = showerPurity * showerEfficiency
    trackEfficiencyError = m.sqrt(trackEfficiency * (1 - trackEfficiency) / sumTracks)
    showerEfficiencyError = m.sqrt(showerEfficiency * (1 - showerEfficiency) / sumShowers)
    trackPurityError = PurityError(trackEfficiency, trackEfficiencyError, showerEfficiency, showerEfficiencyError, sumTracks, sumShowers)
    showerPurityError = PurityError(showerEfficiency, showerEfficiencyError, trackEfficiency, trackEfficiencyError, sumShowers, sumTracks)
    trackPurityEfficiencyError = PurityEfficiencyError(trackPurityEfficiency, trackPurityError, trackEfficiency, trackEfficiencyError)
    showerPurityEfficiencyError = PurityEfficiencyError(showerPurityEfficiency, showerPurityError, showerEfficiency, showerEfficiencyError)
    return (
        trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, trackPurityEfficiency, trackPurityEfficiencyError, 
        showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError, showerPurityEfficiency, showerPurityEfficiencyError
    )

def GraphCutoffLine(ax, cutoff, arrows=False, flipArrows=False):
    ax.axvline(cutoff)
    if arrows:
        if not flipArrows:
            ax.annotate(
                "Track", xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="right", va="center",
                xytext=(-18,0), textcoords="offset points", bbox={'boxstyle': "larrow", 'fc': 'C1', 'ec': 'C1', 'alpha': 0.5}, fontsize=20
            )
            ax.annotate(
                "Shower", xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="left", va="center",
                xytext=(18,0), textcoords="offset points", bbox={'boxstyle': "rarrow", 'fc': 'C0', 'ec': 'C0', 'alpha': 0.5}, fontsize=20
            )
        else:
            ax.annotate(
                "Track", xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="left", va="center",
                xytext=(18,0), textcoords="offset points", bbox={'boxstyle': "rarrow", 'fc': 'C1', 'ec': 'C1', 'alpha': 0.5}, fontsize=20
            )
            ax.annotate(
                "Shower", xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="right", va="center",
                xytext=(-18,0), textcoords="offset points", bbox={'boxstyle': "larrow", 'fc': 'C0', 'ec': 'C0', 'alpha': 0.5}, fontsize=20
            )


def OptimiseCutoff(dfTrackData, dfShowerData, variableName, testCutoffs, showerCutDirection):
    dfTrackVariable = dfTrackData.query(variableName + "!=-1")[variableName]
    dfShowerVariable =  dfShowerData.query(variableName + "!=-1")[variableName]
    trackEfficiencies = []
    trackPurities = []
    showerEfficiencies = []
    showerPurities = []
    trackPurityEfficiencies = []
    showerPurityEfficiencies = []
    bestShowerCutoff = 0
    bestTrackCutoff = 0
    bestTrackPurityEfficiency = 0
    bestShowerPurityEfficiency = 0
    for cutoff in testCutoffs:
        (
            trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, trackPurityEfficiency, trackPurityEfficiencyError, 
            showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError, showerPurityEfficiency, showerPurityEfficiencyError
        ) = PurityEfficiency(dfTrackVariable, dfShowerVariable, cutoff, showerCutDirection)
        trackEfficiencies.append(trackEfficiency)
        trackPurities.append(trackPurity)
        if trackPurityEfficiency > bestTrackPurityEfficiency and cutoff not in (0, 1):
            bestTrackPurityEfficiency = trackPurityEfficiency
            bestTrackCutoff = cutoff
        trackPurityEfficiencies.append(trackPurityEfficiency)

        showerEfficiencies.append(showerEfficiency)
        showerPurities.append(showerPurity)
        if showerPurityEfficiency > bestShowerPurityEfficiency and cutoff not in (0, 1):
            bestShowerPurityEfficiency = showerPurityEfficiency
            bestShowerCutoff = cutoff
        showerPurityEfficiencies.append(showerPurityEfficiency)
    return (
        bestTrackCutoff, trackEfficiencies, trackPurities, trackPurityEfficiencies, 
        bestShowerCutoff, showerEfficiencies, showerPurities, showerPurityEfficiencies
    )

def PrintPurityEfficiency(dfTrackData, dfShowerData, variableName, cutoff):
    dfTrackVariable = dfTrackData.query(variableName + "!=-1")[variableName]
    dfShowerVariable =  dfShowerData.query(variableName + "!=-1")[variableName]
    print(
        "Track Efficiency %.3f+-%.3f\n"
        "Track Purity %.3f+-%.3f\n"
        "Track Purity * Efficiency %.3f+-%.3f\n"
        "Shower Efficiency %.3f+-%.3f\n"
        "Shower Purity %.3f+-%.3f\n"
        "Shower Purity * Efficiency %.3f+-%.3f"
    % PurityEfficiency(dfTrackVariable, dfShowerVariable, cutoff, 'right')
)

if __name__ == "__main__":
    # Load the pickle file.
    dfPfoData = pd.read_pickle(inputPickleFile) #note tha
    nPfoData = len(dfPfoData)

    # Get training PFOs.
    dfPerfPfoData = dfPfoData[:m.floor(nPfoData * trainingFraction)] # note: the likelihood calculator already shuffles the PFOs
    # Apply training pre-filters.
    dfPerfPfoData = dfPerfPfoData.query(' and '.join(performancePreFilters))
    dfPerfShowerData = dfPerfPfoData.query("isShower==1")
    dfPerfTrackData = dfPerfPfoData.query("isShower==0")
    nPerfPfoData = len(dfPerfPfoData)
    nPerfShowerData = len(dfPerfShowerData)
    nPerfTrackData = len(dfPerfTrackData)
    print("Testing likelihood using %d tracks and %d showers." % (nPerfTrackData, nPerfShowerData))

    # Get optimal purity and efficiency
    (
        bestTrackCutoff, trackEfficiencies, trackPurities, trackPurityEfficiencies, 
        bestShowerCutoff, showerEfficiencies, showerPurities, showerPurityEfficiencies
    ) = OptimiseCutoff(dfPerfTrackData, dfPerfShowerData, 'Likelihood', purityEfficiencyCutoffGraph['testValues'], 'right')

    # Printing results for optimal purity and efficiency
    print("\nOptimal track cutoff %.3f" % bestTrackCutoff)
    PrintPurityEfficiency(dfPerfTrackData, dfPerfShowerData, 'Likelihood', bestTrackCutoff)
    print("\nOptimal shower cutoff %.3f" % bestShowerCutoff)
    PrintPurityEfficiency(dfPerfTrackData, dfPerfShowerData, 'Likelihood', bestShowerCutoff)

    # Make likelihood histograms.
    for histogram in likelihoodHistograms:
        histogram['name'] = 'Likelihood'
        fig, ax = CreateHistogramWire(dfPerfPfoData, histogram)
        cutoff = histogram.get('cutoff', '')
        if cutoff == 'shower':
            GraphCutoffLine(ax, bestShowerCutoff, ("Track", "Shower"))
        if cutoff == 'track':
            GraphCutoffLine(ax, bestTrackCutoff, ("Track", "Shower"))
        plt.savefig('%s distribution for %s' % (histogram['name'], ', '.join((filter[0] for filter in histogram['filters'])) + '.svg'), format='svg', dpi=1200)
        plt.show()

    # Plot purity/efficiency against likelihood
    fig = plt.figure(figsize=(20,7.5))
    bx1 = fig.add_subplot(1,2,1)
    bx2 = fig.add_subplot(1,2,2)

    lines = bx1.plot(purityEfficiencyCutoffGraph['testValues'], trackPurities, 'b', purityEfficiencyCutoffGraph['testValues'], trackEfficiencies, 'r', purityEfficiencyCutoffGraph['testValues'], trackPurityEfficiencies, 'g')
    bx1.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
    lines = bx2.plot(purityEfficiencyCutoffGraph['testValues'], showerPurities, 'b', purityEfficiencyCutoffGraph['testValues'], showerEfficiencies, 'r', purityEfficiencyCutoffGraph['testValues'], showerPurityEfficiencies, 'g')
    bx2.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
    GraphCutoffLine(bx1, bestTrackCutoff)
    GraphCutoffLine(bx2, bestShowerCutoff)

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

    # Plot purity/efficiency as a function of hits
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

    binWidth = purityEfficiencyNhitsGraph['bins'][1] - purityEfficiencyNhitsGraph['bins'][0]

    for nHitsBinMin in purityEfficiencyNhitsGraph['bins'][:-1]:
        likelihoodShowersInBin = dfPerfPfoData.query("isShower==1 and (nHitsU + nHitsV + nHitsW) >=@nHitsBinMin and (nHitsU + nHitsV + nHitsW) <@nHitsBinMin+@binWidth")['Likelihood']
        likelihoodTracksInBin = dfPerfPfoData.query("isShower==0 and (nHitsU + nHitsV + nHitsW) >=@nHitsBinMin and (nHitsU + nHitsV + nHitsW) <@nHitsBinMin+@binWidth")['Likelihood']
        (
            trackEfficiency, trackEfficiencyError, trackPurity, trackPurityError, trackPurityEfficiency, trackPurityEfficiencyError, 
            showerEfficiency, showerEfficiencyError, showerPurity, showerPurityError, showerPurityEfficiency, showerPurityEfficiencyError
        ) = PurityEfficiency(likelihoodTracksInBin, likelihoodShowersInBin, bestShowerCutoff, 'right')

        trackEfficiencies.append(trackEfficiency)
        trackEfficiencyErrors.append(trackEfficiencyError)
        trackPurities.append(trackPurity)
        trackPurityErrors.append(trackPurityError)
        trackPurityEfficiencies.append(trackPurityEfficiency)
        trackPurityEfficiencyErrors.append(trackPurityEfficiencyError)

        showerEfficiencies.append(showerEfficiency)
        showerEfficiencyErrors.append(showerEfficiencyError)
        showerPurities.append(showerPurity)
        showerPurityErrors.append(showerPurityError)
        showerPurityEfficiencies.append(showerPurityEfficiency)
        showerPurityEfficiencyErrors.append(showerPurityEfficiencyError)


    Xcoord = np.concatenate((purityEfficiencyNhitsGraph['bins'][:1], np.repeat(purityEfficiencyNhitsGraph['bins'][1:-1], 2), purityEfficiencyNhitsGraph['bins'][-1:]))
    XcoordErrorBars = purityEfficiencyNhitsGraph['bins'][:-1] + (purityEfficiencyNhitsGraph['bins'][1] - purityEfficiencyNhitsGraph['bins'][0]) / 2

    fig, ax = plt.subplots(figsize=(10, 7.5))
    trackPuritiesYcoord = np.repeat(trackPurities, 2)
    trackEfficienciesYcoord = np.repeat(trackEfficiencies, 2)
    trackPurityEfficienciesYcoord = np.repeat(trackPurityEfficiencies, 2)
    ax.plot(Xcoord, trackPuritiesYcoord, label='Purity', color='r')
    ax.plot(Xcoord, trackEfficienciesYcoord, label='Efficiency', color='g')
    ax.plot(Xcoord, trackPurityEfficienciesYcoord, label='Purity * Efficiency', color='b')
    ax.errorbar(XcoordErrorBars, trackPurities, yerr=trackPurityErrors, fmt="none", capsize=2, color='r')
    ax.errorbar(XcoordErrorBars, trackEfficiencies, yerr=trackEfficiencyErrors, fmt="none", capsize=2, color='g')
    ax.errorbar(XcoordErrorBars, trackPurityEfficiencies, yerr=trackPurityEfficiencyErrors, fmt="none", capsize=2, color='b')
    ax.legend(loc='lower center')
    ax.set_ylim((0.5, 1.01))
    #ax.set_title("Purity/Efficiency vs nHits, Cutoff=%.3f, Tracks" % bestShowerCutoff)
    ax.set_xlabel("Number of hits")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    plt.savefig("TrackPurityEfficiencyVsNhits.svg", format='svg', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 7.5))
    showerPuritiesYcoord = np.repeat(showerPurities, 2)
    showerEfficienciesYcoord = np.repeat(showerEfficiencies, 2)
    showerPurityEfficienciesYcoord = np.repeat(showerPurityEfficiencies, 2)
    Xcoord = np.concatenate((purityEfficiencyNhitsGraph['bins'][:1], np.repeat(purityEfficiencyNhitsGraph['bins'][1:-1], 2), purityEfficiencyNhitsGraph['bins'][-1:]))
    XcoordErrorBars = purityEfficiencyNhitsGraph['bins'][:-1] + (purityEfficiencyNhitsGraph['bins'][1] - purityEfficiencyNhitsGraph['bins'][0]) / 2
    ax.plot(Xcoord, showerPuritiesYcoord, label='Purity', color='r')
    ax.plot(Xcoord, showerEfficienciesYcoord, label='Efficiency', color='g')
    ax.plot(Xcoord, showerPurityEfficienciesYcoord, label='Purity * Efficiency', color='b')
    ax.errorbar(XcoordErrorBars, showerPurities, yerr=showerPurityErrors, fmt="none", capsize=2, color='r')
    ax.errorbar(XcoordErrorBars, showerEfficiencies, yerr=showerEfficiencyErrors, fmt="none", capsize=2, color='g')
    ax.errorbar(XcoordErrorBars, showerPurityEfficiencies, yerr=showerPurityEfficiencyErrors, fmt="none", capsize=2, color='b')
    ax.legend(loc='lower center')
    ax.set_ylim((0.5, 1.01))
    #ax.set_title("Purity/Efficiency vs nHits, Cutoff=%.3f, Tracks" % bestShowerCutoff)
    ax.set_xlabel("Number of hits")
    ax.set_ylabel("Fraction")
    plt.tight_layout()
    plt.savefig("ShowerPurityEfficiencyVsNhits.svg", format='svg', dpi=1200)
    plt.show()