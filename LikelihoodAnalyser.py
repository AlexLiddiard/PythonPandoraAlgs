import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo
import HistoSynthesis as hs
from itertools import count
import DataSampler as ds
import LikelihoodCalculator as lc

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
            ('Electrons + Positrons', 'abs(mcPdgCode)==11', 'isShower==1', False),
            ('Photons', 'abs(mcPdgCode)==22', 'isShower==1', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Tracks', 'isShower==0', '', True),
            ('Protons', 'abs(mcPdgCode)==2212', 'isShower==0', False),
            ('Muons', 'abs(mcPdgCode)==13', 'isShower==0', False),
            ('Charged Pions', 'abs(mcPdgCode)==211', 'isShower==0', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'track'
    },
)

purityEfficiencyVsCutoffGraph = {'testValues': np.linspace(0, 1, 1001)}
purityEfficiencyBinnedGraphs = (
    {
        "dependence":
        "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(60, 1400, num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1.5, num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "purityW",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "completenessW", 
        'bins': np.linspace(0, 1, num=40), 
        "pfoClass": "both"
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW", 
        'bins': np.linspace(0, 400, num=40), 
        "pfoClass": "shower", 
        "filter": {
            "name": "Electrons",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(0, 800, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "Photons",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "Electrons",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "shower",
        "filter": {
            "name": "Photons",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
)

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
    dfTrackVariable = dfTrackData[variableName]
    dfShowerVariable =  dfShowerData[variableName]
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

def PrintPurityEfficiency(dfTrackData, dfShowerData, predictorName, cutoff, showerCutDirection='right'):
    dfTrackVariable = dfTrackData[predictorName]
    dfShowerVariable =  dfShowerData[predictorName]
    print(
        "Track Efficiency %.3f+-%.3f\n"
        "Track Purity %.3f+-%.3f\n"
        "Track Purity * Efficiency %.3f+-%.3f\n"
        "Shower Efficiency %.3f+-%.3f\n"
        "Shower Purity %.3f+-%.3f\n"
        "Shower Purity * Efficiency %.3f+-%.3f"
    % PurityEfficiency(dfTrackVariable, dfShowerVariable, cutoff, showerCutDirection)
)

def BinnedPurityEfficiency(dfTrackData, dfShowerData, dependenceName, binEdges, predictorName, cutoff, showerCutDirection):
    nBins = len(binEdges) - 1
    results = {
        "track": {
            "efficiency": np.zeros(nBins),
            "efficiencyError": np.zeros(nBins),
            "purity": np.zeros(nBins),
            "purityError": np.zeros(nBins),
            "purityEfficiency": np.zeros(nBins),
            "purityEfficiencyError": np.zeros(nBins)
        },
        "shower": {
            "efficiency": np.zeros(nBins),
            "efficiencyError": np.zeros(nBins),
            "purity": np.zeros(nBins),
            "purityError": np.zeros(nBins),
            "purityEfficiency": np.zeros(nBins),
            "purityEfficiencyError": np.zeros(nBins)
        }
    }
    for lowerBound, upperBound, i in zip(binEdges[:-1], binEdges[1:], count()):
        binFilter = dependenceName + ">=@lowerBound and " + dependenceName + "<@upperBound"
        likelihoodShowersInBin = dfShowerData.query(binFilter)[predictorName]
        likelihoodTracksInBin = dfTrackData.query(binFilter)[predictorName]  
        (
            results["track"]["efficiency"][i],
            results["track"]["efficiencyError"][i],
            results["track"]["purity"][i],
            results["track"]["purityError"][i],
            results["track"]["purityEfficiency"][i],
            results["track"]["purityEfficiencyError"][i],
            results["shower"]["efficiency"][i],
            results["shower"]["efficiencyError"][i],
            results["shower"]["purity"][i],
            results["shower"]["purityError"][i],
            results["shower"]["purityEfficiency"][i],
            results["shower"]["purityEfficiencyError"][i]
        ) = PurityEfficiency(likelihoodTracksInBin, likelihoodShowersInBin, cutoff, showerCutDirection)
    return results

def BinnedPurityEfficiencyPlot(results, binEdges, pfoClass, dependenceName, cutoff, filterName=None, showPurity=True, yLimits=(0, 1.01)):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    hs.WireBarPlot(ax, results[pfoClass]["efficiency"], binEdges, heightErrors=results[pfoClass]["efficiencyError"], colour='g', label="Efficiency")
    if showPurity:
        hs.WireBarPlot(ax, results[pfoClass]["purity"], binEdges, heightErrors=results[pfoClass]["purityError"], colour='r', label="Purity")
        hs.WireBarPlot(ax, results[pfoClass]["purityEfficiency"], binEdges, heightErrors=results[pfoClass]["purityEfficiencyError"], colour='b', label="Purity*Efficiency")
    ax.legend(loc='lower center')
    ax.set_ylim(yLimits)
    ax.set_title("Purity/Efficiency vs %s, Cutoff=%.3f, %s" % (dependenceName, cutoff, filterName if filterName is not None else pfoClass))
    ax.set_xlabel(dependenceName)
    ax.set_ylabel("Fraction")
    fig.tight_layout()
    fig.savefig(filterName if filterName is not None else pfoClass + "PurityEfficiencyVs" + dependenceName + ".svg", format='svg', dpi=1200)
    return fig, ax

if __name__ == "__main__":
    ds.GetPerfPfoData(viewsUsed=lc.viewsUsed)

    print("Testing likelihood using %d tracks and %d showers." % (len(ds.dfPerfPfoData["track"]['union']), len(ds.dfPerfPfoData["shower"]['union'])))
    # Get optimal purity and efficiency
    (
        bestTrackCutoff, trackEfficiencies, trackPurities, trackPurityEfficiencies, 
        bestShowerCutoff, showerEfficiencies, showerPurities, showerPurityEfficiencies
    ) = OptimiseCutoff(ds.dfPerfPfoData["track"]['union'], ds.dfPerfPfoData["shower"]['union'], 'Likelihood', purityEfficiencyVsCutoffGraph['testValues'], 'right')

    # Printing results for optimal purity and efficiency
    print("\nOptimal track cutoff %.3f" % bestTrackCutoff)
    PrintPurityEfficiency(ds.dfPerfPfoData["track"]['union'], ds.dfPerfPfoData["shower"]['union'], 'Likelihood', bestTrackCutoff)
    print("\nOptimal shower cutoff %.3f" % bestShowerCutoff)
    PrintPurityEfficiency(ds.dfPerfPfoData["track"]['union'], ds.dfPerfPfoData["shower"]['union'], 'Likelihood', bestShowerCutoff)

    # Make likelihood histograms.
    for histogram in likelihoodHistograms:
        histogram['name'] = 'Likelihood'
        fig, ax = hs.CreateHistogramWire(ds.dfPerfPfoData['all']['union'], histogram)
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

    lines = bx1.plot(purityEfficiencyVsCutoffGraph['testValues'], trackPurities, 'b', purityEfficiencyVsCutoffGraph['testValues'], trackEfficiencies, 'r', purityEfficiencyVsCutoffGraph['testValues'], trackPurityEfficiencies, 'g')
    bx1.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
    lines = bx2.plot(purityEfficiencyVsCutoffGraph['testValues'], showerPurities, 'b', purityEfficiencyVsCutoffGraph['testValues'], showerEfficiencies, 'r', purityEfficiencyVsCutoffGraph['testValues'], showerPurityEfficiencies, 'g')
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

    for graph in purityEfficiencyBinnedGraphs:
        dfTrackData = ds.dfPerfPfoData["track"]['union']
        dfShowerData = ds.dfPerfPfoData["shower"]["union"]
        filter = graph.get("filter", {})
        query = filter.get("query", None)
        if query is not None:
            dfTrackData = dfTrackData.query(query)
            dfShowerData = dfShowerData.query(query)

        results = BinnedPurityEfficiency(dfTrackData, dfShowerData, graph["dependence"], graph['bins'], "Likelihood", bestShowerCutoff, 'right')
        if graph["pfoClass"] != "shower":
            fig, ax = BinnedPurityEfficiencyPlot(results, graph['bins'], "track", graph["dependence"], bestShowerCutoff, filter.get("name", None), graph.get("showPurity", True))
            plt.show()
        if graph["pfoClass"] != "track":
            fig, ax = BinnedPurityEfficiencyPlot(results, graph['bins'], "shower", graph["dependence"], bestShowerCutoff, filter.get("name", None), graph.get("showPurity", True))
            plt.show()