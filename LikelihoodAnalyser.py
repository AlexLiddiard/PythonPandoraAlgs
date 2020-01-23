import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo
import HistoSynthesis as hs
from itertools import count
import DataSampler as ds
import LikelihoodCalculator as lc
from FeatureAnalyser import GetBestPurityEfficiency, PlotVariableHistogram, PlotPurityEfficiencyVsCutoff, PurityEfficiency

def BinnedPurityEfficiency(dfClass0Data, dfClass1Data, classNames, dependenceName, binEdges, predictorName, cutoff, class1CutDirection):
    nBins = len(binEdges) - 1
    results = {
        classNames[0]: {
            "efficiency": np.zeros(nBins),
            "efficiencyError": np.zeros(nBins),
            "purity": np.zeros(nBins),
            "purityError": np.zeros(nBins),
            "purityEfficiency": np.zeros(nBins),
            "purityEfficiencyError": np.zeros(nBins)
        },
        classNames[1]: {
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
        likelihoodClass1InBin = dfClass1Data.query(binFilter)[predictorName]
        likelihoodClass0InBin = dfClass0Data.query(binFilter)[predictorName]  
        (
            results[classNames[0]]["efficiency"][i],
            results[classNames[0]]["efficiencyError"][i],
            results[classNames[0]]["purity"][i],
            results[classNames[0]]["purityError"][i],
            results[classNames[0]]["purityEfficiency"][i],
            results[classNames[0]]["purityEfficiencyError"][i],
            results[classNames[1]]["efficiency"][i],
            results[classNames[1]]["efficiencyError"][i],
            results[classNames[1]]["purity"][i],
            results[classNames[1]]["purityError"][i],
            results[classNames[1]]["purityEfficiency"][i],
            results[classNames[1]]["purityEfficiencyError"][i]
        ) = PurityEfficiency(likelihoodClass0InBin, likelihoodClass1InBin, cutoff, class1CutDirection)
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

def PlotPurityEfficiencyVsVariable(dfClass0, dfClass1, classNames, graph):
    filter = graph.get("filter", {})
    query = filter.get("query", None)
    if query is not None:
        dfClass0 = dfClass0.query(query)
        dfClass1 = dfClass1.query(query)

    results = BinnedPurityEfficiency(dfClass0, dfClass1, classNames, graph["dependence"], graph['bins'], "Likelihood", graph["cutoff"], 'right')
    if graph["pfoClass"] != classNames[1]:
        fig, ax = BinnedPurityEfficiencyPlot(results, graph['bins'], classNames[0], graph["dependence"], graph["cutoff"], filter.get("name", None), graph.get("showPurity", True))
        plt.show()
    if graph["pfoClass"] != classNames[0]:
        fig, ax = BinnedPurityEfficiencyPlot(results, graph['bins'], classNames[1], graph["dependence"], graph["cutoff"], filter.get("name", None), graph.get("showPurity", True))
        plt.show()