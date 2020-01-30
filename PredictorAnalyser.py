import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo
import HistoSynthesis as hs
from itertools import count
import DataSampler as ds
from FeatureAnalyser import GetBestPurityEfficiency, PlotVariableHistogram, PlotPurityEfficiencyVsCutoff, PurityEfficiency, PrintPurityEfficiency
import BaseConfig as bc
import GeneralConfig as gc
import PredictorAnalyserConfig as pac
import importlib
pcc = importlib.import_module(pac.predictor["algorithmName"] + "Config") # Get config for the predictor calculation, e.g. LikelihoodCalculatorConfig

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
        predictorClass1InBin = dfClass1Data.query(binFilter)[predictorName]
        predictorClass0InBin = dfClass0Data.query(binFilter)[predictorName]  
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
        ) = PurityEfficiency(predictorClass0InBin, predictorClass1InBin, cutoff, class1CutDirection)
    return results

def BinnedPurityEfficiencyPlot(ax, results, binEdges, pfoClass, dependenceName, cutoff, filterName=None, showPurity=True, yLimits=(0, 1.01)):
    hs.WireBarPlot(ax, results[pfoClass]["efficiency"], binEdges, heightErrors=results[pfoClass]["efficiencyError"], colour='g', label="Efficiency")
    if showPurity:
        hs.WireBarPlot(ax, results[pfoClass]["purity"], binEdges, heightErrors=results[pfoClass]["purityError"], colour='r', label="Purity")
        hs.WireBarPlot(ax, results[pfoClass]["purityEfficiency"], binEdges, heightErrors=results[pfoClass]["purityEfficiencyError"], colour='b', label="Purity*Efficiency")
    ax.legend(loc='lower center', framealpha=0.5)
    ax.set_ylim(yLimits)
    ax.set_title("Purity/Efficiency vs %s\nCutoff=%.3f, %s%s classification" % (dependenceName, cutoff, filterName + " " if filterName is not None else "", pfoClass))
    ax.set_xlabel(dependenceName)
    ax.set_ylabel("Fraction")

def PlotPurityEfficiencyVsVariable(dfClass0, dfClass1, classNames, predictor, graph):
    if predictor["cutDirection"] == "right":
        inequalities = ("<", ">")
    else:
        inequalities = (">", "<")

    filter = graph.get("filter", {})
    query = filter.get("query", None)
    if query is not None:
        dfClass0 = dfClass0.query(query)
        dfClass1 = dfClass1.query(query)
        # Print overall results for filtered data
        print("\nOverall efficiency/purity for " + filter["name"] + " classification:")
        PrintPurityEfficiency(dfClass0, dfClass1, gc.classNames, predictor["name"], graph['cutoff'], predictor["cutDirection"], graph.get("showPurity", True), graph["pfoClass"])
    results = BinnedPurityEfficiency(dfClass0, dfClass1, classNames, graph["dependence"], graph['bins'], predictor["name"], graph["cutoff"], predictor["cutDirection"])
    if graph["pfoClass"] != classNames[1]:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 15))
        BinnedPurityEfficiencyPlot(ax[0], results, graph['bins'], classNames[0], graph["dependence"], graph["cutoff"], filter.get("name", None), graph.get("showPurity", True))
        hs.CreateHistogramWire(ax[1], dfClass0, 
            {
                "name": graph["dependence"], 
                "bins": graph['bins'], 
                "yAxis":"log", 
                "filters": (
                    (filter.get("name", "") +  " " + classNames[0], "", "count", True),
                    ("correct " + filter.get("name", "") +  " " + classNames[0], predictor["name"] + inequalities[0] + str(graph['cutoff']), "count", False)
                )
            })
        if graph.get("showPurity", True):
            hs.CreateHistogramWire(ax[1], dfClass1, 
            {
                "name": graph["dependence"],
                "bins": graph['bins'],
                "yAxis":"log",
                "filters": (
                    ("incorrect " + filter.get("name", "") +  " " + classNames[1], predictor["name"] + inequalities[0] + str(graph['cutoff']), "count", False),
                )
            })
        fig.savefig(filter.get("name", "") + classNames[0] + "PurityEfficiencyVs" + graph["dependence"] + ".svg", format='svg', dpi=1200)
        plt.show()
    if graph["pfoClass"] != classNames[0]:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 15))
        BinnedPurityEfficiencyPlot(ax[0], results, graph['bins'], classNames[1], graph["dependence"], graph["cutoff"], filter.get("name", None), graph.get("showPurity", True))
        hs.CreateHistogramWire(ax[1], dfClass1, 
            {
                "name": graph["dependence"], 
                "bins": graph['bins'], 
                "yAxis":"log", 
                "filters": (
                    (filter.get("name", "") +  " " + classNames[1], "", "count", True),
                    ("correct " + filter.get("name", "") +  " " + classNames[1], predictor["name"] + inequalities[1] + str(graph['cutoff']), "count", False)
                )
            })
        if graph.get("showPurity", True):
            hs.CreateHistogramWire(ax[1], dfClass0, 
            {
                "name": graph["dependence"],
                "bins": graph['bins'],
                "yAxis":"log",
                "filters": (
                    ("incorrect " + filter.get("name", "") +  " " + classNames[0], predictor["name"] + inequalities[1] + str(graph['cutoff']), "count", False),
                )
            })
        fig.savefig(filter.get("name", "") + classNames[1] + "PurityEfficiencyVs" + graph["dependence"] + ".svg", format='svg', dpi=1200)
        plt.show()

if __name__ == "__main__":
    pcc.features.append(pac.predictor)
    ds.LoadPfoData(pcc.features)
    dfPerfDataAll = ds.GetFilteredPfoData("performance", "all", "performance", "union")
    dfPerfDataClass0 = ds.GetFilteredPfoData("performance", gc.classNames[0], "performance", "union")
    dfPerfDataClass1 = ds.GetFilteredPfoData("performance", gc.classNames[1], "performance", "union")
    pac.predictor["bins"] = pac.predictor["range"]
    cutoffValues, cutoffResults = GetBestPurityEfficiency(dfPerfDataClass0, dfPerfDataClass1, pac.predictor, pac.purityEfficiencyVsCutoffGraph['nTestCuts'])

    for histogram in pac.predictorHistograms:
        histogram['name'] = pac.predictor["name"]
        PlotVariableHistogram(dfPerfDataAll, gc.classNames, pac.predictor, histogram, cutoffResults[4])

    PlotPurityEfficiencyVsCutoff("Likelihood", gc.classNames, cutoffValues, cutoffResults)
    
    for graph in pac.purityEfficiencyBinnedGraphs:
        graph["cutoff"] = cutoffResults[4]
        PlotPurityEfficiencyVsVariable(dfPerfDataClass0, dfPerfDataClass1, gc.classNames, pac.predictor, graph)