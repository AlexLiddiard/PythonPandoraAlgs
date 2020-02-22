import BaseConfig as bc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo
import HistoSynthesis as hs
from itertools import count
import DataSampler as ds
from FeatureAnalyser import GetBestPurityEfficiency, PlotVariableHistogram, PlotPurityEfficiencyVsCutoff, PurityEfficiency, PrintPurityEfficiency
import GeneralConfig as gc
import PredictorAnalyserConfig as pac
import importlib
from OpenPickledFigure import SaveFigure

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

def BinnedPurityEfficiencyPlot(ax, results, binEdges, pfoClass, dependenceName, cutoff, predictorName, filterName=None, showPurity=True, yLimits=(0, 1.01)):
    hs.WireBarPlot(ax, results[pfoClass]["efficiency"], binEdges, heightErrors=results[pfoClass]["efficiencyError"], label=predictorName + " Efficiency")
    if showPurity:
        hs.WireBarPlot(ax, results[pfoClass]["purity"], binEdges, heightErrors=results[pfoClass]["purityError"], label=predictorName + " Purity")
        hs.WireBarPlot(ax, results[pfoClass]["purityEfficiency"], binEdges, heightErrors=results[pfoClass]["purityEfficiencyError"], label=predictorName + "Purity*Efficiency")
    ax.legend(loc='lower center', framealpha=0.5)
    ax.set_ylim(yLimits)
    #ax.set_title("Purity/Efficiency vs %s\nCutoff=%.3f, %s%s classification" % (dependenceName, cutoff, filterName + " " if filterName is not None else "", pfoClass))
    ax.set_xlabel(dependenceName)
    ax.set_ylabel("Fraction")

def PlotPurityEfficiencyVsVariable(dfClass0, dfClass1, classNames, predictors, graph):
    # Filter data
    filter = graph.get("filter", {})
    query = filter.get("query", None)
    if query is not None:
        dfClass0 = dfClass0.query(query)
        dfClass1 = dfClass1.query(query) 
        # Print overall results for filtered data
        for predictor in predictors:
            print(predictor["name"])
            print("\nOverall " + predictor["name"] + " efficiency/purity for " + filter["name"] + " classification:")
            PrintPurityEfficiency(dfClass0, dfClass1, gc.classNames, predictor["name"], predictor['cutoff'], predictor["cutDirection"], graph.get("showPurity", True), graph["pfoClass"])
    
    # Get binned results
    allResults = []
    for predictor in predictors:
        allResults.append(BinnedPurityEfficiency(dfClass0, dfClass1, classNames, graph["dependence"], graph['bins'], predictor["name"], predictor["cutoff"], predictor["cutDirection"]))
    
    # Graphs!
    if graph["pfoClass"] != classNames[1]:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 15))
        filters0 = [(filter.get("name", "") +  " " + classNames[0], "", "count", True)]
        filters1 = [(filter.get("name", "") +  " " + classNames[1], "", "count", True)]
        for predictor, results in zip(predictors, allResults):
            inequalities = ("<", ">") if predictor["cutDirection"] == "right" else (">", "<")
            BinnedPurityEfficiencyPlot(ax[0], results, graph['bins'], classNames[0], graph["dependence"], predictor["cutoff"], predictor["name"], filter.get("name", None), graph.get("showPurity", True))
            filters0.append((predictor["name"] + " correct " + filter.get("name", "") +  " " + classNames[0], predictor["name"] + inequalities[0] + str(predictor['cutoff']), "count", False))
            if graph.get("showPurity", True):
                filters1.append((predictor["name"] + " incorrect " + filter.get("name", "") +  " " + classNames[1], predictor["name"] + inequalities[0] + str(predictor['cutoff']), "count", False))  
        
        hs.CreateHistogramWire(ax[1], dfClass0, 
            {
                "name": graph["dependence"], 
                "bins": graph['bins'], 
                "yAxis":"log", 
                "filters": filters0
            })
        if graph.get("showPurity", True):
            hs.CreateHistogramWire(ax[1], dfClass1, 
            {
                "name": graph["dependence"],
                "bins": graph['bins'],
                "yAxis":"log",
                "filters": filters1
            })
        SaveFigure(fig, bc.figureFolderFull + "/" + filter.get("name", "") + classNames[0] + "PurityEfficiencyVs" + graph["dependence"] + ".pickle")
        plt.show()
    if graph["pfoClass"] != classNames[0]:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 15))
        filters0 = [(filter.get("name", "") +  " " + classNames[0], "", "count", True)]
        filters1 = [(filter.get("name", "") +  " " + classNames[1], "", "count", True)]
        for predictor, results in zip(predictors, allResults):
            inequalities = ("<", ">") if predictor["cutDirection"] == "right" else (">", "<")
            BinnedPurityEfficiencyPlot(ax[0], results, graph['bins'], classNames[1], graph["dependence"], predictor["cutoff"], predictor["name"], filter.get("name", None), graph.get("showPurity", True))
            filters1.append((predictor["name"] + " correct " + filter.get("name", "") +  " " + classNames[1], predictor["name"] + inequalities[1] + str(predictor['cutoff']), "count", False))
            if graph.get("showPurity", True):
                filters0.append((predictor["name"] + " incorrect " + filter.get("name", "") +  " " + classNames[0], predictor["name"] + inequalities[1] + str(predictor['cutoff']), "count", False))  
        
        hs.CreateHistogramWire(ax[1], dfClass1, 
            {
                "name": graph["dependence"], 
                "bins": graph['bins'], 
                "yAxis":"log", 
                "filters": filters1
            })
        if graph.get("showPurity", True):
            hs.CreateHistogramWire(ax[1], dfClass0, 
            {
                "name": graph["dependence"],
                "bins": graph['bins'],
                "yAxis":"log",
                "filters": filters0
            })
        SaveFigure(fig, bc.figureFolderFull + "/" + filter.get("name", "") + classNames[1] + "PurityEfficiencyVs" + graph["dependence"] + ".pickle")
        plt.show()

if __name__ == "__main__":
    ds.LoadPfoData()
    dfPerfDataAll = ds.GetFilteredPfoData("performance", "all", "performance", "union")
    dfPerfDataClass0 = ds.GetFilteredPfoData("performance", gc.classNames[0], "performance", "union")
    dfPerfDataClass1 = ds.GetFilteredPfoData("performance", gc.classNames[1], "performance", "union")

    for predictor in pac.predictors:
        fixedCutoff = predictor.get("fixedCutoff", None)
        nTestCuts = pac.purityEfficiencyVsCutoffGraph['nTestCuts']
        if fixedCutoff is not None:
            print("Fixed cutoff specified, overriding purity * efficiency optimisation.")
            predictor["bins"] = [fixedCutoff, fixedCutoff]
            nTestCuts = 1
        cutoffValues, cutoffResults = GetBestPurityEfficiency(dfPerfDataClass0, dfPerfDataClass1, predictor, nTestCuts)
        for histogram in pac.predictorHistograms:
            histogram['name'] = predictor["name"]
            PlotVariableHistogram(dfPerfDataAll, gc.classNames, predictor, histogram, cutoffResults[4])
        PlotPurityEfficiencyVsCutoff("Likelihood", gc.classNames, cutoffValues, cutoffResults)
        predictor["cutoff"] = cutoffResults[4]
    
    for graph in pac.purityEfficiencyBinnedGraphs:
        PlotPurityEfficiencyVsVariable(dfPerfDataClass0, dfPerfDataClass1, gc.classNames, pac.predictors, graph)