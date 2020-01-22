import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math as m
from UpRootFileReader import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire
from LikelihoodAnalyser import GraphCutoffLine, OptimiseCutoff, PrintPurityEfficiency
import DataSampler as ds

def GetFeatureView(featureName):
    if featureName.endswith("3D"):
        return "3D"
    if featureName[-1] in ["U", "V", "W"]:
        return featureName[-1]
    else:
        return "intersection"


def PlotFeatureHistogram(dfPfoData, classNames, feature, featureHistogram, bestCutoff = None):
    feature['filters'] = featureHistogram["filters"]
    feature['yAxis'] = 'log'
    fig, ax = CreateHistogramWire(dfPfoData, feature)
    if bestCutoff is not None:
        GraphCutoffLine(ax, classNames, bestCutoff, True, feature['cutDirection'] == 'left')
    plt.savefig('%s distribution for %s' % (feature['name'], ', '.join((filter[0] for filter in feature['filters'])) + '.svg'), format='svg', dpi=1200)
    plt.show()

def GetBestPurityEfficiency(dfClass0Data, dfClass1Data, classNames, feature, nTestCuts):
    cutoffValues = np.linspace(feature["bins"][0], feature["bins"][-1], nTestCuts)
    cutoffResults = OptimiseCutoff(dfClass0Data, dfClass1Data, feature['name'], cutoffValues, feature['cutDirection'])

    # Printing results for optimal purity and efficiency
    print("Performance results for %s:" % feature['name'])
    print("\nOptimal %s cutoff %.3f" % (classNames[0], cutoffResults[0]))
    PrintPurityEfficiency(dfClass0Data, dfClass1Data, classNames, feature['name'], cutoffResults[0], feature['cutDirection'])
    print("\nOptimal %s cutoff %.3f" % (classNames[1], cutoffResults[4]))
    PrintPurityEfficiency(dfClass0Data, dfClass1Data, classNames, feature['name'], cutoffResults[4], feature['cutDirection'])
    return cutoffValues, cutoffResults

def PlotPurityEfficiencyVsCutoff(featureName, classNames, cutoffValues, cutoffResults):
        fig = plt.figure(figsize=(20,7.5))
        bx1 = fig.add_subplot(1,2,1)
        bx2 = fig.add_subplot(1,2,2)
        lines = bx1.plot(cutoffValues, cutoffResults[1], 'b', cutoffValues, cutoffResults[2], 'r', cutoffValues, cutoffResults[3], 'g')
        bx1.legend(lines, ('Efficiency', 'Purity', 'Purity * Efficiency'), loc='lower center')
        lines = bx2.plot(cutoffValues, cutoffResults[5], 'b', cutoffValues, cutoffResults[6], 'r', cutoffValues, cutoffResults[7], 'g')
        bx2.legend(lines, ('Efficiency', 'Purity', 'Purity * Efficiency'), loc='lower center')
        GraphCutoffLine(bx1, classNames, cutoffResults[0])
        GraphCutoffLine(bx2, classNames, cutoffResults[4])

        bx1.set_ylim([0, 1])
        bx1.set_title("Purity/Efficiency vs %s Cutoff — %s" % (featureName, classNames[0]))
        bx1.set_xlabel(featureName + " Cutoff")
        bx1.set_ylabel("Fraction")

        bx2.set_ylim([0, 1])
        bx2.set_title("Purity/Efficiency vs %s Cutoff — %s" % (featureName, classNames[1]))
        bx2.set_xlabel(featureName + " Cutoff")
        bx2.set_ylabel("Fraction")

        plt.tight_layout()
        plt.savefig("PurityEfficiencyVs%sCutoff.svg" % featureName, format='svg', dpi=1200)
        plt.show()

def CorrelationMatrix(featureNames, viewsUsed, preFilters, pfoData):
    dataCorrFilters = [preFilters[x] for x in viewsUsed]
    dfPfoDataCorr = pfoData["all"]["general"].query("(" + ") and (".join(dataCorrFilters) + ")")
    rMatrix = dfPfoDataCorr[featureNames].corr()
    rSquaredMatrix = rMatrix * rMatrix
    sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 20}, cmap="Blues")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
    plt.show()

#dataCorrFilters = [ds.performancePreFilters[x] for x in ds.GetViewsUsed(features)]
#featureNames = [feature["name"]]
#dfPfoDataCorr = ds.dfPerfPfoData["all"]["general"].query("(" + ") and (".join(dataCorrFilters) + ")")
#rMatrix = ds.dfPerfPfoData["all"]["general"][[feature["name"] for feature in features]].corr()
#rSquaredMatrix = rMatrix * rMatrix
#sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 20}, cmap="Blues")
#plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
#plt.tight_layout()
#plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
#plt.show()