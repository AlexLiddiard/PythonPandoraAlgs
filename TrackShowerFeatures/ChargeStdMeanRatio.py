import statistics
import numpy as np

def GetPlaneChargeStdMeanRatio(energyArray):
    if len(energyArray) < 2:
        return -1

    return statistics.stdev(energyArray)/np.mean(energyArray)

def GetFeatures(pfo, wireViews, binWidth=20, minBins=3):
    featureDict = {}
    if wireViews[0]:
        featureDict.update({ "ChargedStdMeanRatioU" : GetPlaneChargeStdMeanRatio(pfo.energyU)})
    if wireViews[1]:
        featureDict.update({ "ChargedStdMeanRatioV" : GetPlaneChargeStdMeanRatio(pfo.energyV)})
    if wireViews[2]:
        featureDict.update({ "ChargedStdMeanRatioW" : GetPlaneChargeStdMeanRatio(pfo.energyW)})
    return featureDict