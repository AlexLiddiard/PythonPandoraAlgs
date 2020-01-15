import statistics
import numpy as np
import math as m

def GetPlaneChargeStdMeanRatio(energyArray):
    
    if len(energyArray) < 2:
        return m.nan

    return np.std(energyArray)/np.mean(energyArray)

def GetFeatures(pfo, calculateViews, binWidth=20, minBins=3):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "ChargedStdMeanRatioU" : GetPlaneChargeStdMeanRatio(pfo.energyU)})
    if calculateViews["V"]:
        featureDict.update({ "ChargedStdMeanRatioV" : GetPlaneChargeStdMeanRatio(pfo.energyV)})
    if calculateViews["W"]:
        featureDict.update({ "ChargedStdMeanRatioW" : GetPlaneChargeStdMeanRatio(pfo.energyW)})
    if calculateViews["3D"]:
        featureDict.update({ "ChargedStdMeanRatio3D" : GetPlaneChargeStdMeanRatio(pfo.energy3D)})
    return featureDict