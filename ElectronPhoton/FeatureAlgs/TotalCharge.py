import math as m
import numpy as np

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({"TotalChargeU": pfo.energyU.sum()})
    if calculateViews["V"]:
        featureDict.update({"TotalChargeV": pfo.energyV.sum()})
    if calculateViews["W"]:
        featureDict.update({"TotalChargeW": pfo.energyW.sum()})
    if calculateViews["3D"]:
        featureDict.update({"TotalCharge3D": pfo.energy3D.sum()})
    return featureDict