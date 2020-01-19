import math as m
import numpy as np
import TrackShowerFeatures.PCAnalysis as pca

def MoliereRadius(xCoord, yCoord, zCoord, chargeArray, fraction):
    reducedCoordSets = pca.PcaReduce((xCoord, yCoord, zCoord))
    tCoordSets = reducedCoordSets[1:]
    tDistance = np.linalg.norm(tCoordSets, axis=0)
    sort = tDistance.argsort()
    totalChargeFraction = np.sum(chargeArray) * fraction
    runningTotalCharge = 0
    for i in sort:
        if runningTotalCharge < totalChargeFraction:
            runningTotalCharge += chargeArray[i]
            moliereRadius = tDistance[i]
        else:
            break
    #return moliereRadius / np.sum(chargeArray)
    length = max(reducedCoordSets[0]) - min(reducedCoordSets[0])
    if length == 0:
        return m.nan
    return moliereRadius / length

def GetFeatures(pfo, calculateViews, fraction = 0.4):
    featureDict = {}
    moliere = m.nan
    if  calculateViews["3D"]:
        if pfo.ValidVertex():
            moliere =  MoliereRadius(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.energy3D, fraction)
        featureDict.update({ "Moliere3D" : moliere})
    return featureDict

