import math as m
import numpy as np

def BraggPeak(xCoords, wireCoords, vertex, chargeArray, fraction):
    if len(xCoords) == 0:
        return m.nan
    xCoordsNew = xCoords - vertex[0]
    wireCoordsNew = wireCoords - vertex[1]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(wireCoordsNew))
    order = magnitudes.argsort()
    orderedChargeArray = chargeArray[order]
    
    frac = m.floor(fraction*len(orderedChargeArray) - 1)
    braggArray = orderedChargeArray[-frac:]
    
    return np.sum(braggArray)/np.sum(chargeArray)

def BraggPeak3D(xCoords, yCoords, zCoords, vertex, chargeArray, fraction):
    if len(xCoords) == 0:
        return m.nan
    xCoordsNew = xCoords - vertex[0]
    yCoordsNew = yCoords - vertex[1]
    zCoordsNew = zCoords - vertex[2]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(yCoordsNew) + np.square(zCoordsNew))
    order = magnitudes.argsort()
    orderedChargeArray = chargeArray[order]

    frac = m.floor(fraction*len(orderedChargeArray) - 1)
    braggArray = orderedChargeArray[-frac:]

    return np.sum(braggArray)/np.sum(chargeArray)



def GetFeatures(pfo, calculateViews, fraction = 0.1):
    featureDict = {}
    braggPeak = m.nan
    if calculateViews["U"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, pfo.energyU, fraction)
        featureDict.update({ "BraggPeakU" : braggPeak})
    if calculateViews["V"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, pfo.energyV, fraction)
        featureDict.update({ "BraggPeakV" : braggPeak})
    if calculateViews["W"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, pfo.energyW, fraction)
        featureDict.update({ "BraggPeakW" : braggPeak})
    if calculateViews["3D"]:
        if pfo.ValidVertex():
            braggPeak = BraggPeak3D(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, pfo.energy3D, fraction)
        featureDict.update({"BraggPeak3D": braggPeak})
    return featureDict

