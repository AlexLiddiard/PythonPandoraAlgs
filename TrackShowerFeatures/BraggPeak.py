import math as m
import numpy as np

def BraggPeak(xCoords, wireCoords, vertex, chargeArray, startFraction, endFraction):
    if len(xCoords) == 0:
        return m.nan
    xCoordsNew = xCoords - vertex[0]
    wireCoordsNew = wireCoords - vertex[1]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(wireCoordsNew))
    order = magnitudes.argsort()
    orderedChargeArray = chargeArray[order]
    
    fracEnd = m.floor(endFraction*len(orderedChargeArray) - 1)
    fracStart = m.floor(startFraction*len(orderedChargeArray))

    if fracStart == 0:
        return m.nan

    braggEndArray = orderedChargeArray[-fracEnd:]
    braggStartArray = orderedChargeArray[:fracStart]
    
    return np.sum(braggEndArray)/np.sum(braggStartArray)

def BraggPeak3D(xCoords, yCoords, zCoords, vertex, chargeArray, startFraction, endFraction):
    if len(xCoords) == 0:
        return m.nan
    xCoordsNew = xCoords - vertex[0]
    yCoordsNew = yCoords - vertex[1]
    zCoordsNew = zCoords - vertex[2]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(yCoordsNew) + np.square(zCoordsNew))
    order = magnitudes.argsort()
    orderedChargeArray = chargeArray[order]

    fracEnd = m.floor(endFraction*len(orderedChargeArray) - 1)
    fracStart = m.floor(startFraction*len(orderedChargeArray))

    if fracStart == 0:
        return m.nan

    braggEndArray = orderedChargeArray[-fracEnd:]
    braggStartArray = orderedChargeArray[:fracStart]

    return np.sum(braggEndArray)/np.sum(braggStartArray)



def GetFeatures(pfo, calculateViews, startFraction = 0.8, endFraction = 0.1):
    featureDict = {}
    braggPeak = m.nan
    if calculateViews["U"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, pfo.energyU, startFraction, endFraction)
        featureDict.update({ "BraggPeakU" : braggPeak})
    if calculateViews["V"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, pfo.energyV, startFraction, endFraction)
        featureDict.update({ "BraggPeakV" : braggPeak})
    if calculateViews["W"]:
        if pfo.ValidVertex():
            braggPeak =  BraggPeak(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, pfo.energyW, startFraction, endFraction)
        featureDict.update({ "BraggPeakW" : braggPeak})
    if calculateViews["3D"]:
        if pfo.ValidVertex():
            braggPeak = BraggPeak3D(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, pfo.energy3D, startFraction, endFraction)
        featureDict.update({"BraggPeak3D": braggPeak})
    return featureDict

