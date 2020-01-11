import math
import numpy as np

def BraggPeak(xCoords, wireCoords, vertex, chargeArray, fraction):
    if len(xCoords) == 0:
        return -1
    xCoordsNew = xCoords - vertex[0]
    wireCoordsNew = wireCoords - vertex[1]
    magnitudes = np.sqrt(np.square(xCoordsNew) + np.square(wireCoordsNew))
    order = magnitudes.argsort()
    orderedChargeArray = chargeArray[order]
    
    frac = math.floor(fraction*len(orderedChargeArray) - 1)
    braggArray = orderedChargeArray[-frac:]
    
    return np.sum(braggArray)/np.sum(chargeArray)

def GetFeatures(pfo, wireViews, fraction = 0.2):
    featureDict = {}
    if wireViews[0]:
        featureDict.update({ "BraggPeakU" : BraggPeak(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, pfo.energyU, fraction)})
    if wireViews[1]:
        featureDict.update({ "BraggPeakV" : BraggPeak(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, pfo.energyV, fraction)})
    if wireViews[2]:
        featureDict.update({ "BraggPeakW" : BraggPeak(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, pfo.energyW, fraction)})
    return featureDict

