import numpy as np
import math as m
import PCAnalysis as pca

def GetInitialDeDx(coordSets, charge, maxTransVar):
    if len(coordSets[0]) < 3:
        return
    reducedCoords = pca.PcaReduce(coordSets)
    order = reducedCoords[0].argsort()
    reducedCoords = reducedCoords[:,order]
    lineSegmentForwards = GetInitialStraightSegment(reducedCoords, maxTransVar)
    if (lineSegmentForwards is None) or (len(lineSegmentForwards[0]) < len(coordSets[0])):
        reducedCoords = np.flip(reducedCoords, axis=1)
        lineSegmentBackwards = GetInitialStraightSegment(reducedCoords, maxTransVar)
        if lineSegmentBackwards is None:
            return
        if (lineSegmentForwards is None) or (len(lineSegmentBackwards[0]) > len(lineSegmentForwards[0])):
            charge = np.flip(charge)
            lineSegmentForwards = lineSegmentBackwards 
    distance = np.linalg.norm(lineSegmentForwards[:,-1] - lineSegmentForwards[:,0])
    charge = charge[:len(lineSegmentForwards)].sum()
    return charge / distance


def GetInitialStraightSegment(coordSets, maxTransVar):
    length = len(coordSets[0])
    if length < 3:
        return
    dimensions = len(coordSets)
    sums = np.zeros((dimensions, 1), dtype='double')
    sumSquares = np.zeros((dimensions, dimensions), dtype='double')
    transVar = 0
    count = 0
    while (maxTransVar >= transVar) and (count < length):
        point = coordSets[:, count].reshape(dimensions, 1)
        sums += point
        sumSquares += point @ point.T
        count += 1
        if count >= 3:
            transVar = pca.GetEigenValues(sumSquares - sums @ sums.T / (count + 1), False)[:-1].sum() / (count + 1)
    if count > 3:
        return coordSets[:,:count]

def GetFeatures(pfo, calculateViews, maxRms=0.3):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "TestDeDxU" : GetInitialDeDx((pfo.driftCoordU, pfo.wireCoordU), pfo.energyU, maxRms * maxRms)})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDxV" : GetInitialDeDx((pfo.driftCoordV, pfo.wireCoordV), pfo.energyV, maxRms * maxRms)})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDxW" : GetInitialDeDx((pfo.driftCoordW, pfo.wireCoordW), pfo.energyW, maxRms * maxRms)})
    if calculateViews["3D"]:
        featureDict.update({ "TestDeDx3D" : GetInitialDeDx((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.energyU, maxRms * maxRms)})
    return featureDict