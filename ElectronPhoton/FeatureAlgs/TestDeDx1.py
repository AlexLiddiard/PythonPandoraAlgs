import numpy as np
import math as m
import PCAnalysis as pca
import matplotlib.pyplot as plt

def GetInitialDeDx(pfo, coordSets, charge, maxTransVar, vertex=None):
    if len(coordSets[0]) < 3:
        return
    reducedCoords = pca.PcaReduce(coordSets, vertex)

    if vertex is not None:
        lCoordLowerBound = reducedCoords[0] >= 0
        if lCoordLowerBound.sum() < len(reducedCoords[0]) / 2:
            reducedCoords[0] *= -1
            lCoordLowerBound = np.invert(lCoordLowerBound)
        reducedCoords = reducedCoords[:,lCoordLowerBound]
        charge = charge[lCoordLowerBound]

    order = reducedCoords[0].argsort()
    reducedCoords = reducedCoords[:,order]
    lineSegmentForwards = GetInitialStraightSegment(reducedCoords, maxTransVar)

    if (vertex is None) and ((lineSegmentForwards is None) or (len(lineSegmentForwards[0]) < len(coordSets[0]))):
        reducedCoords = np.flip(reducedCoords, axis=1)
        lineSegmentBackwards = GetInitialStraightSegment(reducedCoords, maxTransVar)
        if (
            (lineSegmentForwards is None) or 
            ((lineSegmentBackwards is not None) and len(lineSegmentBackwards[0]) > len(lineSegmentForwards[0]))
        ):
            charge = np.flip(charge)
            lineSegmentForwards = lineSegmentBackwards
    
    if lineSegmentForwards is None:
        return

    distance = np.linalg.norm(lineSegmentForwards[:,-1] - lineSegmentForwards[:,0])
    medianCharge = np.median(charge[:len(lineSegmentForwards)])

    '''
    if abs(pfo.mcPdgCode) == 11:
        plt.axes().set_aspect('equal')
        plt.scatter(reducedCoords[0], reducedCoords[1])
        plt.scatter(lineSegmentForwards[0], lineSegmentForwards[1])
        plt.show()
    '''
    return medianCharge * len(lineSegmentForwards) / distance


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
        return coordSets[:,:count - 1]

def GetFeatures(pfo, calculateViews, maxRms=0.15):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "TestDeDxU" : GetInitialDeDx(pfo, (pfo.driftCoordU, pfo.wireCoordU), pfo.energyU, maxRms * maxRms, pfo.vertexU)})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDxV" : GetInitialDeDx(pfo, (pfo.driftCoordV, pfo.wireCoordV), pfo.energyV, maxRms * maxRms, pfo.vertexV)})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDxW" : GetInitialDeDx(pfo, (pfo.driftCoordW, pfo.wireCoordW), pfo.energyW, maxRms * maxRms, pfo.vertexW)})
    if calculateViews["3D"]:
        featureDict.update({ "TestDeDx3D" : GetInitialDeDx(pfo, (pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.energy3D, maxRms * maxRms, pfo.vertex3D)})
    return featureDict