import AlgorithmConfig as cfg
import numpy as np
import math as m
import PCAnalysis as pca
import matplotlib.pyplot as plt
from PfoVertexing import CalculateShower3DVertex
from NewInitialDeDx import GetInitialDirection
from UpRootFileReader import ProjectVector

def GetInitialDeDx(pfo, coordSets, charge, maxTransVar, vertex, scaleFactor=1):
    if len(coordSets[0]) < 3:
        return
    reducedCoords = pca.PcaReduce(coordSets=coordSets, intercept=vertex, lDirectionCheck=True)

    lPositive = reducedCoords[0] >= 0
    reducedCoords = reducedCoords[:,lPositive]
    charge = charge[lPositive]

    order = reducedCoords[0].argsort()
    reducedCoords = reducedCoords[:,order]
    lineSegmentForwards = GetInitialStraightSegment(reducedCoords, maxTransVar)
    if lineSegmentForwards is None:
        return

    distance = np.linalg.norm(lineSegmentForwards[:,-1] - lineSegmentForwards[:,0])
    medianCharge = np.median(charge[:len(lineSegmentForwards)])
    dedxUnscaled = medianCharge * len(lineSegmentForwards[0]) / distance
    return dedxUnscaled * scaleFactor


def GetInitialStraightSegment(coordSets, maxTransVar):
    length = len(coordSets[0])
    if length < 3:
        return
    coordSets = np.array(coordSets)
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
            transVar = pca.GetEigenValues(sumSquares - sums @ sums.T / count, False)[:-1].sum() / count
    if count > 3:
        return coordSets[:,:count - 1]

def GetFeatures(pfo, calculateViews):
    vertex3D = pfo.vertex3D
    if cfg.newInitialDeDx["calcVertex"]:
        calculatedVertex = CalculateShower3DVertex(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, cfg.vertexCalculation["initialLength"], cfg.vertexCalculation["outlierFraction"])
        if calculatedVertex is not None:
            vertex3D = calculatedVertex
    direction3D = GetInitialDirection((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), vertex3D, 4)
    if direction3D is None:
        direction3D = np.array([1, 1, 1]) / m.sqrt(2) # No scaling
    
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "TestDeDxU" : GetInitialDeDx(pfo, (pfo.driftCoordU, pfo.wireCoordU), pfo.energyU, cfg.testDeDx1["maxRms"]**2, ProjectVector(vertex3D, "U"), np.linalg.norm(ProjectVector(direction3D, "U")))})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDxV" : GetInitialDeDx(pfo, (pfo.driftCoordV, pfo.wireCoordV), pfo.energyV, cfg.testDeDx1["maxRms"]**2, ProjectVector(vertex3D, "V"), np.linalg.norm(ProjectVector(direction3D, "V")))})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDxW" : GetInitialDeDx(pfo, (pfo.driftCoordW, pfo.wireCoordW), pfo.energyW, cfg.testDeDx1["maxRms"]**2, ProjectVector(vertex3D, "W"), np.linalg.norm(ProjectVector(direction3D, "W")))})
    if calculateViews["3D"]:
        featureDict.update({ "TestDeDx3D" : GetInitialDeDx(pfo, (pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.energy3D, cfg.testDeDx1["maxRms"]**2, vertex3D)})
    return featureDict