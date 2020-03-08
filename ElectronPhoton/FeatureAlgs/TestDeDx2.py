import AlgorithmConfig as cfg
import numpy as np
import math as m
import PCAnalysis as pca
import matplotlib.pyplot as plt
from PfoVertexing import CalculateShower3DVertex
from NewInitialDeDx import GetInitialDirection, GetHitsInRadius
from UpRootFileReader import ProjectVector

def GetInitialDeDx(coordSets, charge, vertex, radius, rectWidth, rectLength, scaleFactor = 1):
    if len(coordSets[0]) <= 1:
        return np.nan
    coordSets = np.array(coordSets)

    # Find initial direction and change basis
    filter = GetHitsInRadius(coordSets=coordSets, centre=vertex, radius=rectLength)
    if filter.sum() <= 1:
        return np.nan
    newBasis = pca.Pca(coordSets=coordSets[:,filter], intercept=vertex, withVectors=True)[1]
    reducedCoords = pca.ChangeCoordBasis(coordSets=coordSets, basisVectors=np.flip(newBasis, 1), normed=True, preTranslation=-vertex)
    if not pca.CorrectDirection(reducedCoords[0]):
        reducedCoords[0] *= -1

    # Find hits in rectangle
    filter = (reducedCoords[0] >= 0) & (reducedCoords[0] <= rectLength) & (np.linalg.norm(reducedCoords[1:], axis=0) <= rectWidth / 2)
    nHits = filter.sum()
    if nHits == 0:
        return np.nan

    # Make dE/dx estimate
    medianCharge = np.median(charge[filter])
    dedxUnscaled = medianCharge * nHits / rectLength
    return dedxUnscaled * scaleFactor

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
        featureDict.update({ "TestDeDx2U" : GetInitialDeDx((pfo.driftCoordU, pfo.wireCoordU), pfo.energyU, ProjectVector(vertex3D, "U"), cfg.testDeDx2["initialDirectionRadius"], cfg.testDeDx2["rectangleWidth"], cfg.testDeDx2["rectangleLength"], np.linalg.norm(ProjectVector(direction3D, "U")))})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDx2V" : GetInitialDeDx((pfo.driftCoordV, pfo.wireCoordV), pfo.energyV, ProjectVector(vertex3D, "V"), cfg.testDeDx2["initialDirectionRadius"], cfg.testDeDx2["rectangleWidth"], cfg.testDeDx2["rectangleLength"], np.linalg.norm(ProjectVector(direction3D, "V")))})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDx2W" : GetInitialDeDx((pfo.driftCoordW, pfo.wireCoordW), pfo.energyW, ProjectVector(vertex3D, "W"), cfg.testDeDx2["initialDirectionRadius"], cfg.testDeDx2["rectangleWidth"], cfg.testDeDx2["rectangleLength"], np.linalg.norm(ProjectVector(direction3D, "W")))})
    if calculateViews["3D"]:
        featureDict.update({ "TestDeDx2_3D" : GetInitialDeDx((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.energy3D, vertex3D, cfg.testDeDx2["initialDirectionRadius"], cfg.testDeDx2["rectangleWidth"], cfg.testDeDx2["rectangleLength"] )})
    return featureDict