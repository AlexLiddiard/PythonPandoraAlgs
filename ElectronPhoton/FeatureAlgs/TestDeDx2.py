import numpy as np
import math as m
import PCAnalysis as pca
import matplotlib.pyplot as plt
from NewInitialDeDx import Calculate3dVertex, GetInitialDirection
from UpRootFileReader import ProjectVector

def GetInitialDeDx(pfo, coordSets, vertex, charge, rectWidth, rectLength, scaleFactor = 1):
    reducedCoords = pca.PcaReduce(coordSets, vertex)
    calohits = np.row_stack((reducedCoords, charge))
    lCoordLowerBound = calohits[0] >= 0
    if lCoordLowerBound.sum() < len(calohits[0]) / 2:
        calohits[0] *= -1
        lCoordLowerBound = np.invert(lCoordLowerBound)
    calohits = calohits[:, lCoordLowerBound & (calohits[0] <= rectLength) & (np.abs(calohits[1]) < rectWidth / 2)]
    medianCharge = np.median(calohits[2])
    dedxUnscaled = medianCharge / rectLength * len(calohits[2])
    return dedxUnscaled * scaleFactor

def GetInitialDeDx3D(pfo, coordSets, vertex, charge, cylinderDiameter, cylinderLength):
    reducedCoords = pca.PcaReduce(coordSets, vertex)
    calohits = np.row_stack((reducedCoords[0], np.linalg.norm(reducedCoords[1:], axis=0), charge))
    lCoordLowerBound = calohits[0] >= 0
    if lCoordLowerBound.sum() < len(calohits[0]) / 2:
        calohits[0] *= -1
        lCoordLowerBound = np.invert(lCoordLowerBound)
    calohits = calohits[:, lCoordLowerBound & (calohits[0] <= cylinderLength) & (calohits[1] < cylinderDiameter / 2)]
    medianCharge = np.median(calohits[2])
    return medianCharge / cylinderLength * len(calohits[2])



def GetFeatures(pfo, calculateViews, rectWidth=1, rectLength=4):
    vertex3D = Calculate3dVertex(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, initialLength=4, outlierFraction=0.85)
    if vertex3D is None:
        vertex3D = pfo.vertex3D
    direction3D = GetInitialDirection(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, vertex3D, 4)
    if direction3D is None:
        direction3D = np.array([1, 1, 1]) / m.sqrt(2) # No scaling
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "TestDeDx2U" : GetInitialDeDx(pfo, (pfo.driftCoordU, pfo.wireCoordU), ProjectVector(vertex3D, "U"), pfo.energyU, rectWidth, rectLength, np.linalg.norm(ProjectVector(direction3D, "U")))})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDx2V" : GetInitialDeDx(pfo, (pfo.driftCoordV, pfo.wireCoordV), ProjectVector(vertex3D, "V"), pfo.energyV, rectWidth, rectLength, np.linalg.norm(ProjectVector(direction3D, "V")))})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDx2W" : GetInitialDeDx(pfo, (pfo.driftCoordW, pfo.wireCoordW), ProjectVector(vertex3D, "W"), pfo.energyW, rectWidth, rectLength, np.linalg.norm(ProjectVector(direction3D, "W")))})
    if calculateViews["3D"]:
        featureDict.update({ "TestDeDx2_3D" : GetInitialDeDx3D(pfo, (pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), vertex3D, pfo.energy3D, rectWidth, rectLength )})
    return featureDict