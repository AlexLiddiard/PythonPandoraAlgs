import numpy as np
import math as m
import PCAnalysis as pca
import matplotlib.pyplot as plt

def GetInitialDeDx(pfo, coordSets, vertex, charge, rectWidth, rectLength, dxstep = None):
    reducedCoords = pca.PcaReduce(coordSets, vertex)
    calohits = np.row_stack((reducedCoords, charge))
    lCoordLowerBound = calohits[0] >= 0
    if lCoordLowerBound.sum() < len(calohits[0]) / 2:
        calohits[0] *= -1
        lCoordLowerBound = np.invert(lCoordLowerBound)
    calohits = calohits[:, lCoordLowerBound]
    calohits = calohits[:, calohits[0] <= rectLength]
    calohits = calohits[:, np.abs(calohits[1]) < rectWidth / 2]
    '''
    if abs(pfo.mcPdgCode) == 11:
        plt.axes().set_aspect('equal')
        plt.scatter(reducedCoords[0], reducedCoords[1])
        plt.scatter(calohits[0], calohits[1])
        plt.show()
    '''
    medianCharge = np.median(calohits[2])
    return medianCharge / rectLength * len(calohits[2])


def GetFeatures(pfo, calculateViews, rectWidth=1, rectLength=4):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "TestDeDx2U" : GetInitialDeDx(pfo, (pfo.driftCoordU, pfo.wireCoordU), pfo.vertexU, pfo.energyU, rectWidth, rectLength)})
    if calculateViews["V"]:
        featureDict.update({ "TestDeDx2V" : GetInitialDeDx(pfo, (pfo.driftCoordV, pfo.wireCoordV), pfo.vertexV, pfo.energyV, rectWidth, rectLength)})
    if calculateViews["W"]:
        featureDict.update({ "TestDeDx2W" : GetInitialDeDx(pfo, (pfo.driftCoordW, pfo.wireCoordW), pfo.vertexW, pfo.energyW, rectWidth, rectLength)})
    return featureDict