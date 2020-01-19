from sklearn.decomposition import PCA
import numpy as np
import math as m
import matplotlib.pyplot as plt

tolerance = 1e-5
def ZeroCorrect(values):
    values[np.abs(values) < tolerance] = 0

def PcaVariance2D(xCoords, yCoords):
    if len(xCoords) < 3:
        return m.nan, m.nan
    eigenvalues, eigenvectors = Pca((xCoords, yCoords))
    return eigenvalues[0], eigenvalues[0] / eigenvalues[1]

def PcaVariance3D(xCoords, yCoords, zCoords, vertex):
    if len(xCoords) < 3:
        return m.nan, m.nan
    eigenvalues, eigenvectors = Pca((xCoords, yCoords, zCoords), vertex)
    axialVariance = m.sqrt(eigenvalues[0] + eigenvalues[1])
    return axialVariance, axialVariance / m.sqrt(eigenvalues[2])

def PcaReduce(coordSets, intercept = None):
    if len(coordSets[0]) == 0:
        return coordSets
    if type(intercept) == type(None):
        intercept = np.mean(coordSets, axis=1)
    intercept = np.reshape(intercept, (-1, 1))
    if len(coordSets[0]) == 1:
        return coordSets - intercept
    eigenvalues, eigenvectors = Pca(coordSets, intercept)
    reducedCoordSets = np.flip(eigenvectors.transpose(), 0) @ (coordSets - intercept)
    return reducedCoordSets

def Pca(coordSets, intercept = None):
    if type(intercept) == type(None):
        intercept = np.mean(coordSets, axis=1)
    dimensions = len(coordSets)
    ncoords = len(coordSets[0])
    pcamatrix = np.zeros((dimensions, dimensions), dtype='double')
    for i in range(0, dimensions):
        for j in range(i, dimensions):
            pcamatrix[i, j] = pcamatrix[j, i] = np.sum((coordSets[i] - intercept[i]) * (coordSets[j] - intercept[j])) / (ncoords - 1)
    eigenvalues, eigenvectors = np.linalg.eig(pcamatrix)
    ZeroCorrect(eigenvalues)
    order = eigenvalues.argsort()
    return np.real(eigenvalues[order]), np.real(eigenvectors[:,order]) # numpy can sometimes return a complex matrix for no apparent reason!

def GetFeatures(pfo, calculateViews):
    PcaReduce((pfo.driftCoordW, pfo.wireCoordW))
    featureDict = {}
    if calculateViews["U"]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaVarU" : var2d, "PcaRatioU": ratio2d})
    if calculateViews["V"]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "PcaVarV" : var2d, "PcaRatioV": ratio2d})
    if calculateViews["W"]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordW, pfo.wireCoordW)
        featureDict.update({ "PcaVarW" : var2d, "PcaRatioW": ratio2d})
    if calculateViews["3D"]:
        var3d, ratio3d = PcaVariance3D(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D)
        featureDict.update({ "PcaVar3D": var3d, "PcaRatio3D": ratio3d})
    return featureDict