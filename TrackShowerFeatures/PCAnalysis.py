from sklearn.decomposition import PCA
import numpy as np
import TrackShowerFeatures.HitBinning as hb
import math as m
import matplotlib.pyplot as plt

tolerance = 1e-5
def ZeroCorrect(values):
    values[np.abs(values) < tolerance] = 0

def PcaVariance2D(xCoords, yCoords):
    if len(xCoords) < 3:
        return -1, -1
    eigenvalues, eigenvectors = Pca((xCoords, yCoords))
    ZeroCorrect(eigenvalues)
    return eigenvalues[0], eigenvalues[0] / eigenvalues[1]

def PcaVariance3D(xCoords, yCoords, zCoords):
    if len(xCoords) < 3:
        return -1, -1
    eigenvalues, eigenvectors = Pca((xCoords, yCoords, zCoords))
    ZeroCorrect(eigenvalues)
    axialVariance = m.sqrt(eigenvalues[0] + eigenvalues[1])
    return axialVariance, axialVariance / m.sqrt(eigenvalues[2])

def PcaReduce2D(xCoords, yCoords, xIntercept = None, yIntercept = None):
    if len(xCoords) < 2:
        return xCoords, yCoords
    if xIntercept == None or yIntercept == None:
        xIntercept = np.mean(xCoords)
        yIntercept = np.mean(yCoords)
    eigenvalues, eigenvectors = Pca((xCoords, yCoords), (xIntercept, yIntercept))
    return hb.RotatePointsClockwise(xCoords, yCoords, *eigenvectors[0])

def PcaReduce(coordSets, intercept = None):
    if len(coordSets[0]) < 2:
        return coordSets
    if type(intercept) == type(None):
        intercept = np.mean(coordSets, axis=1)
    eigenvalues, eigenvectors = Pca(coordSets, intercept)
    intercept = np.reshape(intercept, (-1, 1))
    return np.flip(eigenvectors.transpose(), 0) @ (coordSets - intercept)

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
    order = eigenvalues.argsort()
    return eigenvalues[order], eigenvectors[:,order]

def GetFeatures(pfo, wireViews):
    PcaReduce((pfo.driftCoordW, pfo.wireCoordW))
    featureDict = {}
    var3d, ratio3d = PcaVariance3D(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D)
    featureDict.update({ "PcaVar3D": var3d, "PcaRatio3D": ratio3d})
    if wireViews[0]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaVarU" : var2d, "PcaRatioU": ratio2d})
    if wireViews[1]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "PcaVarV" : var2d, "PcaRatioV": ratio2d})
    if wireViews[2]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordW, pfo.wireCoordW)
        featureDict.update({ "PcaVarW" : var2d, "PcaRatioW": ratio2d})
    return featureDict