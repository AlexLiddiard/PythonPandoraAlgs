from sklearn.decomposition import PCA
import numpy as np
import TrackShowerFeatures.HitBinning as hb
import math as m

def PcaVariance2D(xCoords, yCoords):
    if len(xCoords) < 2:
        return -1, -1
    eigenvalues, eigenvectors = Pca((xCoords, yCoords))
    return eigenvalues[0], eigenvalues[0] / eigenvalues[1]

def PcaVariance3D(xCoords, yCoords, zCoords):
    if len(xCoords) < 2:
        return -1, -1
    eigenvalues, eigenvectors = Pca((xCoords, yCoords, zCoords))
    axialVariance = m.sqrt(eigenvalues[0] * eigenvalues[0] + eigenvalues[1] * eigenvalues[1])
    return axialVariance, axialVariance / eigenvalues[2]

def PcaReduce2D(xCoords, yCoords, xIntercept = None, yIntercept = None):
    if len(xCoords) < 2:
        return xCoords, yCoords
    if xIntercept == None or yIntercept == None:
        xIntercept = np.mean(xCoords)
        yIntercept = np.mean(yCoords)
    eigenvalues, eigenvectors = Pca((xCoords, yCoords), (xIntercept, yIntercept))
    return hb.RotatePointsClockwise(xCoords, yCoords, *eigenvectors[0])

def Pca(coordSets, intercept = None):
    if intercept == None:
        intercept = np.mean(coordSets, axis=1)
    dimensions = len(coordSets)
    ncoords = len(coordSets[0])
    pcamatrix = np.zeros((dimensions, dimensions), dtype='double')
    for i in range(0, dimensions):
        for j in range(i, dimensions):
            pcamatrix[i, j] = pcamatrix[j, i] = np.sum((coordSets[i] - intercept[i]) * (coordSets[j] - intercept[j])) / (ncoords - 1)
    eigenvalues, eigenvectors = np.linalg.eig(pcamatrix)
    order = eigenvalues.argsort()
    return eigenvalues[order], eigenvectors[order]


def GetFeatures(pfo, wireViews):
    featureDict = {}
    var3d, ratio3d = PcaVariance3D(pfo.xCoordThreeD, pfo.yCoordThreeD, pfo.zCoordThreeD)
    featureDict.update({ "PcaVar3d": var3d, "PcaRatio3d": ratio3d})
    if wireViews[0]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaVar2dU" : var2d, "PcaRatio2dU": ratio2d})
    if wireViews[1]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "PcaVar2dV" : var2d, "PcaRatio2dV": ratio2d})
    if wireViews[2]:
        var2d, ratio2d = PcaVariance2D(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaVar2dW" : var2d, "PcaRatio2dW": ratio2d})
    return featureDict