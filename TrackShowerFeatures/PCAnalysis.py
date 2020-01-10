from sklearn.decomposition import PCA
import numpy as np
import TrackShowerFeatures.HitBinning as hb

def PcaVariance2D(xCoords, yCoords):
    if len(xCoords) < 2:
        return -1, -1
    eigenvalues, eigenvectors = Pca((xCoords, yCoords), (np.mean(xCoords), np.mean(yCoords)))
    return eigenvalues[0], eigenvalues[0] / eigenvalues[1]

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
    if wireViews[0]:
        minVar, minRatio = PcaVariance2D(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaMinVarU" : minVar, "PcaMinRatioU": minRatio})
    if wireViews[1]:
        minVar, minRatio = PcaVariance2D(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "PcaMinVarV" : minVar, "PcaMinRatioV": minRatio})
    if wireViews[2]:
        minVar, minRatio = PcaVariance2D(pfo.driftCoordW, pfo.wireCoordW)
        featureDict.update({ "PcaMinVarW" : minVar, "PcaMinRatioW": minRatio})

    return featureDict