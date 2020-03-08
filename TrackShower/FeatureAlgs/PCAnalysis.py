from sklearn.decomposition import PCA
import numpy as np
import math as m
import matplotlib.pyplot as plt

tolerance = 1e-5
def ZeroCorrect(values):
    values[np.abs(values) < tolerance] = 0

def PcaVariance(coordSets, vertex=None):
    if len(coordSets[0]) < 3:
        return m.nan, m.nan
    eigenvalues = Pca(coordSets, vertex, False)
    axialVariance = eigenvalues[:-1].sum()
    return axialVariance, axialVariance / eigenvalues[-1] / (len(coordSets) - 1)

def PcaReduce(coordSets, intercept=None, lDirectionCheck=False):
    if len(coordSets[0]) == 0:
        return coordSets
    if intercept is None:
        intercept = np.mean(coordSets, axis=1)
    intercept = np.reshape(intercept, (-1, 1))
    if len(coordSets[0]) == 1:
        return coordSets - intercept
    eigenvectors = Pca(coordSets, intercept)[1]
    reducedCoordSets = ChangeCoordBasis(coordSets, np.flip(eigenvectors, 1), True, -intercept)
    if lDirectionCheck and not CorrectDirection(reducedCoordSets[0]): # useful when intercept=vertex
        reducedCoordSets[0] *= -1
    return reducedCoordSets

# Check to ensure majority of hits have positive coord
def CorrectDirection(coords):
    return (coords >= 0).sum() > len(coords) / 2

# basisVectors = [[basis vector x coords], [basis vector y coords], ...]
# coordSets = [[x coords], [y coords], ...]
# normed = are the basis vectors normed?
# translation = [x coord, y coord, ...]
def ChangeCoordBasis(coordSets, basisVectors, normed=False, preTranslation=None):
    if preTranslation is not None:
        coordSets += np.reshape(preTranslation, (-1, 1))
    if not normed:
        basisVectors /= np.linalg.norm(basisVectors, axis=0).reshape(-1, 1)  
    return np.transpose(basisVectors) @ coordSets

def Pca(coordSets, intercept = None, withVectors=True):
    if intercept is None:
        intercept = np.mean(coordSets, axis=1)
    dimensions = len(coordSets)
    ncoords = len(coordSets[0])
    covmatrix = np.zeros((dimensions, dimensions), dtype='double')
    for i in range(0, dimensions):
        for j in range(i, dimensions):
            covmatrix[i, j] = covmatrix[j, i] = np.sum((coordSets[i] - intercept[i]) * (coordSets[j] - intercept[j])) / (ncoords - 1)
    return GetEigenValues(covmatrix, withVectors)

def GetEigenValues(covmatrix, withEigenvectors=True):
    eigenvalues, eigenvectors = np.linalg.eigh(covmatrix)
    ZeroCorrect(eigenvalues)
    if withEigenvectors:
        order = eigenvalues.argsort()
        return eigenvalues[order].real, eigenvectors[:,order].real # numpy can sometimes return a complex matrix for no apparent reason!
    else:
        eigenvalues.sort()
        return eigenvalues.real

def FindOutliers(coordSets, intercept = None, fraction=0.85):
    reducedCoordSets = PcaReduce(coordSets, intercept)
    tCoordSets = reducedCoordSets[1:]
    tDistance = np.linalg.norm(tCoordSets, axis=0)
    sort = tDistance.argsort()
    index = int(len(coordSets[0]) * fraction)
    filter = np.repeat(True, len(coordSets[0]))
    filter[sort[index:]] = False
    return filter

def GetFeatures(pfo, calculateViews):
    PcaReduce((pfo.driftCoordW, pfo.wireCoordW))
    featureDict = {}
    if calculateViews["U"]:
        var2d, ratio2d = PcaVariance((pfo.driftCoordU, pfo.wireCoordU))
        featureDict.update({ "PcaVarU" : var2d, "PcaRatioU": ratio2d})
    if calculateViews["V"]:
        var2d, ratio2d = PcaVariance((pfo.driftCoordV, pfo.wireCoordV))
        featureDict.update({ "PcaVarV" : var2d, "PcaRatioV": ratio2d})
    if calculateViews["W"]:
        var2d, ratio2d = PcaVariance((pfo.driftCoordW, pfo.wireCoordW))
        featureDict.update({ "PcaVarW" : var2d, "PcaRatioW": ratio2d})
    if calculateViews["3D"]:
        var3d, ratio3d = PcaVariance((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.vertex3D)
        featureDict.update({ "PcaVar3D": var3d, "PcaRatio3D": ratio3d})
    return featureDict