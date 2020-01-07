from sklearn.decomposition import PCA
import numpy as np

def PcaVariance(xCoords, yCoords):
    if len(xCoords) < 2:
        return -1, -1
    X = np.column_stack((xCoords, yCoords))
    pca = PCA(n_components=2)
    pca.fit(X)
    return min(pca.explained_variance_), min(pca.explained_variance_ratio_)

def PcaReduce(xCoords, yCoords):
    X = np.column_stack((xCoords, yCoords))
    if len(xCoords) < 2:
        return X
    else:
        pca = PCA(n_components=2)
        return pca.fit_transform(X)

def GetFeatures(pfo, wireViews):
    featureDict = {}
    if wireViews[0]:
        minVar, minRatio = PcaVariance(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "PcaMinVarU" : minVar, "PcaMinRatioU": minRatio})
    if wireViews[1]:
        minVar, minRatio = PcaVariance(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "PcaMinVarV" : minVar, "PcaMinRatioV": minRatio})
    if wireViews[2]:
        minVar, minRatio = PcaVariance(pfo.driftCoordW, pfo.wireCoordW)
        featureDict.update({ "PcaMinVarW" : minVar, "PcaMinRatioW": minRatio})
    return featureDict