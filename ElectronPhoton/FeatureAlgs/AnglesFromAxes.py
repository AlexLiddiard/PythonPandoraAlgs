import math as m
import numpy as np
import PCAnalysis as pca

def CalcAnglesFromAxes(xCoords3D, yCoords3D, zCoords3D, outlierFraction=0.85):
    nCoords = len(xCoords3D)
    if nCoords < 3:
        return

    # Do PCA to find the outliers
    filter = pca.FindOutliers(coordSets=(xCoords3D, yCoords3D, zCoords3D), fraction=outlierFraction)
    if filter.sum() < 3:
        return

    # Do PCA for the whole PFO minus the outliers
    centroid = np.mean((xCoords3D[filter], yCoords3D[filter], zCoords3D[filter]), axis=1)
    eigenvectors = pca.Pca(coordSets=(xCoords3D[filter], yCoords3D[filter], zCoords3D[filter]))[1]
    reducedCoordSets = pca.ChangeCoordBasis(coordSets=(xCoords3D, yCoords3D, zCoords3D), basisVectors=np.flip(eigenvectors, 1), normed=True, preTranslation=-centroid)

    # Order hits by the longitudinal axis
    order = reducedCoordSets[0].argsort()
    reducedCoordSets = reducedCoordSets[:,order]

    # Use transverse variance to determine the shower direction (keep outliers here)
    indexHalf = int(nCoords/2)
    transDintance2 = np.linalg.norm(reducedCoordSets[1:], axis=0)
    avg1 = np.mean(transDintance2[:indexHalf])
    avg2 = np.mean(transDintance2[indexHalf:])

    showerAxis = eigenvectors[:, 2]
    # Use this to ensure that we are dotting with a principal axis unit vector pointing in the direction of the shower
    if avg2 < avg1:
        showerAxis *= -1
    return np.arccos(showerAxis)

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["3D"]:
        angles = CalcAnglesFromAxes(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D)
        if angles is None:
            angles = np.repeat(np.nan, 3)
        featureDict.update({"AngleFromX_3D": angles[0], "AngleFromY_3D": angles[1], "AngleFromZ_3D": angles[2]})
    return featureDict