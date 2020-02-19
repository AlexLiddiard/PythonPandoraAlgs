import math as m
import numpy as np
import PCAnalysis as pca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import TestDeDx1 as tdd1
from UpRootFileReader import ProjectVector

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# Gets all 3D hits within a 4 cm radius of the vertex, and returns the PCA eigenvector which has the largest corresponding eigenvalue.
def GetInitialDirection(xCoords, yCoords, zCoords, vertex, sphereRadius = 4):
    if len(xCoords) <= 1:
        return
    filt = GetHitsInRadius((xCoords, yCoords, zCoords), vertex, sphereRadius)
    if filt.sum() <= 1:
        return
    return pca.Pca((xCoords[filt], yCoords[filt], zCoords[filt]), intercept=vertex)[1][:,-1]

def GetHitsInRadius(coordSets, centre, radius=4):
    centre = np.reshape(centre, (-1, 1))
    return np.linalg.norm(coordSets - centre, axis=0) <= radius

# Get hits and charge in rectangle in 2D view
def Get2dHitsInRectangle(xCoords, yCoords, rectangleTopLeft=(0, 0.5), rectangleWidth=1, rectangleLength=4):
    filt = (
        (xCoords >= rectangleTopLeft[0]) & 
        (xCoords <= rectangleTopLeft[0] + rectangleLength) &
        (yCoords >= rectangleTopLeft[1] - rectangleWidth) &
        (yCoords <= rectangleTopLeft[1])
    )
    return filt

def GetLongitudinalError(driftCoordErrors, wireCoordErrors, longDirection):
    return driftCoordErrors * wireCoordErrors / np.sqrt((np.square(wireCoordErrors * longDirection[0]) + np.square(driftCoordErrors * longDirection[1])))

def GetDeDx(xCoords3D, yCoords3D, zCoords3D, view2D, driftCoords2D, driftCoordErrors2D, wireCoords2D, wireCoordError2D, charges2D, sphereRadius, rectangleWidth, rectangleLength, vertex3D = None, initialLength = 4, outlierFraction=0.85):
    if vertex3D is None:
        vertex3D = Calculate3dVertex(xCoords3D, yCoords3D, zCoords3D, initialLength, outlierFraction)
        if vertex3D is None:
            return m.nan
    vertex2D = ProjectVector(vertex3D, view2D)

    showerLongDirection3D = GetInitialDirection(xCoords3D, yCoords3D, zCoords3D, vertex3D, sphereRadius)
    if showerLongDirection3D is None:
        return m.nan
    showerLongDirection2D = ProjectVector(showerLongDirection3D, view2D)
    showerLongDirection2Dmag = np.linalg.norm(showerLongDirection2D)
    if showerLongDirection2Dmag == 0:
        return m.nan
    showerLongDirection2Dnormed = np.divide(showerLongDirection2D, showerLongDirection2Dmag)
    basisVectors = np.column_stack((showerLongDirection2Dnormed, [showerLongDirection2Dnormed[1], -showerLongDirection2Dnormed[0]]))
    reducedCoords = pca.ChangeCoordBasis((driftCoords2D, wireCoords2D), basisVectors, normed=True, preTranslation=-vertex2D)
    filt = Get2dHitsInRectangle(reducedCoords[0], reducedCoords[1], (0, rectangleWidth/2), rectangleWidth, rectangleLength * showerLongDirection2Dmag)
    #plt.scatter(driftCoords2D, wireCoords2D)
    #plt.scatter(driftCoords2D[filt], wireCoords2D[filt])
    #plt.plot([vertex2D[0], vertex2D[0] + showerLongDirection2Dnormed[0]], [vertex2D[1], vertex2D[1] + showerLongDirection2Dnormed[1]])
    #plt.scatter(vertex2D[0], vertex2D[1])
    #plt.axes().set_aspect('equal')
    #plt.show()def axisEqual3D(ax):
    if filt.sum() == 0:
        return np.nan
    driftCoordErrorsInrectangle = driftCoordErrors2D[filt]
    chargesInRectangle = charges2D[filt]
    lCoordErrors = GetLongitudinalError(driftCoordErrorsInrectangle, wireCoordError2D, showerLongDirection2Dnormed)
    dedxs1 = chargesInRectangle / lCoordErrors
    dedx1 = np.median(dedxs1) * showerLongDirection2Dmag

    #dedxs2 = chargesInRectangle / wireCoordError
    #dedx2 = np.median(dedxs2) * abs(showerLongDirection2D[1])
    return dedx1

def Calculate3dVertex(xCoords3D, yCoords3D, zCoords3D, initialLength=4, outlierFraction=0.85):
    nCoords = len(xCoords3D)
    if nCoords < 3:
        return

    # Do PCA to find the outliers
    filter = pca.FindOutliers(coordSets=(xCoords3D, yCoords3D, zCoords3D), fraction=outlierFraction)
    if filter.sum() < 3:
        return

    # Debug
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xCoords3D, yCoords3D, zCoords3D)
    #ax.scatter(xCoords3D[np.invert(filter)], yCoords3D[np.invert(filter)], zCoords3D[np.invert(filter)])

    # Do PCA for the whole PFO minus the outliers
    centroid = np.mean((xCoords3D[filter], yCoords3D[filter], zCoords3D[filter]), axis=1)
    eigenvectors = pca.Pca(coordSets=(xCoords3D[filter], yCoords3D[filter], zCoords3D[filter]), intercept=centroid)[1]
    reducedCoordSets = pca.ChangeCoordBasis(coordSets=(xCoords3D, yCoords3D, zCoords3D), basisVectors=np.flip(eigenvectors, 1), normed=True, preTranslation=-centroid)
    
    # Debug
    #ax.plot([centroid[0], centroid[0] + eigenvectors[0,2] * 100], [centroid[1], centroid[1] + eigenvectors[1,2] * 100], [centroid[2], centroid[2] + eigenvectors[2,2] * 100])

    # Order hits by the longitudinal axis
    order = reducedCoordSets[0].argsort()
    reducedCoordSets = reducedCoordSets[:,order]

    # Use transverse variance to determine the shower direction (keep outliers here)
    indexHalf = int(nCoords/2)
    transDintance2 = np.linalg.norm(reducedCoordSets[1:], axis=0)
    avg1 = np.mean(transDintance2[:indexHalf])
    avg2 = np.mean(transDintance2[indexHalf:])

    # Ensure that the xyz coords are in the direction of increasing longitudinal coord (remove outliers here)
    if avg2 < avg1:
        order = np.flip(order)
    xCoords3D = xCoords3D[order[filter]]
    yCoords3D = yCoords3D[order[filter]]
    zCoords3D = zCoords3D[order[filter]]

    # Debug
    #indexHalf = int(len(xCoords3D)/2)
    #ax.scatter(xCoords3D[indexHalf:], yCoords3D[indexHalf:], zCoords3D[indexHalf:])
    #ax.scatter(xCoords3D[:indexHalf], yCoords3D[:indexHalf], zCoords3D[:indexHalf])

    # Get the initial segment of the shower
    initialIndex = np.argmax((reducedCoordSets[0] - reducedCoordSets[0,0]) > initialLength)
    if initialIndex < 3: # Fallback for when we don't have enough hits
        initialIndex = 3
    initialCoords = np.array((xCoords3D[:initialIndex], yCoords3D[:initialIndex], zCoords3D[:initialIndex]))
    
    # Do PCA to find the initial straight line segment.
    #initialCoords = tdd1.GetInitialStraightSegment(coordSets=(xCoords3D, yCoords3D, zCoords3D), maxTransVar=maxTransVar)
    # As a fallback, use all the points in the PFO
    #if initialCoords is None:
    #    print("Using fallback for initial straight line!")
    #    initialCoords = np.array((xCoords3D[:3], yCoords3D[:3], zCoords3D[:3]))
    
    # Do PCA on the initial line segment.
    eigenvectors = pca.Pca(initialCoords)[1]
    centroid = np.mean(initialCoords, axis=1)

    # Debug
    #ax.scatter(initialCoords[0], initialCoords[1], initialCoords[2])

    # Use the initial line segment and some algebra to estimate the vertex
    cp = centroid - initialCoords[:,0]
    a = np.dot(cp, eigenvectors[:,0])
    b = np.dot(cp, eigenvectors[:,1])
    vertex = initialCoords[:,0] + eigenvectors[:,0] * a + eigenvectors[:,1] * b

    #ax.scatter(vertex[0], vertex[1], vertex[2])
    #axisEqual3D(ax)
    #plt.show()
    return vertex
    

def GetFeatures(pfo, calculateViews, sphereRadius=4, rectangleWidth=1, rectangleLength=4):
    featureDict = {}
    dedx = m.nan
    if calculateViews["U"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "U", pfo.driftCoordU, pfo.driftCoordErrU, pfo.wireCoordU, 0.3, pfo.energyU, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxU": dedx })
    if calculateViews["V"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "V", pfo.driftCoordV, pfo.driftCoordErrV, pfo.wireCoordV, 0.3, pfo.energyV, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxV": dedx })
    if calculateViews["W"]:
        if pfo.ValidVertex():
            dedx = GetDeDx(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, "W", pfo.driftCoordW, pfo.driftCoordErrW, pfo.wireCoordW, 0.3, pfo.energyW, sphereRadius, rectangleWidth, rectangleLength)
        featureDict.update({ "dedxW": dedx })
    return featureDict