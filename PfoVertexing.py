import BaseConfig as bc
import numpy as np
import PCAnalysis as pca

def InteractionVertex2D(pfoVertex, pfoXCoords, pfoZCoords, parentPfoVertex, parentPfoXCoords, parentPfoZCoords, pfoHits, parentPfoHits):
    if pfoHits < 2 or parentPfoHits < 2:
        return
    pfoEigenVector = FindPfoEigenVector2D(pfoXCoords, pfoZCoords, pfoVertex)
    parentPfoEigenVector = FindPfoEigenVector2D(parentPfoXCoords, parentPfoZCoords, parentPfoVertex)
    matrix = np.transpose([pfoEigenVector, -parentPfoEigenVector])
    vertexDifference = parentPfoVertex - pfoVertex
    try:
        solutions = np.linalg.solve(matrix, vertexDifference)
        intersection = pfoVertex + solutions[0]*pfoEigenVector
        return intersection
    except:
        return

def FindPfoEigenVector2D(pfoXCoords, pfoZCoords, pfoVertex):
    eigenvectors = pca.Pca((pfoXCoords, pfoZCoords), pfoVertex)[1]
    return eigenvectors[:,-1]

def FindPfoEigenVector3D(pfoXCoords, pfoYCoords, pfoZCoords, pfoVertex):
    eigenvectors = pca.Pca((pfoXCoords, pfoYCoords, pfoZCoords), pfoVertex)[1]
    return eigenvectors[:,-1]

def InteractionVertex3D(pfoVertex, pfoXCoords, pfoYCoords, pfoZCoords, parentPfoVertex, parentPfoXCoords, parentPfoYCoords, parentPfoZCoords, pfoHits, parentPfoHits):
    if pfoHits < 2 or parentPfoHits < 2:
        return
    pfoEigenVector = FindPfoEigenVector3D(pfoXCoords, pfoYCoords, pfoZCoords, pfoVertex)
    parentPfoEigenVector = FindPfoEigenVector3D(parentPfoXCoords, parentPfoYCoords, parentPfoZCoords, parentPfoVertex)
    perpendicularVector = np.cross(pfoEigenVector, parentPfoEigenVector)
    matrix = np.transpose([pfoEigenVector, perpendicularVector, -parentPfoEigenVector])
    vertexDifference = parentPfoVertex - pfoVertex
    try:
        solutions = np.linalg.solve(matrix, vertexDifference)
        intersection = solutions[0]*pfoEigenVector + pfoVertex + solutions[1]*perpendicularVector/2
        return intersection
    except:
        return

def CalculateShower3DVertex(xCoords3D, yCoords3D, zCoords3D, initialLength=4, outlierFraction=0.85):
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
    