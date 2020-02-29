# This module is for track/shower feature #2
import AlgorithmConfig as cfg
import math as m
import numpy as np
import LinearRegression as lr
import HitBinning as hb
from itertools import count
import PCAnalysis as pca

# Finds the nearest neighbour of a 2D point (out of a list of points)
# It's quite slow
def NearestPoint(pointX, pointY, pointListX, pointListY):
    nearestPointIndex = 0
    shortestDistance2 = m.inf
    for i in range(0, len(pointListX)):
        tmp = Distance2(pointX, pointY, pointListX[i], pointListY[i])
        if tmp < shortestDistance2:
            nearestPointIndex = i
            shortestDistance2 = tmp
    return nearestPointIndex, shortestDistance2

# 3D version.
def NearestPoint3D(pointX, pointY, pointZ, pointListX, pointListY, pointListZ):
    nearestPointIndex = 0
    shortestDistance2 = m.inf
    for i in range(0, len(pointListX)):
        tmp = Distance23D(pointX, pointY, pointZ, pointListX[i], pointListY[i], pointListZ[i])
        if tmp < shortestDistance2:
            nearestPointIndex = i
            shortestDistance2 = tmp
    return nearestPointIndex, shortestDistance2

# Searches for the nearest neighbour of a 2D point (out of a list of points)
# It only checks points that are within a specified rectangular box (relative to the point).
# If there are no points in this search box, a distance of infinity and a point index of -1 is returned.
def NearestPointInRectangleAdvanced(pointX, pointY, pointListX, pointListY,
                            rectWidth, rectHeight, rectOffsetX = 0, rectOffsetY = 0, rectMirrorX = False, rectMirrorY = False, rectRotateSin = 0, rectRotateCos = 1):
    xLow = rectOffsetX - rectWidth / 2
    xHigh = rectOffsetX + rectWidth / 2
    yLow = rectOffsetY - rectHeight / 2
    yHigh = rectOffsetY + rectHeight / 2
    if rectMirrorX:
        xLow = -xHigh
        xHigh = -xLow
    if rectMirrorY:
        yLow = -yHigh
        yHigh = -yLow
    pointXnew = pointX * rectRotateCos + pointY * rectRotateSin
    pointYnew = pointY * rectRotateCos - pointX * rectRotateSin
    xLow += pointXnew
    xHigh += pointXnew
    yLow += pointYnew
    yHigh += pointYnew

    nearestPointIndex = m.nan
    shortestDistance2 = m.inf
    for i, x, y in zip(count(), pointListX, pointListY):
        yNew = y * rectRotateCos - x * rectRotateSin
        if yNew < yLow:
            continue
        if yNew > yHigh:
            continue
        xNew = x * rectRotateCos + y * rectRotateSin
        if xNew < xLow:
            continue
        if xNew > xHigh:
            continue
        distance2 = Distance2(pointX, pointY, x, y)
        if distance2 < shortestDistance2:
            nearestPointIndex = i
            shortestDistance2 = distance2
    return nearestPointIndex, shortestDistance2

# Finds the nearest neighbour of a 2D point (out of a list of points)
# The list of points checked is a subsample of the input list.
# It consists of all points within a rectangle of width rectWidth and height
# rectHeight centred on the point.
# If the subsample is empty, a distance of infinity and a point index of -1 is
# returned.
def NearestPointInRectangleSimple(pointX, pointY, pointListX, pointListY, rectWidth, rectHeight):
    xLow = pointX - rectWidth / 2
    xHigh = pointX + rectWidth / 2
    yLow = pointY - rectHeight / 2
    yHigh = pointY + rectHeight / 2
    nearestPointIndex = m.nan
    shortestDistance2 = m.inf
    for i in range(0, len(pointListX)):
        xTest = pointListX[i]
        yTest = pointListY[i]
        if xTest < xLow:
            continue
        if xTest > xHigh:
            continue
        if yTest < yLow:
            continue
        if yTest > yHigh:
            continue

        distance2 = Distance2(pointX, pointY, pointListX[i], pointListY[i])
        if distance2 < shortestDistance2:
            nearestPointIndex = i
            shortestDistance2 = distance2
    return nearestPointIndex, shortestDistance2

# 3D version
def NearestPointInCuboidSimple(pointX, pointY, pointZ, pointListX, pointListY, pointListZ, cubWidth, cubHeight, cubThickness):
    xLow = pointX - cubWidth / 2
    xHigh = pointX + cubWidth / 2
    yLow = pointY - cubHeight / 2
    yHigh = pointY + cubHeight / 2
    zLow = pointZ - cubThickness / 2
    zHigh = pointZ + cubThickness / 2
    #nearestPointIndex = m.nan
    #shortestDistance2 = m.inf

    pointListX = np.array(pointListX)
    pointListY = np.array(pointListY)
    pointListZ = np.array(pointListZ)

    pointList = np.column_stack((pointListX, pointListY, pointListZ, np.arange(len(pointListX))))
    pointList = pointList[pointList[:,0] > xLow]
    if len(pointList) == 0:
        return m.nan, m.inf
    pointList = pointList[pointList[:,0] < xHigh]
    if len(pointList) == 0:
        return m.nan, m.inf
    pointList = pointList[pointList[:,1] > yLow]
    if len(pointList) == 0:
        return m.nan, m.inf
    pointList = pointList[pointList[:,1] < yHigh]
    if len(pointList) == 0:
        return m.nan, m.inf
    pointList = pointList[pointList[:,2] > zLow]
    if len(pointList) == 0:
        return m.nan, m.inf
    pointList = pointList[pointList[:,2] < zHigh]
    if len(pointList) == 0:
        return m.nan, m.inf
    distances = np.linalg.norm(pointList[:, [0, 1, 2]], axis=1)
    i = np.argmin(distances)
    

    #for i in range(0, len(pointListX)):
    #    xTest = pointListX[i]
    #    yTest = pointListY[i]
    #    zTest = pointListZ[i]
    #    if xTest < xLow:
    #        continue
    #    if xTest > xHigh:
    #        continue
    #    if yTest < yLow:
    #        continue
    #    if yTest > yHigh:
    #        continue
    #    if zTest < zLow:
    #        continue
    #    if zTest > zHigh:
    #        continue
    #    distance2 = Distance23D(pointX, pointY, pointZ, pointListX[i], pointListY[i], pointListZ[i])
    #    if distance2 < shortestDistance2:
    #        nearestPointIndex = i
    #        shortestDistance2 = distance2
    #return nearestPointIndex, shortestDistance2

    return int(pointList[i, 3]), distances[i]
    
# Returns the square of the separation distance of a pair of 2D points
def Distance2(pointAX, pointAY, pointBX, pointBY):
    deltaX = pointAX - pointBX
    deltaY = pointAY - pointBY
    return deltaX * deltaX + deltaY * deltaY

# 3D version
def Distance23D(pointAX, pointAY, pointAZ, pointBX, pointBY, pointBZ):
    deltaX = pointAX - pointBX
    deltaY = pointAY - pointBY
    deltaZ = pointAZ - pointBZ
    return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ

# Rotates points A and B by an angle (given by sin, cos) and then checks if the point B has a greater
# x component than A. If so then returns true, otherwise false.

def RotateCompareX(xA, yA, xB, yB, sin, cos):
    return (xA - xB) * sin + (yA - yB) * cos

# Creates a list of points that form a chain. Each point pair in the chain has a
# squared separation smaller than maxSeparation2.
# The chain starts with the first point in the input list
# The chain ends when no more nearby points can be found.
# The points added to the chain are removed from the input list of points.
def CreatePointChain1(pointListX, pointListY, maxSeparation2):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    chainLength = 0
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance2 = NearestPoint(currentX, currentY, pointListX, pointListY)
        if distance2 < maxSeparation2:
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += m.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainLength

# 3D version
def Create3DPointChain1(pointListX, pointListY, pointListZ, maxSeparation2):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    currentZ = pointListZ.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    chainZ = [currentZ]
    chainLength = 0
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance2 = NearestPoint3D(currentX, currentY, currentZ, pointListX, pointListY, pointListZ)
        if distance2 < maxSeparation2:
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            currentZ = pointListZ.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainZ.append(currentZ)
            chainLength += m.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainZ, chainLength

# Using a square instead of a circle to limit the max separation
# A bit more efficient than original version
def CreatePointChain2(pointListX, pointListY, rectWidth, rectHeight):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    chainLength = 0
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance2 = NearestPointInRectangleSimple(currentX, currentY, pointListX, pointListY, rectWidth, rectHeight)
        if not m.isnan(nearestPointIndex):
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += m.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainLength, m.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1]))

# 3D version
def Create3DPointChain2(pointListX, pointListY, pointListZ, cubWidth, cubHeight, cubThickness):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    currentZ = pointListZ.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    chainZ = [currentZ]
    chainLength = 0
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance2 = NearestPointInCuboidSimple(currentX, currentY, currentZ, pointListX, pointListY, pointListZ, cubWidth, cubHeight, cubThickness)
        if not m.isnan(nearestPointIndex):
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            currentZ = pointListZ.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainZ.append(currentZ)
            chainLength += m.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainZ, chainLength, m.sqrt(Distance23D(chainX[0], chainY[0], chainZ[0], chainX[-1], chainY[-1], chainZ[-1]))
# Same as above, but with an intelligent placement of the square based on local correlation
# Less efficient than previous version
def CreatePointChain3(pointListX, pointListY, rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    chainLength = 0
    sumRSquared = 0
    nRSquared = 0
    nearbyPoints = True
    while (nearbyPoints):
        correlationPointsX = np.array(chainX[-localCorrelationPoints:])
        correlationPointsY = np.array(chainY[-localCorrelationPoints:])
        a, b, r2 = lr.OLS(correlationPointsX, correlationPointsY)
        if not m.isnan(r2):
            if len(chainX) >= localCorrelationPoints:
                sumRSquared += r2
                nRSquared += 1
            rectRotateSin, rectRotateCos = hb.TanToSinCos(b)
            # Need to make sure the rectangle search is in the local direction of the chain
            # Rotate first and last correlation points, based on the correlation gradient. Then check if the first point has a greater x component than the last.
            rectMirrorX = 0 < ((correlationPointsX[0] - correlationPointsX[-1]) * rectRotateSin + (correlationPointsY[0] - correlationPointsY[-1]) * rectRotateCos)
            nextPointIndex, distance2 = NearestPointInRectangleAdvanced(currentX, currentY, pointListX, pointListY, rectWidth, rectHeight, rectOffsetX, rectOffestY, rectMirrorX, False, b)
        else:
            nextPointIndex, distance2 = NearestPointInRectangleSimple(currentX, currentY, pointListX, pointListY, squareSideLength, squareSideLength)
        if not m.isnan(nextPointIndex):
            currentX = pointListX.pop(nextPointIndex)
            currentY = pointListY.pop(nextPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += m.sqrt(distance2)
        else:
            nearbyPoints = False

    avgRSquared = sumRSquared / nRSquared if nRSquared > 0 else m.nan
    return chainX, chainY, chainLength, m.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1])), avgRSquared

def SlidingPearsonRSquared(chainX, chainY, pointsPerSlide):
    chainX, chainY = np.array(chainX), np.array(chainY)
    if len(chainX) <= pointsPerSlide:
        return lr.OLS(chainX, chainY)[2], m.nan
    else:
        chainRSquareds = []
        n = len(chainX) - pointsPerSlide + 1
        for i in range(n):
            subChainX = chainX[i: i + pointsPerSlide]
            subChainY = chainY[i: i + pointsPerSlide]
            chainRSquareds.append(lr.OLS(subChainX, subChainY)[2])
        return np.mean(chainRSquareds), np.std(chainRSquareds)

def SlidingPCA3D(chainX, chainY, chainZ, pointsPerSlide):
    chainX, chainY, chainZ = np.array(chainX), np.array(chainY), np.array(chainZ)
    if len(chainX) <= pointsPerSlide:
        return pca.PcaVariance((chainX, chainY, chainZ))[1], m.nan
    else:
        chainRSquareds = []
        n = len(chainX) - pointsPerSlide + 1
        for i in range(n):
            subChainX = chainX[i: i + pointsPerSlide]
            subChainY = chainY[i: i + pointsPerSlide]
            subChainZ = chainZ[i: i + pointsPerSlide]
            chainRSquareds.append(pca.PcaVariance((subChainX, subChainY, subChainZ))[1])
        return np.mean(chainRSquareds), np.std(chainRSquareds)

def GetChainInfoAdvanced(driftCoord, wireCoord, rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints):
    chainCount = 0
    lengthRatios = []
    avgR2s = []

    while driftCoord:    # While pfo hit list is not empty
        chainX, chainY, chainLength, chainDisplacement, avgR2 = CreatePointChain3(driftCoord, wireCoord, rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints)
        if not m.isnan(avgR2):
            lengthRatios.append(chainDisplacement / chainLength)
            avgR2s.append(avgR2)
        chainCount += 1
    if lengthRatios:
        avgLengthRatio = np.mean(lengthRatios)
        avgAvgR2 = np.mean(avgR2s)
        stdLengthRatio = np.std(lengthRatios)
    else:
        avgLengthRatio = m.nan
        avgAvgR2 = m.nan
    return chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio

def GetChainInfoSimple(driftCoord, wireCoord, squareSideLength, localCorrelationPoints):
    chainCount = 0
    avgAvgR2 = m.nan
    avgStdR2 = m.nan
    avgLengthRatio = m.nan
    stdLengthRatio = m.nan
    lengthRatios = []
    avgR2s = []
    stdR2s = []
    while driftCoord:    # While pfo hit list is not empty
        chainX, chainY, chainLength, chainDisplacement = CreatePointChain2(driftCoord, wireCoord, squareSideLength, squareSideLength)
        avgR2, stdR2 = SlidingPearsonRSquared(chainX, chainY, localCorrelationPoints)
        if not m.isnan(avgR2):
            avgR2s.append(avgR2)
        if not m.isnan(stdR2):
            stdR2s.append(stdR2)
        if chainLength != 0:
            lengthRatios.append(chainDisplacement / chainLength)
        chainCount += 1
    nLengthRatios = len(lengthRatios)
    nAvgR2s = len(avgR2s)
    nStdR2s = len(stdR2s)
    if nLengthRatios > 0:
        avgLengthRatio = np.mean(lengthRatios)
    if nLengthRatios > 1:
        stdLengthRatio = np.std(lengthRatios)
    if nAvgR2s > 0:
        avgAvgR2 = np.mean(avgR2s)
    if nStdR2s > 0:
        avgStdR2 = np.mean(stdR2s)
    return chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2

def Get3DChainInfoSimple(xCoord, yCoord, zCoord, cubeSideLength, localCorrelationPoints):
    chainCount = 0
    avgAvgR2 = m.nan
    avgStdR2 = m.nan
    avgLengthRatio = m.nan
    stdLengthRatio = m.nan
    lengthRatios = []
    avgR2s = []
    stdR2s = []
    while xCoord:    # While pfo hit list is not empty
        chainX, chainY, chainZ, chainLength, chainDisplacement = Create3DPointChain2(xCoord, yCoord, zCoord, cubeSideLength, cubeSideLength, cubeSideLength)
        avgR2, stdR2 = 0, 0 #SlidingPCA3D(chainX, chainY, chainZ, localCorrelationPoints)
        if not m.isnan(avgR2):
            avgR2s.append(avgR2)
        if not m.isnan(stdR2):
            stdR2s.append(stdR2)
        if chainLength != 0:
            lengthRatios.append(chainDisplacement / chainLength)
        chainCount += 1
    nLengthRatios = len(lengthRatios)
    nAvgR2s = len(avgR2s)
    nStdR2s = len(stdR2s)
    if nLengthRatios > 0:
        avgLengthRatio = np.mean(lengthRatios)
    if nLengthRatios > 1:
        stdLengthRatio = np.std(lengthRatios)
    if nAvgR2s > 0:
        avgAvgR2 = np.mean(avgR2s)
    if nStdR2s > 0:
        avgStdR2 = np.mean(stdR2s)
    return chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["U"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordU.tolist(), pfo.wireCoordU.tolist(), cfg.chainCreation["squareSideLength"], cfg.chainCreation["localCorrelationPoints"])
        featureDict.update({ "ChainCountU": chainCount, "ChainRatioAvgU": avgLengthRatio, "ChainRSquaredAvgU": avgAvgR2, "ChainRatioStdU": stdLengthRatio, "ChainRSquaredStdU": avgStdR2 })
    if calculateViews["V"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordV.tolist(), pfo.wireCoordV.tolist(), cfg.chainCreation["squareSideLength"], cfg.chainCreation["localCorrelationPoints"])
        featureDict.update({ "ChainCountV": chainCount, "ChainRatioAvgV": avgLengthRatio, "ChainRSquaredAvgV": avgAvgR2, "ChainRatioStdV": stdLengthRatio, "ChainRSquaredStdV": avgStdR2 })
    if calculateViews["W"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordW.tolist(), pfo.wireCoordW.tolist(),  cfg.chainCreation["squareSideLength"], cfg.chainCreation["localCorrelationPoints"])
        featureDict.update({ "ChainCountW": chainCount, "ChainRatioAvgW": avgLengthRatio, "ChainRSquaredAvgW": avgAvgR2, "ChainRatioStdW": stdLengthRatio, "ChainRSquaredStdW": avgStdR2 })
    #if calculateViews["3D"]:
    #    chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = Get3DChainInfoSimple(pfo.xCoord3D.tolist(), pfo.yCoord3D.tolist(), pfo.zCoord3D.tolist(), cubeSideLength, localCorrelationPoints)
    #    featureDict.update({ "ChainCount3D": chainCount, "ChainRatioAvg3D": avgLengthRatio, "ChainRSquaredAvg3D": avgAvgR2, "ChainRatioStd3D": stdLengthRatio, "ChainRSquaredStd3D": avgStdR2 })

    # Advanced chain creation
    #chainCount, avgLengthRatio, avgChainRSquareds = GetChainInfoAdvanced(pfo.driftCoordW.tolist(), pfo.wireCoordW.tolist(), rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints)
    return featureDict
