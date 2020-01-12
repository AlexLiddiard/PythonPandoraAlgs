# This module is for track/shower feature #2
import math
import numpy as np
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.HitBinning as hb
from itertools import count
from statistics import stdev, mean

# Finds the nearest neighbour of a 2D point (out of a list of points)
# It's quite slow
def NearestPoint(pointX, pointY, pointListX, pointListY):
    nearestPointIndex = 0
    shortestDistance2 = float("inf")
    for i in range(0, len(pointListX)):
        tmp = Distance2(pointX, pointY, pointListX[i], pointListY[i])
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

    nearestPointIndex = -1
    shortestDistance2 = float("inf")
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
    nearestPointIndex = -1
    shortestDistance2 = float("inf")
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

# Returns the square of the separation distance of a pair of 2D points
def Distance2(pointAX, pointAY, pointBX, pointBY):
    deltaX = pointAX - pointBX
    deltaY = pointAY - pointBY
    return deltaX * deltaX + deltaY * deltaY

# Rotates points A and B by an angle (given by sin, cos) and then checks if the point B has a greater
# x component than A. If so then returns true, otherwise false.

def RotateCompareX(xA, yA, xB, yB, sin, cos):
    return (xA - xB) * sin + (yA - yB) * cos

# Creates a list of points that form a chain. Each point pair in the chan has a
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
            chainLength += math.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainLength

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
        if nearestPointIndex >= 0:
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += math.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainLength, math.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1]))

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
        if r2 != -1:
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
        if nextPointIndex >= 0:
            currentX = pointListX.pop(nextPointIndex)
            currentY = pointListY.pop(nextPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += math.sqrt(distance2)
        else:
            nearbyPoints = False

    avgRSquared = sumRSquared / nRSquared if nRSquared > 0 else -1
    return chainX, chainY, chainLength, math.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1])), avgRSquared



def SlidingPearsonRSquared(chainX, chainY, pointsPerSlide):
    chainX, chainY = np.array(chainX), np.array(chainY)
    if len(chainX) <= pointsPerSlide:
        return lr.OLS(chainX, chainY)[2], -1
    else:
        chainRSquareds = []
        n = len(chainX) - pointsPerSlide + 1
        for i in range(n):
            subChainX = chainX[i:i+pointsPerSlide]
            subChainY = chainY[i:i+pointsPerSlide]
            chainRSquareds.append(lr.OLS(subChainX, subChainY)[2])
        return mean(chainRSquareds), stdev(chainRSquareds)


def GetChainInfoAdvanced(driftCoord, wireCoord, rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints):
    chainCount = 0
    lengthRatios = []
    avgR2s = []

    while driftCoord:    # While pfo hit list is not empty
        chainX, chainY, chainLength, chainDisplacement, avgR2 = CreatePointChain3(driftCoord, wireCoord, rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints)
        if avgR2 != -1:
            lengthRatios.append(chainDisplacement / chainLength)
            avgR2s.append(avgR2)
        chainCount += 1
    if lengthRatios:
        avgLengthRatio = mean(lengthRatios)
        avgAvgR2 = mean(avgR2s)
        stdLengthRatio = stdev(lengthRatios)
    else:
        avgLengthRatio = -1
        avgAvgR2 = -1
    return chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio

def GetChainInfoSimple(driftCoord, wireCoord, squareSideLength, localCorrelationPoints):
    chainCount = 0
    avgAvgR2 = -1
    avgStdR2 = -1
    avgLengthRatio = -1
    stdLengthRatio = -1
    lengthRatios = []
    avgR2s = []
    stdR2s = []
    while driftCoord:    # While pfo hit list is not empty
        chainX, chainY, chainLength, chainDisplacement = CreatePointChain2(driftCoord, wireCoord, squareSideLength, squareSideLength)
        avgR2, stdR2 = SlidingPearsonRSquared(chainX, chainY, localCorrelationPoints)
        if avgR2 != -1:
            avgR2s.append(avgR2)
        if stdR2 != -1:
            stdR2s.append(stdR2)
        if chainLength != 0:
            lengthRatios.append(chainDisplacement / chainLength)
        chainCount += 1
    nLengthRatios = len(lengthRatios)
    nAvgR2s = len(avgR2s)
    nStdR2s = len(stdR2s)
    if nLengthRatios > 0:
        avgLengthRatio = mean(lengthRatios)
    if nLengthRatios > 1:
        stdLengthRatio = stdev(lengthRatios)
    if nAvgR2s > 0:
        avgAvgR2 = mean(avgR2s)
    if nStdR2s > 0:
        avgStdR2 = mean(stdR2s)
    return chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2


def GetFeatures(pfo, calculateViews, rectWidth=10, rectHeight=2.5, rectOffsetX=2.5, rectOffestY=0, squareSideLength=5, localCorrelationPoints=5):
    # Advanced chain creation
    #chainCount, avgLengthRatio, avgChainRSquareds = GetChainInfoAdvanced(pfo.driftCoordW.tolist(), pfo.wireCoordW.tolist(), rectWidth, rectHeight, rectOffsetX, rectOffestY, squareSideLength, localCorrelationPoints)
    featureDict = {}
    if calculateViews["U"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordU.tolist(), pfo.wireCoordU.tolist(), squareSideLength, localCorrelationPoints)
        featureDict.update({ "ChainCountU": chainCount, "ChainRatioAvgU": avgLengthRatio, "ChainRSquaredAvgU": avgAvgR2, "ChainRatioStdU": stdLengthRatio, "ChainRSquaredStdU": avgStdR2 })
    if calculateViews["V"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordV.tolist(), pfo.wireCoordV.tolist(), squareSideLength, localCorrelationPoints)
        featureDict.update({ "ChainCountV": chainCount, "ChainRatioAvgV": avgLengthRatio, "ChainRSquaredAvgV": avgAvgR2, "ChainRatioStdV": stdLengthRatio, "ChainRSquaredStdV": avgStdR2 })
    if calculateViews["W"]:
        chainCount, avgLengthRatio, avgAvgR2, stdLengthRatio, avgStdR2 = GetChainInfoSimple(pfo.driftCoordW.tolist(), pfo.wireCoordW.tolist(), squareSideLength, localCorrelationPoints)
        featureDict.update({ "ChainCountW": chainCount, "ChainRatioAvgW": avgLengthRatio, "ChainRSquaredAvgW": avgAvgR2, "ChainRatioStdW": stdLengthRatio, "ChainRSquaredStdW": avgStdR2 })
    return featureDict
