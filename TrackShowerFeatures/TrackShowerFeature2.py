# This module is for track/shower feature #2
import math
import numpy as np
import TrackShowerFeatures.TrackShowerFeature0 as TSFO

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


# Finds the nearest neighbour of a 2D point (out of a list of points)
# The list of points checked is a subsample of the input list.
# It consists of all points within a rectangle of width rectWidth and height
# rectHeight centred on the point.
# If the subsample is empty, a distance of infinity and a point index of -1 is
# returned.
def NearestPointInRectangle(pointX, pointY, pointListX, pointListY, rectWidth, rectHeight):
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


# Resurns the square of the separation distance of a pair of 2D points
def Distance2(pointAX, pointAY, pointBX, pointBY):
    deltaX = pointAX - pointBX
    deltaY = pointAY - pointBY
    return deltaX * deltaX + deltaY * deltaY


# Creates a list of points that form a chain. Each point pair in the chan has a
# squared separation smaller than maxSeparation2.
# The chain starts with the first point in the input list
# The chain ends when no more nearby points can be found.
# The points added to the chain are removed from the input list of points.
def CreatePointChain(pointListX, pointListY, maxSeparation2):
    currentX = pointListX.pop(0)
    currentY = pointListY.pop(0)
    chainX = [currentX]
    chainY = [currentY]
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance = NearestPoint(currentX, currentY, pointListX, pointListY)
        if distance < maxSeparation2:
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
        else:
            nearbyPoints = False
    return chainX, chainY


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
        nearestPointIndex, distance2 = NearestPointInRectangle(currentX, currentY, pointListX, pointListY, rectWidth, rectHeight)
        if nearestPointIndex >= 0:
            currentX = pointListX.pop(nearestPointIndex)
            currentY = pointListY.pop(nearestPointIndex)
            chainX.append(currentX)
            chainY.append(currentY)
            chainLength += math.sqrt(distance2)
        else:
            nearbyPoints = False
    return chainX, chainY, chainLength

def SlidingPearsonRSquared(chainX, chainY, pointsPerSlide):
    chainX, chainY = np.array(chainX), np.array(chainY)
    if len(chainX) <= pointsPerSlide:
        return TSFO.RSquared(chainX, chainY)
    else:
        sumChainRSquared = 0
        n = len(chainX) - pointsPerSlide + 1
        for i in range(n):
            subChainX = chainX[i:i+pointsPerSlide]
            subChainY = chainY[i:i+pointsPerSlide]
            sumChainRSquared += TSFO.RSquared(subChainX, subChainY)
        return sumChainRSquared/n


def GetChainInfo(driftCoord, wireCoord, rectWidth, rectHeight, pointsPerSlide):
    chainCount = 0
    sumLengthRatios = 0
    sumChainRSquareds = 0
    while driftCoord:    # While pfo hit list is not empty
        chainX, chainY, chainLength = CreatePointChain2(driftCoord, wireCoord, rectWidth, rectHeight)
        if len(chainX) > 1:
            sumLengthRatios += math.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1])) / chainLength
            sumChainRSquareds += SlidingPearsonRSquared(chainX, chainY, pointsPerSlide)
            chainCount += 1
        else:
            lengthRatio = 0
    avgLengthRatio = sumLengthRatios / chainCount
    avgChainRSquareds = sumChainRSquareds/chainCount
    return chainCount, avgLengthRatio, avgChainRSquareds


def GetFeature(pfo, rectWidth=5, rectHeight=5, pointsPerSlide=10):
    chainCount, avgLengthRatio, avgChainRSquareds = GetChainInfo(pfo.driftCoordW.tolist(), pfo.wireCoordW.tolist(), rectWidth, rectHeight, pointsPerSlide)
    return { "F2a": chainCount, "F2b": avgLengthRatio, "F2c": avgChainRSquareds }
