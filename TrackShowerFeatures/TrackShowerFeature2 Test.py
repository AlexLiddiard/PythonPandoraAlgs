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
def NearestPointInRectangle(pointIndex, pointListX, pointListY, rectWidth, rectHeight, availablePointIndices=None):
    if availablePointIndices is None:
        availablePointIndices = list(range(0, len(pointListX)))

    pointX = pointListX[pointIndex]
    pointY = pointListY[pointIndex]
    xLow = pointX - rectWidth / 2
    xHigh = pointX + rectWidth / 2
    yLow = pointY - rectHeight / 2
    yHigh = pointY + rectHeight / 2
    nearestPointIndex = -1
    shortestDistance2 = float("inf")
    for i in range(0, len(availablePointIndices)):
        j = availablePointIndices[i]
        xTest = pointListX[j]
        yTest = pointListY[j]
        if xTest < xLow:
            continue
        if xTest > xHigh:
            continue
        if yTest < yLow:
            continue
        if yTest > yHigh:
            continue

        distance2 = Distance2(pointX, pointY, xTest, yTest)
        if distance2 < shortestDistance2:
            nearestPointIndex = i
            shortestDistance2 = distance2
    return nearestPointIndex, shortestDistance2
# nearestPointIndex is an index for a position in availablePointIndices.
# It is NOT the index that is given at that position in availablePointIndices!


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
def CreatePointChain2(pointListX, pointListY, rectWidth, rectHeight, availablePointIndices=None):
    if availablePointIndices is None:
        availablePointIndices = list(range(0, len(pointListX)))
    currentPointIndex = availablePointIndices.pop(0)
    chain = [currentPointIndex]
    chainLength = 0
    nearbyPoints = True
    while (nearbyPoints):
        nearestPointIndex, distance2 = NearestPointInRectangle(currentPointIndex, pointListX, pointListY, rectWidth, rectHeight, availablePointIndices)
        if nearestPointIndex >= 0:
            currentPointIndex = availablePointIndices.pop(nearestPointIndex)
            chain.append(currentPointIndex)
            chainLength += math.sqrt(distance2)
        else:
            nearbyPoints = False
    return chain, chainLength

def SlidingPearsonRSquared(pointListX, pointListY, chain, pointsPerSlide):
    chainX = pointListX.take(chain)
    chainY = pointListY.take(chain)
    if len(chain) <= pointsPerSlide:
        return TSFO.RSquared(chainX, chainY)
    else:
        sumChainRSquared = 0
        n = len(chainX) - pointsPerSlide + 1
        for i in range(n):
            subChainX = chainX[i:i+pointsPerSlide]
            subChainY = chainY[i:i+pointsPerSlide]
            sumChainRSquared += TSFO.RSquared(subChainX, subChainY)
        return sumChainRSquared / n


def GetChainInfo(driftCoord, wireCoord, rectWidth, rectHeight, pointsPerSlide):
    chainCount = 0
    sumLengthRatios = 0
    sumChainRSquareds = 0
    availablePointIndices = list(range(0, len(driftCoord)))
    chains = []
    while availablePointIndices:    # While pfo hit list is not empty
        chain, chainLength = CreatePointChain2(driftCoord, wireCoord, rectWidth, rectHeight, availablePointIndices)
        chains.append(chain)
        if len(chain) > 1:
            startPoint = chain[0]
            endPoint = chain[-1]
            lengthRatio = math.sqrt(Distance2(driftCoord[startPoint], wireCoord[startPoint], driftCoord[endPoint], wireCoord[endPoint])) / chainLength
        else:
            lengthRatio = 0
        chainCount += 1
        sumLengthRatios += lengthRatio
        sumChainRSquareds += SlidingPearsonRSquared(driftCoord, wireCoord, chain, pointsPerSlide)
    avgLengthRatio = sumLengthRatios / chainCount
    avgChainRSquareds = sumChainRSquareds / chainCount
    return chainCount, avgLengthRatio, avgChainRSquareds, chains


def GetFeature(pfo, rectWidth=5, rectHeight=5, pointsPerSlide=10):
    chainCount, avgLengthRatio, avgChainRSquareds, chains = GetChainInfo(pfo.driftCoordW, pfo.wireCoordW, rectWidth, rectHeight, pointsPerSlide)
    print(chainCount)
    return { "F2a": chainCount, "F2b": avgLengthRatio, "F2c": avgChainRSquareds }
