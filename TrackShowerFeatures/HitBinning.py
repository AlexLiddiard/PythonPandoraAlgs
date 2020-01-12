# This module is for track/shower algorithm #1
import math
import numpy as np
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.PCAnalysis as pca
import TrackShowerFeatures.AngularSpan as asp

# This function counts how many numbers fall in a set of bins of a given width.
# Empty bins are ignored.

def GetBinCounts(numbers, binWidth):
    if len(numbers) == 0:
        return []

    numbers.sort()
    # The upper bound used for checking if a number falls in the current bin
    upperBound = numbers[0]
    # The numbers in each bin
    binCounts = []

    count = 0
    for n in numbers:
        if n < upperBound:
            # The number falls into the current bin
            count += 1
        else:
            if count > 0:
                # Only consider non-empty bins
                binCounts.append(count)
            # Find the bin that this number falls into
            while n > upperBound:
                upperBound += binWidth
            count = 1
    return binCounts


# Rotate a set of points clockwise by angle theta
# Note the variable: tan = tan(theta) = gradient

def RotatePointsClockwise(xCoords, yCoords, sin, cos):
    xCoordsNew = xCoords * cos + yCoords * sin
    yCoordsNew = yCoords * cos - xCoords * sin
    return xCoordsNew, yCoordsNew

def TanToSinCos(tan):
    cos = 1 / math.sqrt(1 + tan * tan)
    sin = tan * cos
    return sin, cos


def GetRotatedBinStdevOLS(driftCoord, wireCoord, binWidth, minBins):
    a, b, r = lr.OLS(driftCoord, wireCoord)

    # Rotate the coords so that any tracks are roughly parallel to the x axis.
    # Prevents tracks from having hits in very few bins, giving high stdev.
    driftCoordRotated = RotatePointsClockwise(driftCoord, wireCoord, *TanToSinCos(b))[0]
    rotatedBins = GetBinCounts(driftCoordRotated, binWidth)

    # Ensure there are enough bins
    if len(rotatedBins) >= minBins:
        # Get stdev of the bin counts
        return np.std(rotatedBins)
    else:
        # Insufficient bins
        return -1

def GetRotatedBinStdevPCA(driftCoord, wireCoord, binWidth, minBins):
    driftCoordRotated = pca.PcaReduce((driftCoord, wireCoord))[0]
    rotatedBins = GetBinCounts(driftCoordRotated, binWidth)

    # Ensure there are enough bins
    if len(rotatedBins) >= minBins:
        # Get stdev of the bin counts
        return np.std(rotatedBins)
    else:
        # Insufficient bins
        return -1

def GetRadialBinStdev(coordSets, vertex, hitFraction, binWidth, minBins):  
    distances, hitsInside = GetFilteredVertexDistances(coordSets, vertex, hitFraction)
    binCounts = GetBinCounts(distances, binWidth)
    if len(binCounts) < minBins:
        return -1
    return np.std(binCounts)

def GetFilteredVertexDistances(coordSets, vertex, hitFraction):
    if len(coordSets[0]) == 0:
        return np.array([]), np.array([])
    coordSetsNew = pca.PcaReduce(coordSets, vertex)
    hitAnglesFromAxis, halfOpeningAngle = asp.CalcAngles(coordSetsNew, hitFraction)
    hitsInside = (hitAnglesFromAxis >= 0) & (hitAnglesFromAxis <= halfOpeningAngle)
    try:
        distances = np.linalg.norm(coordSetsNew[:,hitsInside], axis=0)
    except:
        print(coordSetsNew)
    return distances, hitsInside

def GetFeatures(pfo, calculateViews, binWidth=1, minBins=3, hitFraction=1):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ 
            "BinnedHitStdU": GetRotatedBinStdevPCA(pfo.driftCoordU, pfo.wireCoordU, binWidth, minBins),
            "RadialBinStdU": GetRadialBinStdev((pfo.driftCoordU, pfo.wireCoordU), pfo.vertexU, hitFraction, binWidth, minBins)
        })
    if calculateViews["V"]:
        featureDict.update({
            "BinnedHitStdV": GetRotatedBinStdevPCA(pfo.driftCoordV, pfo.wireCoordV, binWidth, minBins),
            "RadialBinStdV": GetRadialBinStdev((pfo.driftCoordV, pfo.wireCoordV), pfo.vertexV, hitFraction, binWidth, minBins)
        })
    if calculateViews["W"]:
        featureDict.update({
            "BinnedHitStdW": GetRotatedBinStdevPCA(pfo.driftCoordW, pfo.wireCoordW, binWidth, minBins),
            "RadialBinStdW": GetRadialBinStdev((pfo.driftCoordW, pfo.wireCoordW), pfo.vertexW, hitFraction, binWidth, minBins)
        })
    if calculateViews["3D"]:
        featureDict.update({ 
            "RadialBinStd3D": GetRadialBinStdev((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.vertex3D, hitFraction, binWidth, minBins)
        })
    return featureDict
