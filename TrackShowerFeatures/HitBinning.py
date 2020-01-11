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

def GetRadialBinStdev(driftCoord, wireCoord, vertex, hitFraction, binWidth, minBins):    
    distances, selectedHitIndices = GetFilteredVertexDistances(driftCoord, wireCoord, vertex, hitFraction)
    binCounts = GetBinCounts(distances, binWidth)
    if len(binCounts) < minBins:
        return -1
    return np.std(binCounts)

def GetFilteredVertexDistances(driftCoord, wireCoord, vertex, hitFraction):
    driftCoordNew, wireCoordNew = pca.PcaReduce((driftCoord, wireCoord), vertex)
    hitAnglesFromAxis, openingAngleIndex, selectedHitIndices = asp.CalcAngles(driftCoordNew, wireCoordNew, hitFraction)
    selectedHitIndices = hitAnglesFromAxis <= hitAnglesFromAxis[openingAngleIndex]
    distances = np.sqrt(np.square(driftCoordNew[selectedHitIndices]) + np.square(wireCoordNew[selectedHitIndices]))
    return distances, selectedHitIndices

def GetFeatures(pfo, wireViews, binWidth=1, minBins=3, hitFraction=0.8):
    featureDict = {}
    if wireViews[0]:
        featureDict.update({ 
            "BinnedHitStdU": GetRotatedBinStdevPCA(pfo.driftCoordU, pfo.wireCoordU, binWidth, minBins),
            "RadialBinStdU": GetRadialBinStdev(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, hitFraction, binWidth, minBins)
        })
    if wireViews[1]:
        featureDict.update({
            "BinnedHitStdV": GetRotatedBinStdevPCA(pfo.driftCoordV, pfo.wireCoordV, binWidth, minBins),
            "RadialBinStdV": GetRadialBinStdev(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, hitFraction, binWidth, minBins)
        })
    if wireViews[2]:
        featureDict.update({
            "BinnedHitStdW": GetRotatedBinStdevPCA(pfo.driftCoordW, pfo.wireCoordW, binWidth, minBins),
            "RadialBinStdW": GetRadialBinStdev(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, hitFraction, binWidth, minBins)
        })
    return featureDict
