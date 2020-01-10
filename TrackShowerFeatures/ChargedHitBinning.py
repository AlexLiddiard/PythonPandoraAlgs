# This module is for track/shower algorithm #4
import statistics
import math
import numpy as np
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.PCAnalysis as pca

# This function counts how much charge falls in a set of bins of a given width.
# Empty bins are ignored.

def GetBinCharge(xCoordsArray, chargeArray, binWidth):
    if len(xCoordsArray) == 0:
        return []

    caloHitsArray = np.column_stack((xCoordsArray, chargeArray))
    caloHitsArray = caloHitsArray[xCoordsArray.argsort()]
    # The upper bound used for checking if a number falls in the current bin
    upperBound = caloHitsArray[0][0] + binWidth
    # The charge in each bin
    binCharges = []

    charge = 0
    for x in caloHitsArray:
        if x[0] < upperBound:
            # The number falls into the current bin
            charge += x[1]
        else:
            if charge > 0:
                # Only consider non-empty bins
                binCharges.append(charge)
            # Find the bin that this number falls into
            while x[0] > upperBound:
                upperBound += binWidth
            charge = x[1]
    return binCharges


# Rotate a set of points clockwise by angle theta
# Note the variable: tan = tan(theta) = gradient

def RotatePointsClockwise(xCoords, yCoords, tan):
    sin, cos = TanToSinCos(tan)
    xCoordsNew = xCoords * cos + yCoords * sin
    yCoordsNew = yCoords * cos - xCoords * sin
    return xCoordsNew, yCoordsNew

def TanToSinCos(tan):
    cos = 1 / math.sqrt(1 + tan * tan)
    sin = tan * cos
    return sin, cos


def GetRotatedBinStdevOLS(driftCoord, wireCoord, binWidth, minBins, energyArray):
    a, b, r = lr.OLS(driftCoord, wireCoord)

    # Rotate the coords so that any tracks are roughly parallel to the x axis.
    # Prevents tracks from having hits in very few bins, giving high stdev.
    driftCoordRotated = RotatePointsClockwise(driftCoord, wireCoord, b)[0]
    rotatedBins = GetBinCharge(driftCoordRotated, energyArray, binWidth)

    # Ensure there are enough bins
    if len(rotatedBins) >= minBins:
        # Get stdev of the bin counts
        return statistics.stdev(rotatedBins)
    else:
        # Insufficient bins
        return -1

def GetRotatedBinStdevPCA(driftCoord, wireCoord, binWidth, minBins, energyArray):
    driftCoordRotated = pca.PcaReduce2D(driftCoord, wireCoord)[0]
    rotatedBins = GetBinCharge(driftCoordRotated, energyArray, binWidth)

    # Ensure there are enough bins
    if len(rotatedBins) >= minBins:
        # Get stdev of the bin counts
        return statistics.stdev(rotatedBins)/np.mean(energyArray)
    else:
        # Insufficient bins
        return -1


def GetFeatures(pfo, wireViews, binWidth=5, minBins=3):
    featureDict = {}
    if wireViews[0]:
        featureDict.update({ "ChargedBinnedHitStdU" : GetRotatedBinStdevPCA(pfo.driftCoordU, pfo.wireCoordU, binWidth, minBins, pfo.energyU)})
    if wireViews[1]:
        featureDict.update({ "ChargedBinnedHitStdV" : GetRotatedBinStdevPCA(pfo.driftCoordV, pfo.wireCoordV, binWidth, minBins, pfo.energyV)})
    if wireViews[2]:
        featureDict.update({ "ChargedBinnedHitStdW" : GetRotatedBinStdevPCA(pfo.driftCoordW, pfo.wireCoordW, binWidth, minBins, pfo.energyW)})
    return featureDict
