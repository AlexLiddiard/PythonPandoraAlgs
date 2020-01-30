# This module is for track/shower algorithm #4
import math as m
import numpy as np
import LinearRegression as lr
import PCAnalysis as pca

# This function counts how much charge falls in a set of bins of a given width.
# Empty bins are ignored.

def GetBinCharge(xCoordsArray, chargeArray, binWidth):
    if len(xCoordsArray) == 0:
        return []

    sort = xCoordsArray.argsort()
    # The upper bound used for checking if a number falls in the current bin
    upperBound = xCoordsArray[sort[0]] + binWidth
    # The charge in each bin
    binCharges = []
    chargeSum = 0
    for xCoord, charge in zip(xCoordsArray[sort], chargeArray[sort]):
        if xCoord < upperBound:
            # The number falls into the current bin
            chargeSum += charge
        else:
            if chargeSum > 0:
                # Only consider non-empty bins
                binCharges.append(chargeSum)
            # Find the bin that this number falls into
            while xCoord > upperBound:
                upperBound += binWidth
            chargeSum = charge
    return binCharges


# Rotate a set of points clockwise by angle theta
# Note the variable: tan = tan(theta) = gradient

def RotatePointsClockwise(xCoords, yCoords, tan):
    sin, cos = TanToSinCos(tan)
    xCoordsNew = xCoords * cos + yCoords * sin
    yCoordsNew = yCoords * cos - xCoords * sin
    return xCoordsNew, yCoordsNew

def TanToSinCos(tan):
    cos = 1 / m.sqrt(1 + tan * tan)
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
        return np.std(rotatedBins)
    else:
        # Insufficient bins
        return m.nan

def GetRotatedBinStdevPCA(driftCoord, wireCoord, binWidth, minBins, energyArray):
    lCoord, tCoord = pca.PcaReduce((driftCoord, wireCoord))
    rotatedBins = GetBinCharge(lCoord, energyArray, binWidth)

    # Ensure there are enough bins
    if len(rotatedBins) < minBins:
        return m.nan
    # Get stdev of the bin counts
    return np.std(rotatedBins)/np.mean(energyArray)


def GetFeatures(pfo, calculateViews, binWidth=2, minBins=3):
    featureDict = {}
    if calculateViews["U"]:
        featureDict.update({ "ChargedBinnedHitStdU" : GetRotatedBinStdevPCA(pfo.driftCoordU, pfo.wireCoordU, binWidth, minBins, pfo.energyU)})
    if calculateViews["V"]:
        featureDict.update({ "ChargedBinnedHitStdV" : GetRotatedBinStdevPCA(pfo.driftCoordV, pfo.wireCoordV, binWidth, minBins, pfo.energyV)})
    if calculateViews["W"]:
        featureDict.update({ "ChargedBinnedHitStdW" : GetRotatedBinStdevPCA(pfo.driftCoordW, pfo.wireCoordW, binWidth, minBins, pfo.energyW)})
    return featureDict
