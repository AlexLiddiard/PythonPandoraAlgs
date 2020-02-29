# This module is for track/shower algorithm #1
import AlgorithmConfig as cfg
import math as m
import numpy as np
import LinearRegression as lr
import PCAnalysis as pca
import AngularSpan as asp

# This function counts how many numbers fall in a set of bins of a given width.
# Empty bins are ignored.

def GetBinCounts(coords, charges, binWidth):
    if len(coords) == 0:
        return [], []

    sort = coords.argsort()
    # The upper bound used for checking if a number falls in the current bin
    upperBound = coords[sort[0]] + binWidth
    # The count in each bin
    hitCounts = []
    chargeCounts = []
    hitCount = 0
    chargeCount = 0
    for xCoord, charge in zip(coords[sort], charges[sort]):
        if xCoord < upperBound:
            # The number falls into the current bin
            hitCount += 1
            chargeCount += charge
        else:
            if hitCount > 0:
                # Only consider non-empty bins
                hitCounts.append(hitCount)
                chargeCounts.append(chargeCount)
            # Find the bin that this number falls into
            while xCoord > upperBound:
                upperBound += binWidth
            hitCount = 1
            chargeCount = charge
    return hitCounts, chargeCounts


# Rotate a set of points clockwise by angle theta
# Note the variable: tan = tan(theta) = gradient

def RotatePointsClockwise(xCoords, yCoords, sin, cos):
    xCoordsNew = xCoords * cos + yCoords * sin
    yCoordsNew = yCoords * cos - xCoords * sin
    return xCoordsNew, yCoordsNew

def TanToSinCos(tan):
    cos = 1 / m.sqrt(1 + tan * tan)
    sin = tan * cos
    return sin, cos


def GetRotatedBinStdevOLS(driftCoord, wireCoord, charges, binWidth, minBins,):
    a, b, r = lr.OLS(driftCoord, wireCoord)

    # Rotate the coords so that any tracks are roughly parallel to the x axis.
    # Prevents tracks from having hits in very few bins, giving high stdev.
    driftCoordRotated = RotatePointsClockwise(driftCoord, wireCoord, *TanToSinCos(b))[0]
    hitCounts, chargeCounts = GetBinCounts(driftCoordRotated, charges, binWidth)

    if len(hitCounts) < minBins:
        return m.nan, m.nan
    # Get stdev of the bin counts
    return np.std(hitCounts), np.std(chargeCounts) / np.mean(charges)

def GetRotatedBinStdevPCA(driftCoord, wireCoord, charges, binWidth, minBins):
    driftCoordRotated = pca.PcaReduce((driftCoord, wireCoord))[0]
    hitCounts, chargeCounts = GetBinCounts(driftCoordRotated, charges, binWidth)

    # Ensure there are enough bins
    if len(hitCounts) < minBins:
        return m.nan, m.nan
    # Get stdev of the bin counts
    return np.std(hitCounts), np.std(chargeCounts) / np.mean(charges)

def GetRadialBinStdev(coordSets, charges, vertex, hitFraction, binWidth, minBins):  
    distances, hitsInside = GetFilteredVertexDistances(coordSets, vertex, hitFraction)
    if len(hitsInside) == 0:
        return m.nan, m.nan
    hitCounts, chargeCounts = GetBinCounts(distances, charges[hitsInside], binWidth)

    # Ensure there are enough bins
    if len(hitCounts) < minBins:
        return m.nan, m.nan
    # Get stdev of the bin counts
    return np.std(hitCounts), np.std(chargeCounts) / np.mean(charges)

def GetFilteredVertexDistances(coordSets, vertex, hitFraction):
    if len(coordSets[0]) == 0:
        return np.array([]), np.array([])
    coordSetsNew = pca.PcaReduce(coordSets, vertex)
    hitAnglesFromAxis, halfOpeningAngle = asp.CalcAngles(coordSetsNew, hitFraction)
    hitsInside = (hitAnglesFromAxis >= 0) & (hitAnglesFromAxis <= halfOpeningAngle)
    distances = np.linalg.norm(coordSetsNew[:,hitsInside], axis=0)
    return distances, hitsInside

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["U"]:
        BinnedHitStd, BinnedChargeStd = GetRotatedBinStdevPCA(pfo.driftCoordU, pfo.wireCoordU, pfo.energyU, cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        RadialBinHitStd, RadialBinChargeStd = GetRadialBinStdev((pfo.driftCoordU, pfo.wireCoordU), pfo.energyU, pfo.vertexU, cfg.hitBinning["hitFraction"], cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        featureDict.update({ 
            "BinnedHitStdU": BinnedHitStd, "BinnedChargeStdU": BinnedChargeStd, "RadialBinHitStdU": RadialBinHitStd, "RadialBinChargeStdU": RadialBinChargeStd
        })
    if calculateViews["V"]:
        BinnedHitStd, BinnedChargeStd = GetRotatedBinStdevPCA(pfo.driftCoordV, pfo.wireCoordV, pfo.energyV, cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        RadialBinHitStd, RadialBinChargeStd = GetRadialBinStdev((pfo.driftCoordV, pfo.wireCoordV), pfo.energyV, pfo.vertexV, cfg.hitBinning["hitFraction"], cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        featureDict.update({ 
            "BinnedHitStdV": BinnedHitStd, "BinnedChargeStdV": BinnedChargeStd, "RadialBinHitStdV": RadialBinHitStd, "RadialBinChargeStdV": RadialBinChargeStd
        })
    if calculateViews["W"]:
        BinnedHitStd, BinnedChargeStd = GetRotatedBinStdevPCA(pfo.driftCoordW, pfo.wireCoordW, pfo.energyW, cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        RadialBinHitStd, RadialBinChargeStd = GetRadialBinStdev((pfo.driftCoordW, pfo.wireCoordW), pfo.energyW, pfo.vertexW, cfg.hitBinning["hitFraction"], cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        featureDict.update({ 
            "BinnedHitStdW": BinnedHitStd, "BinnedChargeStdW": BinnedChargeStd, "RadialBinHitStdW": RadialBinHitStd, "RadialBinChargeStdW": RadialBinChargeStd
        })
    if calculateViews["3D"]:
        RadialBinHitStd, RadialBinChargeStd = GetRadialBinStdev((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.energy3D, pfo.vertex3D, cfg.hitBinning["hitFraction"], cfg.hitBinning["binWidth"], cfg.hitBinning["minBins"])
        featureDict.update({ 
            "RadialBinHitStd3D": RadialBinHitStd, "RadialBinChargeStd3D": RadialBinChargeStd
        })
    return featureDict
