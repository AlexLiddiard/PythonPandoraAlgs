# This module is for track/shower feature #3
import AlgorithmConfig as cfg
import numpy as np
import math as m
import PCAnalysis as pca

def CalcAngles(coordSetsReduced, hitFraction):
    lCoord = coordSetsReduced[0]
    tCoords = coordSetsReduced[1:]
    if len(lCoord) == 0:
        return np.array([]), m.nan
    if (lCoord > 0).sum() < len(lCoord) / 2:
        lCoord *= -1 # Flip the longitudinal coords so the majority are positive
    if (lCoord > 0).sum() == 0: # Rarely there is an array of (0, 0, 0)s, and it will fail without this check
        return np.array([]), m.nan
    hitAnglesFromAxis = np.arctan2(np.linalg.norm(tCoords, axis=0), lCoord)
    halfOpeningAngle = np.percentile(hitAnglesFromAxis[lCoord > 0], hitFraction * 100)
    return hitAnglesFromAxis, halfOpeningAngle

def GetConicSpan(coordSets, vertex, hitFraction):
    if len(coordSets[0]) == 0:
        return m.nan, m.nan
    coordSetsReduced = pca.PcaReduce(coordSets, vertex)
    hitAnglesFromAxis, halfOpeningAngle = CalcAngles(coordSetsReduced, hitFraction)
    if m.isnan(halfOpeningAngle):
        return m.nan, m.nan
    hitsInside = (hitAnglesFromAxis >= 0) & (hitAnglesFromAxis <= halfOpeningAngle)
    distance = np.amax(np.abs(coordSetsReduced[0,hitsInside]))
    return halfOpeningAngle * 2, distance

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    openingAngle, distance = m.nan, m.nan
    if calculateViews["U"]:
        if pfo.ValidVertex():
            openingAngle, distance = GetConicSpan((pfo.driftCoordU, pfo.wireCoordU), pfo.vertexU, cfg.angularSpan["hitFraction"])
        featureDict.update({ "AngularSpanU": openingAngle, "LongitudinalSpanU": distance })
    if calculateViews["V"]:
        if pfo.ValidVertex():
            openingAngle, distance = GetConicSpan((pfo.driftCoordV, pfo.wireCoordV), pfo.vertexV, cfg.angularSpan["hitFraction"])
        featureDict.update({ "AngularSpanV": openingAngle, "LongitudinalSpanV": distance })
    if calculateViews["W"]:
        if pfo.ValidVertex():
            openingAngle, distance = GetConicSpan((pfo.driftCoordW, pfo.wireCoordW), pfo.vertexW, cfg.angularSpan["hitFraction"])
        featureDict.update({ "AngularSpanW": openingAngle, "LongitudinalSpanW": distance })
    if calculateViews["3D"]:
        if pfo.ValidVertex():
            openingAngle, distance = GetConicSpan((pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D), pfo.vertex3D, cfg.angularSpan["hitFraction"])
        featureDict.update({ "AngularSpan3D": openingAngle, "LongitudinalSpan3D": distance })
    return featureDict
