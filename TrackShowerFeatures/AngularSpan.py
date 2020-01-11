# This module is for track/shower feature #3
import numpy as np
import math as m
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.HitBinning as hb
import TrackShowerFeatures.PCAnalysis as pca
from PfoGraphicalAnalyser import MicroBooneGeo
#import matplotlib.pyplot as plt
#from PfoGraphAnalysis import DisplayPfo

def GetTrianglarSpan(driftCoord, wireCoord, vertex, hitFraction):
    if len(driftCoord) < 2:
        return -1, -1
    transDriftCoord, transWireCoord = pca.PcaReduce((driftCoord, wireCoord), vertex)
    hitAnglesFromAxis, openingAngleIndex = CalcAngles(transDriftCoord, transWireCoord, hitFraction)
    distance = np.amax(np.abs(transDriftCoord[:(openingAngleIndex + 1)]))
    openingAngle = 2 * hitAnglesFromAxis[openingAngleIndex]
    return openingAngle, distance

def CalcAngles(transDriftCoord, transWireCoord, hitFraction):
    transDriftCoordCheck = transDriftCoord > 0
    if 2 * np.sum(transDriftCoordCheck) < len(transDriftCoord):
        transDriftCoordCheck = np.invert(transDriftCoordCheck)
    transDriftCoord = transDriftCoord[transDriftCoordCheck]
    transWireCoord = transWireCoord[transDriftCoordCheck]
    openingAngleIndex = m.floor(hitFraction * (len(transDriftCoord) - 1))
    hitAnglesFromAxis = np.arctan2(np.abs(transWireCoord), np.abs(transDriftCoord))
    return hitAnglesFromAxis, openingAngleIndex

def GetConicSpan(xCoord, yCoord, zCoord, vertex, hitFraction):
    if len(xCoord) < 2:
        return -1, -1

    newCoordSets = pca.PcaReduce((xCoord, yCoord, zCoord), vertex)
    xCoordCheck = newCoordSets[0] > 0
    if 2 * np.sum(xCoordCheck) < len(xCoordCheck):
        xCoordCheck = np.invert(xCoordCheck)
    newCoordSets = newCoordSets[:,xCoordCheck]

    openingAngleIndex = m.floor(hitFraction * (len(newCoordSets[0]) - 1))
    hitAnglesFromAxis = np.arctan2(np.sqrt(np.square(newCoordSets[1]) + np.square(newCoordSets[2])), np.abs(newCoordSets[0]))
    hitAnglesFromAxis.sort()

    distance = np.amax(np.abs(newCoordSets[0,:(openingAngleIndex + 1)]))
    openingAngle = 2 * hitAnglesFromAxis[openingAngleIndex]
    return openingAngle, distance


def GetFeatures(pfo, wireViews, hitFraction=0.7):
    featureDict = {}
    openingAngle, distance = -1, -1
    if pfo.ValidVertex():
        openingAngle, distance = GetConicSpan(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, hitFraction)
    featureDict.update({ "AngularSpan3D": openingAngle, "LongitudinalSpan3D": distance })
    if wireViews[0]:
        if pfo.ValidVertex():
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, hitFraction)
        featureDict.update({ "AngularSpanU": openingAngle, "LongitudinalSpanU": distance })
    if wireViews[1]:
        if pfo.ValidVertex():
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, hitFraction)
        featureDict.update({ "AngularSpanV": openingAngle, "LongitudinalSpanV": distance })
    if wireViews[2]:
        if pfo.ValidVertex():
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, hitFraction)
        featureDict.update({ "AngularSpanW": openingAngle, "LongitudinalSpanW": distance })
    return featureDict
