# This module is for track/shower feature #3
import numpy as np
import math as m
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
#import matplotlib.pyplot as plt
#from PfoGraphAnalysis import DisplayPfo

def GetTrianglarSpan(driftCoord, wireCoord, vertexDriftCoord, vertexWireCoord, hitFraction):
    if len(driftCoord) < 2:
        return -1, -1
    transDriftCoord = driftCoord - vertexDriftCoord
    transWireCoord = wireCoord - vertexWireCoord
    principleAxisGradient = tsf0.OLSNoIntercept(transDriftCoord, transWireCoord)
    transDriftCoord, transWireCoord = tsf1.RotatePointsClockwise(transDriftCoord, transWireCoord, principleAxisGradient)

    transDriftCoordCheck = transDriftCoord > 0
    if 2 * np.sum(transDriftCoordCheck) < len(driftCoord):
        transDriftCoordCheck = np.invert(transDriftCoordCheck)
    transDriftCoord = transDriftCoord[transDriftCoordCheck]
    transWireCoord = transWireCoord[transDriftCoordCheck]

    openingAngleIndex = m.floor(hitFraction * (len(transDriftCoord) - 1))
    hitAnglesFromAxis = np.arctan2(np.abs(transWireCoord), np.abs(transDriftCoord))
    hitAnglesFromAxis.sort()

    #fig = plt.figure(figsize=(13,10))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.set_aspect('equal', 'box')
    #ax.plot(transDriftCoord, transWireCoord, linewidth=0, marker='o', markersize=10)
    #plt.show()

    #plt.hist(hitAnglesFromAxis)
    #plt.show()

    openingAngle = 2 * hitAnglesFromAxis[openingAngleIndex]
    distance = np.amax(np.abs(transDriftCoord[:(openingAngleIndex + 1)]))
    return openingAngle, distance


def GetFeature(pfo, wireViews, hitFraction=0.7):
    #DisplayPfo(pfo)
    featureDict = {}
    if wireViews[0]:
        openingAngle, distance = GetTrianglarSpan(pfo.driftCoordU, pfo.wireCoordU, pfo.vertex[0], 0.5 * pfo.vertex[2] - 0.8660254 * pfo.vertex[1], hitFraction)
        featureDict.update({ "AngularSpanU": openingAngle, "LongitudinalSpanU": distance })
    if wireViews[1]:
        openingAngle, distance = GetTrianglarSpan(pfo.driftCoordV, pfo.wireCoordV, pfo.vertex[0], 0.5 * pfo.vertex[2] + 0.8660254 * pfo.vertex[1], hitFraction)
        featureDict.update({ "AngularSpanV": openingAngle, "LongitudinalSpanV": distance })
    if wireViews[2]:
        openingAngle, distance = GetTrianglarSpan(pfo.driftCoordW, pfo.wireCoordW, pfo.vertex[0], pfo.vertex[2], hitFraction)
        featureDict.update({ "AngularSpanW": openingAngle, "LongitudinalSpanW": distance })
    return featureDict
