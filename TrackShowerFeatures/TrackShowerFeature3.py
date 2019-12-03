# This module is for track/shower feature #3
import numpy as np
import math as m
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1

def GetTrianglarSpan(driftCoord, wireCoord, vertex, hitFraction):
    openingAngleIndex = m.floor(hitFraction * (len(driftCoord) - 1))
    if len(driftCoord) < 2 or openingAngleIndex == 0:
        return -1, -1
    transDriftCoord = driftCoord - vertex[0]
    transWireCoord = wireCoord - vertex[1]
    principleAxisGradient = tsf0.OLSNoIntercept(transDriftCoord, transWireCoord)
    transDriftCoord, transWireCoord = tsf1.RotatePointsClockwise(transDriftCoord, transWireCoord, principleAxisGradient)
    hitAnglesFromAxis = np.abs(np.arctan(transWireCoord / transDriftCoord))
    hitAnglesFromAxis.sort()
    openingAngle = 2 * hitAnglesFromAxis[openingAngleIndex]
    distance = np.amax(np.abs(transDriftCoord[:openingAngleIndex]))
    return openingAngle, distance


def GetFeature(pfo, hitFraction=0.8):
    openingAngle, distance = GetTrianglarSpan(pfo.driftCoordW, pfo.wireCoordW, pfo.vertex, hitFraction)
    return { "F3a": openingAngle, "F3b": distance }
