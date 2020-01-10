# This module is for track/shower feature #3
import numpy as np
import math as m
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.HitBinning as hb
import TrackShowerFeatures.PCAnalysis as pca
from PfoGraphicalAnalyser import MicroBooneGeo
#import matplotlib.pyplot as plt
#from PfoGraphAnalysis import DisplayPfo

def GetTrianglarSpan(driftCoord, wireCoord, vertexDriftCoord, vertexWireCoord, hitFraction):
    if len(driftCoord) < 2:
        return -1, -1
    
    transDriftCoord = driftCoord - vertexDriftCoord
    transWireCoord = wireCoord - vertexWireCoord
    transDriftCoord, transWireCoord = pca.PcaReduce2D(transDriftCoord, transWireCoord, 0, 0)

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

    distance = np.amax(np.abs(transDriftCoord[:(openingAngleIndex + 1)]))
    openingAngle = 2 * hitAnglesFromAxis[openingAngleIndex]
    return openingAngle, distance


def GetFeatures(pfo, wireViews, hitFraction=0.7):
    #DisplayPfo(pfo)
    featureDict = {}

    # Some invalid vertices break calculations e.g. for correlation, this is a workaround
    validVertex = (
        pfo.vertex[0] > MicroBooneGeo.RangeX[0] and
        pfo.vertex[0] < MicroBooneGeo.RangeX[1] and
        pfo.vertex[1] > MicroBooneGeo.RangeY[0] and
        pfo.vertex[1] < MicroBooneGeo.RangeY[1] and
        pfo.vertex[2] > MicroBooneGeo.RangeZ[0] and
        pfo.vertex[2] < MicroBooneGeo.RangeZ[1]
    )
    
    openingAngle, distance = -1, -1
    if wireViews[0]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordU, pfo.wireCoordU, pfo.vertex[0], 0.5 * pfo.vertex[2] - 0.8660254 * pfo.vertex[1], hitFraction)
        featureDict.update({ "AngularSpanU": openingAngle, "LongitudinalSpanU": distance })
    if wireViews[1]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordV, pfo.wireCoordV, pfo.vertex[0], 0.5 * pfo.vertex[2] + 0.8660254 * pfo.vertex[1], hitFraction)
        featureDict.update({ "AngularSpanV": openingAngle, "LongitudinalSpanV": distance })
    if wireViews[2]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordW, pfo.wireCoordW, pfo.vertex[0], pfo.vertex[2], hitFraction)
        featureDict.update({ "AngularSpanW": openingAngle, "LongitudinalSpanW": distance })
    return featureDict
