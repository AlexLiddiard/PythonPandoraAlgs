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
    #DisplayPfo(pfo)
    featureDict = {}

    # Some invalid vertices break calculations e.g. for correlation, this is a workaround
    validVertex = (
        pfo.vertex3D[0] > MicroBooneGeo.RangeX[0] and
        pfo.vertex3D[0] < MicroBooneGeo.RangeX[1] and
        pfo.vertex3D[1] > MicroBooneGeo.RangeY[0] and
        pfo.vertex3D[1] < MicroBooneGeo.RangeY[1] and
        pfo.vertex3D[2] > MicroBooneGeo.RangeZ[0] and
        pfo.vertex3D[2] < MicroBooneGeo.RangeZ[1]
    )
    
    openingAngle, distance = -1, -1
    if validVertex:
        openingAngle, distance = GetConicSpan(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, hitFraction)
    featureDict.update({ "AngularSpan3D": openingAngle, "LongitudinalSpan3D": distance })
    if wireViews[0]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordU, pfo.wireCoordU, pfo.vertexU, hitFraction)
        featureDict.update({ "AngularSpanU": openingAngle, "LongitudinalSpanU": distance })
    if wireViews[1]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordV, pfo.wireCoordV, pfo.vertexV, hitFraction)
        featureDict.update({ "AngularSpanV": openingAngle, "LongitudinalSpanV": distance })
    if wireViews[2]:
        if validVertex:
            openingAngle, distance = GetTrianglarSpan(pfo.driftCoordW, pfo.wireCoordW, pfo.vertexW, hitFraction)
        featureDict.update({ "AngularSpanW": openingAngle, "LongitudinalSpanW": distance })
    return featureDict
