import numpy as np
import PCAnalysis as pca
import math as m

def VertexSeparation(pfoVertex, pfoInteractionVertex, coordSet):
    if pfoInteractionVertex is None:
        return m.nan
    vertexSeparation = np.linalg.norm(pfoInteractionVertex - pfoVertex)
    try:
        return vertexSeparation#/pca.Pca(coordSet, pfoVertex)[0][-1]
    except:
        return m.nan

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["U"]:
        vertexSeparation = VertexSeparation(pfo.vertexU, pfo.interactionVertexU, (pfo.driftCoordU, pfo.wireCoordU))
        featureDict.update({ "VertexSepU": vertexSeparation})
    if calculateViews["V"]:
        vertexSeparation = VertexSeparation(pfo.vertexV, pfo.interactionVertexV, (pfo.driftCoordV, pfo.wireCoordV))
        featureDict.update({ "VertexSepV": vertexSeparation})
    if calculateViews["W"]:
        vertexSeparation = VertexSeparation(pfo.vertexW, pfo.interactionVertexW, (pfo.driftCoordW, pfo.wireCoordW))
        featureDict.update({ "VertexSepW": vertexSeparation})
    if calculateViews["3D"]:
        vertexSeparation = VertexSeparation(pfo.vertex3D, pfo.interactionVertex3D, (pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D))
        featureDict.update({ "VertexSep3D": vertexSeparation})
    return featureDict
