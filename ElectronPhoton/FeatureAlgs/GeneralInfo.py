# Gets the general PFO info used for analysis

def GetFeatures(pfo, calculateViews):
    pfoDataDict = {
        'fileName': pfo.fileName,
        'eventId': pfo.eventId,
        'pfoId': pfo.pfoId,
        'mcNuanceCode': pfo.mcNuanceCode,
        'mcPdgCode': pfo.mcPdgCode,
        'mcpMomentum': pfo.mcpMomentum,
        'nuMcPdgCode': pfo.nuMcPdgCode,
        'nuMcpMomentum': pfo.nuMcpMomentum,
        'hierarchyTier': pfo.hierarchyTier,
        'minCoordX': min(pfo.xCoord3D),
        'minCoordY': min(pfo.yCoord3D),
        'minCoordZ': min(pfo.zCoord3D),
        'maxCoordX': max(pfo.xCoord3D),
        'maxCoordY': max(pfo.yCoord3D),
        'maxCoordZ': max(pfo.zCoord3D)
    }
    if calculateViews["U"]:
        pfoDataDict.update({
        'nHitsU': pfo.nHitsPfoU,
        'purityU': pfo.PurityU(),
        'completenessU': pfo.CompletenessU()
        })
    if calculateViews["V"]:
        pfoDataDict.update({
        'nHitsV': pfo.nHitsPfoV,
        'purityV': pfo.PurityV(),
        'completenessV': pfo.CompletenessV()
        })
    if calculateViews["W"]:
        pfoDataDict.update({
        'nHitsW': pfo.nHitsPfoW,
        'purityW': pfo.PurityW(),
        'completenessW': pfo.CompletenessW()
        })
    if calculateViews["3D"]:
        pfoDataDict.update({'nHits3D': pfo.nHitsPfo3D})
    return pfoDataDict