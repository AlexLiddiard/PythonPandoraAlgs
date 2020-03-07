# Gets the general PFO info used for analysis

def GetFeatures(pfo, calculateViews):
    pfoDataDict = {
        'fileName': pfo.fileName,
        'eventId': pfo.eventId,
        'pfoId': pfo.pfoId,
        'hierarchyTier': pfo.hierarchyTier,
        'minCoordX3D': min(pfo.xCoord3D),
        'minCoordY3D': min(pfo.yCoord3D),
        'minCoordZ3D': min(pfo.zCoord3D),
        'maxCoordX3D': max(pfo.xCoord3D),
        'maxCoordY3D': max(pfo.yCoord3D),
        'maxCoordZ3D': max(pfo.zCoord3D),
        'mcNuanceCode': pfo.mcNuanceCode,
        'mcPdgCode': pfo.mcPdgCode,
        'mcpMomentum': pfo.mcpMomentum,
        'mcParentPdgCode': pfo.mcParentPdgCode,
        'mcNuPdgCode': pfo.incidentPfo.mcPdgCode,
        'mcNuMomentum':  pfo.incidentPfo.mcpMomentum,
        'mcHierarchyTier': pfo.mcHierarchyTier,
        'purityOverall': pfo.PurityOverall(),
        'completenessOverall': pfo.CompletenessOverall(),
    }
    if calculateViews["U"]:
        pfoDataDict.update({
        'nHitsU': pfo.nHitsPfoU,
        'purityU': pfo.PurityU(),
        'completenessU': pfo.CompletenessU(),
        #'hierarchyTierU': pfo.hierarchyTier,
        })
    if calculateViews["V"]:
        pfoDataDict.update({
        'nHitsV': pfo.nHitsPfoV,
        'purityV': pfo.PurityV(),
        'completenessV': pfo.CompletenessV(),
        #'hierarchyTierV': pfo.hierarchyTier,
        })
    if calculateViews["W"]:
        pfoDataDict.update({
        'nHitsW': pfo.nHitsPfoW,
        'purityW': pfo.PurityW(),
        'completenessW': pfo.CompletenessW(),
        #'hierarchyTierW': pfo.hierarchyTier,
        })
    if calculateViews["3D"]:
        pfoDataDict.update({
        'nHits3D': pfo.nHitsPfo3D,
        #'hierarchyTier3D': pfo.hierarchyTier,
        })
    return pfoDataDict
