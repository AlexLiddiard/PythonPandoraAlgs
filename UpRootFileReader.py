# This module is used to transcribe PFO data from a ROOT file (stored as vectors) into python arrays.
import uproot as up
import numpy as np
import os
import PfoVertexFinder as pvf

# This class defines the data associated with each PFO.
class PfoClass(object):

    wireCoordErr = 0.3  # Sets wire coord error to 3 millimetres.
    particleTypes = {
        11: "Electron",
        -11: "Positron",
        12: "Electron Neutrino",
        -12: "Electron Anti-Neutrino",
        13: "Positive Muon",
        -13: "Negative Muon",
        14: "Muon Neutrino",
        -14: "Muon Anti-Neutrino",
        22: "Photon",
        111: "Neutral Pion",
        130: "Neutral Kaon",
        211: "Positive Pion",
        -211: "Negative Pion",
        310: "Neutral Kaon Short",
        321: "Positive Kaon",
        -321: "Negative Kaon",
        2212: "Proton",
        2112: "Neutron",
        3112: "Negative Sigma",
        3122: "Lambda",
        3222: "Positive Sigma",
        1000010020: "Deuteron",
        1000010030: "Triton"
    }

    def __init__(self, pfo, fileName):
        # PFO identification + relations
        self.fileName = fileName
        self.eventId = pfo.eventId
        self.pfoId = pfo.pfoId
        self.parentPfoId = pfo.parentPfoId
        self.daughterPfoIds = np.array(pfo.daughterPfoIds)
        self.heirarchyTier = pfo.hierarchyTier

        # Simulation info
        self.mcNuanceCode = pfo.mcNuanceCode
        self.mcPdgCode = pfo.mcPdgCode
        self.mcpMomentum = pfo.mcpMomentum
        self.mcNeutrinoEnergy = 0 # Associated data, to be set later

        # U view
        self.driftCoordU = np.array(pfo.driftCoordU, dtype = np.double)
        self.driftCoordErrU = np.array(pfo.driftCoordErrorU, dtype = np.double)
        self.wireCoordU = np.array(pfo.wireCoordU, dtype = np.double)
        self.energyU = np.array(pfo.energyU, dtype = np.double)
        self.nHitsPfoU = pfo.nHitsPfoU
        self.nHitsMcpU = pfo.nHitsMcpU
        self.nHitsMatchU = pfo.nHitsMatchU
        self.vertexU = np.array([pfo.get("vertex[0]"), 0.5 * pfo.get("vertex[2]") - 0.8660254 * pfo.get("vertex[1]")], dtype = np.double)
        self.interactionVertexU = None

        # V view
        self.driftCoordV = np.array(pfo.driftCoordV, dtype = np.double)
        self.driftCoordErrV = np.array(pfo.driftCoordErrorV, dtype = np.double)
        self.wireCoordV = np.array(pfo.wireCoordV, dtype = np.double)
        self.energyV = np.array(pfo.energyV, dtype = np.double)
        self.nHitsPfoV = pfo.nHitsPfoV
        self.nHitsMcpV = pfo.nHitsMcpV
        self.nHitsMatchV = pfo.nHitsMatchV
        self.vertexV = np.array([pfo.get("vertex[0]"), 0.5 * pfo.get("vertex[2]") + 0.8660254 * pfo.get("vertex[1]")], dtype = np.double)
        self.interactionVertexV = None

        # W view
        self.driftCoordW = np.array(pfo.driftCoordW, dtype = np.double)
        self.driftCoordErrW = np.array(pfo.driftCoordErrorW, dtype = np.double)
        self.wireCoordW = np.array(pfo.wireCoordW, dtype = np.double)
        self.energyW = np.array(pfo.energyW, dtype = np.double)
        self.nHitsPfoW = pfo.nHitsPfoW
        self.nHitsMcpW = pfo.nHitsMcpW
        self.nHitsMatchW = pfo.nHitsMatchW
        self.vertexW = np.array([pfo.get("vertex[0]"), pfo.get("vertex[2]")], dtype = np.double)
        self.interactionVertexW = None

        # 3D view
        self.xCoord3D = pfo.xCoordThreeD
        self.yCoord3D = pfo.yCoordThreeD
        self.zCoord3D = pfo.zCoordThreeD
        self.energy3D = pfo.energyThreeD
        self.nHitsPfo3D = len(self.xCoord3D)
        self.vertex3D = np.array([pfo.get("vertex[0]"), pfo.get("vertex[1]"), pfo.get("vertex[2]")], dtype = np.double)
        self.interactionVertex3D = None

    # These change how the PFO is printed to the screen
    def __str__(self):
        return "{PFO eventID=%d pfoID=%d}" % (self.eventId, self.pfoId)
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return str(self)

    def nHitsPfo(self):
        return self.nHitsPfoU + self.nHitsPfoV + self.nHitsPfoW
    def IsShower(self):
        if self.mcPdgCode != 0:
            return 1 if abs(self.mcPdgCode) in (11, 22) else 0
        else:
            return -1
    def IsTrack(self):
        if self.mcPdgCode != 0:
            return 0 if abs(self.mcPdgCode) in (11, 22) else 1
        else:
            return -1
    def TrueParticle(self):
        if self.mcPdgCode in self.particleTypes:
            return  self.particleTypes[self.mcPdgCode]
        else:
            return str(self.mcPdgCode)

    def PurityOverall(self):
        return self.Purity(self.nHitsMatchU + self.nHitsMatchV + self.nHitsMatchW, self.nHitsPfoU + self.nHitsPfoV + self.nHitsPfoW)
    def PurityU(self):
        return self.Purity(self.nHitsMatchU, self.nHitsPfoU)
    def PurityV(self):
        return self.Purity(self.nHitsMatchV, self.nHitsPfoV)
    def PurityW(self):
        return self.Purity(self.nHitsMatchW, self.nHitsPfoW)
    def Purity(self, nHitsMatch, nHitsPfo):
        return nHitsMatch / nHitsPfo if nHitsPfo > 0 else -1

    def CompletenessOverall(self):
        return self.Purity(self.nHitsMatchU + self.nHitsMatchV + self.nHitsMatchW, self.nHitsMcpU + self.nHitsMcpV + self.nHitsMcpW)
    def CompletenessU(self):
        return self.Purity(self.nHitsMatchU, self.nHitsMcpU)
    def CompletenessV(self):
        return self.Purity(self.nHitsMatchV, self.nHitsMcpV)
    def CompletenessW(self):
        return self.Purity(self.nHitsMatchW, self.nHitsMcpW)
    def Completeness(self, nHitsMatch, nHitsMcp):
        return nHitsMatch / nHitsMcp if nHitsMcp > 0 else -1

    def ValidVertex(self):
        return (
            self.vertex3D[0] > MicroBooneGeo.RangeX[0] and
            self.vertex3D[0] < MicroBooneGeo.RangeX[1] and
            self.vertex3D[1] > MicroBooneGeo.RangeY[0] and
            self.vertex3D[1] < MicroBooneGeo.RangeY[1] and
            self.vertex3D[2] > MicroBooneGeo.RangeZ[0] and
            self.vertex3D[2] < MicroBooneGeo.RangeZ[1]
        )

# Microboone Geometry stuff
class MicroBooneGeo:
    RangeX = (0, 256.35)
    RangeY = (-116.5, 116.5)
    RangeZ = (0, 1036.8)
    WireRangeU = (-96.7, 619.1)
    WireRangeV = (-96.7, 619.1)
    DeadZonesW = ((5.8, 6.1),
                  (24.7, 25),
                  (25.3, 25.6),
                  (52.9, 57.7),
                  (91.3, 96.1),
                  (100.9, 105.7),
                  (120.1, 124.9),
                  (226.3, 226.6),
                  (226.9, 227.2),
                  (244.9, 249.7),
                  (288.1, 292.9),
                  (345.7, 350.5),
                  (398.5, 403.3),
                  (412.9, 417.7),
                  (477.7, 478),
                  (669.4, 669.7),
                  (700.9, 720.1),
                  (720.4, 724.6),
                  (724.9, 739.3),
                  (787.6, 787.9),
                  (806.5, 811.3),
                  (820.9, 825.7),
                  (873.7, 878.5))

# This function inserts a pfo into an event. It also ensures that the list is ordered by the pfo ID.
def AddPfoToEvent(eventPfos, pfo):
    for i in range(0, len(eventPfos)):
        if pfo.pfoId < eventPfos[i].pfoId:
            eventPfos.insert(i, pfo)
            return
    eventPfos.append(pfo)

# Transcribes data from a ROOT file to Python
def ReadRootFile(filepath):
    file = up.open(filepath)
    tree = file["PFOs"].pandas.df(flatten=False)
    events = []        # Array containing arrays of Pfos from the same event.
    currentEventId = -1    # Allows function writing to the events and eventPfos arrays to work (see below).

    for index, pfo in tree.iterrows():
        PfoBeingRead = PfoClass(pfo, os.path.basename(filepath))  # Inputing the variables read from the ROOT file into the class to create the PfoObject.
        if currentEventId == pfo.eventId:
            AddPfoToEvent(eventPfos, PfoBeingRead)
        else:
            if currentEventId != -1:
                SetAssociatedData(eventPfos)
                events.append(eventPfos)
            eventPfos = [PfoBeingRead]
            currentEventId = pfo.eventId

    # The for loop does not append the last event to the array
    SetAssociatedData(eventPfos)
    events.append(eventPfos)
    return events

# Set any data that is based on hierarchy associations
def SetAssociatedData(eventPfos):
    neutrinoEnergy = eventPfos[0].mcpMomentum
    for pfo in eventPfos[1:]:
        parentPfo = eventPfos[pfo.parentPfoId]
        if pfo.heirarchyTier == 1:
            pfo.interactionVertexU = parentPfo.vertexU
            pfo.interactionVertexV = parentPfo.vertexV
            pfo.interactionVertexW = parentPfo.vertexW
            pfo.interactionVertex3D = parentPfo.vertex3D
        elif pfo.heirarchyTier > 1:
            pfo.interactionVertexU = pvf.InteractionVertex2D(pfo.vertexU, pfo.driftCoordU, pfo.wireCoordU, parentPfo.vertexU, parentPfo.driftCoordU, parentPfo.wireCoordU, pfo.nHitsPfoU, parentPfo.nHitsPfoU)
            pfo.interactionVertexV = pvf.InteractionVertex2D(pfo.vertexV, pfo.driftCoordV, pfo.wireCoordV, parentPfo.vertexV, parentPfo.driftCoordV, parentPfo.wireCoordV, pfo.nHitsPfoV, parentPfo.nHitsPfoV)
            pfo.interactionVertexW = pvf.InteractionVertex2D(pfo.vertexW, pfo.driftCoordW, pfo.wireCoordW, parentPfo.vertexW, parentPfo.driftCoordW, parentPfo.wireCoordW, pfo.nHitsPfoW, parentPfo.nHitsPfoW)
            pfo.interactionVertex3D = pvf.InteractionVertex3D(pfo.vertex3D, pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, parentPfo.vertex3D, parentPfo.xCoord3D, parentPfo.yCoord3D, parentPfo.zCoord3D, pfo.nHitsPfo3D, parentPfo.nHitsPfo3D)
        
        pfo.mcNeutrinoEnergy = neutrinoEnergy

def ReadPfoFromRootFile(filepath, eventId, pfoId):
    file = up.open(filepath)
    tree = file["PFOs"].pandas.df(flatten=False)
    dfFilter = (tree['eventId'] == eventId) & (tree['pfoId'] == pfoId)
    tree = tree[dfFilter]
    if len(tree) == 1:
        return PfoClass(tree.iloc[0], os.path.basename(filepath))
    else:
        return None
