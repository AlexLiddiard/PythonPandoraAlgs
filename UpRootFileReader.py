# This module is used to transcribe PFO data from a ROOT file (stored as vectors) into python arrays.
import uproot as up
import numpy as np
import os

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
        self.eventId = pfo.EventId
        self.pfoId = pfo.PfoId
        self.parentPfoId = pfo.ParentPfoId
        self.daughterPfoIds = np.array(pfo.DaughterPfoIds)
        self.heirarchyTier = pfo.HierarchyTier

        # U view
        self.driftCoordU = np.array(pfo.DriftCoordU, dtype = np.double)
        self.driftCoordErrU = np.array(pfo.DriftCoordErrorU, dtype = np.double)
        self.wireCoordU = np.array(pfo.WireCoordU, dtype = np.double)
        self.energyU = np.array(pfo.EnergyU, dtype = np.double)
        self.mcPdgCodeU = pfo.MCPdgCodeU
        self.nHitsPfoU = pfo.nHitsPfoU
        self.nHitsMcpU = pfo.nHitsMcpU
        self.nHitsMatchU = pfo.nHitsMatchU

        # V view
        self.driftCoordV = np.array(pfo.DriftCoordV, dtype = np.double)
        self.driftCoordErrV = np.array(pfo.DriftCoordErrorV, dtype = np.double)
        self.wireCoordV = np.array(pfo.WireCoordV, dtype = np.double)
        self.energyV = np.array(pfo.EnergyV, dtype = np.double)
        self.mcPdgCodeV = pfo.MCPdgCodeV
        self.nHitsPfoV = pfo.nHitsPfoV
        self.nHitsMcpV = pfo.nHitsMcpV
        self.nHitsMatchV = pfo.nHitsMatchV

        # W view
        self.driftCoordW = np.array(pfo.DriftCoordW, dtype = np.double)
        self.driftCoordErrW = np.array(pfo.DriftCoordErrorW, dtype = np.double)
        self.wireCoordW = np.array(pfo.WireCoordW, dtype = np.double)
        self.energyW = np.array(pfo.EnergyW, dtype = np.double)
        self.mcPdgCodeW = pfo.MCPdgCodeW
        self.nHitsPfoW = pfo.nHitsPfoW
        self.nHitsMcpW = pfo.nHitsMcpW
        self.nHitsMatchW = pfo.nHitsMatchW

        # 3D view
        self.xCoordThreeD = pfo.XCoordThreeD
        self.yCoordThreeD = pfo.YCoordThreeD
        self.zCoordThreeD = pfo.ZCoordThreeD
        self.energyThreeD = pfo.EnergyThreeD
        self.vertex = np.array([pfo.get("Vertex[0]"), pfo.get("Vertex[1]"), pfo.get("Vertex[2]")], dtype = np.double)
        self.nHitsPfoThreeD = len(self.xCoordThreeD)

    # These change how the PFO is printed to the screen
    def __str__(self):
        return "{PFO eventID=%d pfoID=%d}" % (self.eventId, self.pfoId)
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return str(self)

    def nHitsPfo(self):
        return self.nHitsPfoU + self.nHitsPfoV + self.nHitsPfoW
    def IsShowerW(self):
        return self.IsShower(self.mcPdgCodeW)
    def IsShowerV(self):
        return self.IsShower(self.mcPdgCodeV)
    def IsShowerU(self):
        return self.IsShower(self.mcPdgCodeU)
    def IsShower(self, pdgCode):
        if pdgCode != 0:
            return 1 if abs(pdgCode) in (11, 22) else 0
        else:
            return -1

    def IsTrackW(self):
        return self.IsTrack(self.mcPdgCodeW)
    def IsTrackV(self):
        return self.IsTrack(self.mcPdgCodeV)
    def IsTrackU(self):
        return self.IsTrack(self.mcPdgCodeU)
    def IsTrack(self, pdgCode):
        if pdgCode != 0:
            return 1 if abs(pdgCode) not in (11, 22) else 0
        else:
            return -1

    def TrueParticleW(self):
        return self.TrueParticle(self.mcPdgCodeW)
    def TrueParticleV(self):
        return self.TrueParticle(self.mcPdgCodeV)
    def TrueParticleU(self):
        return self.TrueParticle(self.mcPdgCodeU)
    def TrueParticle(self, pdgCode):
            if pdgCode in self.particleTypes:
                return  self.particleTypes[pdgCode]
            else:
                return str(pdgCode)

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
    eventPfos = []        # Array containing PfoObjects
    currentEventId = 0    # Allows function writing to the events and eventPfos arrays to work (see below).

    for index, pfo in tree.iterrows():
        PfoBeingRead = PfoClass(pfo, os.path.basename(filepath))  # Inputing the variables read from the ROOT file into the class to create the PfoObject.
        if currentEventId == pfo.EventId:
            AddPfoToEvent(eventPfos, PfoBeingRead)
        else:
            events.append(eventPfos)
            eventPfos = [PfoBeingRead]
            currentEventId = pfo.EventId

    # The for loop does not append the last event to the array
    events.append(eventPfos)
    return events

def ReadPfoFromRootFile(filepath, eventId, pfoId):
    file = up.open(filepath)
    tree = file["PFOs"].pandas.df(flatten=False)
    dfFilter = (tree['EventId'] == eventId) & (tree['PfoId'] == pfoId)
    tree = tree[dfFilter]
    if len(tree) == 1:
        return PfoClass(tree.iloc[0], os.path.basename(filepath))
    else:
        return None
