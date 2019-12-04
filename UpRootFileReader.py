# This module is used to transcribe PFO data from a ROOT file (stored as vectors) into python arrays.
import uproot as up
import numpy as np
import os

# This class defines the data associated with each PFO.
class PfoClass(object):

    wireCoordErr = 0.3  # Sets wire coord error to 3 millimetres.

    def __init__(self, pfo, fileName):
        self.eventId = pfo.EventId
        self.pfoId = pfo.PfoId
        self.parentPfoId = pfo.ParentPfoId
        self.daughterPfoIds = np.array(pfo.DaughterPfoIds)
        self.heirarchyTier = pfo.HierarchyTier
        self.monteCarloPDGU = pfo.MCPdgCodeU
        self.monteCarloPDGV = pfo.MCPdgCodeV
        self.monteCarloPDGW = pfo.MCPdgCodeW
        self.vertex = np.array([pfo.get("Vertex[0]"), pfo.get("Vertex[1]"), pfo.get("Vertex[2]")], dtype = np.double)
        self.driftCoordW = np.array(pfo.DriftCoordW, dtype = np.double)
        self.driftCoordErrW = np.array(pfo.DriftCoordErrorW, dtype = np.double)
        self.wireCoordW = np.array(pfo.WireCoordW, dtype = np.double)
        self.driftCoordU = np.array(pfo.DriftCoordU, dtype = np.double)
        self.driftCoordErrU = np.array(pfo.DriftCoordErrorU, dtype = np.double)
        self.wireCoordU = np.array(pfo.WireCoordU, dtype = np.double)
        self.driftCoordV = np.array(pfo.DriftCoordV, dtype = np.double)
        self.driftCoordErrV = np.array(pfo.DriftCoordErrorV, dtype = np.double)
        self.wireCoordV = np.array(pfo.WireCoordV, dtype = np.double)
        self.energyU = np.array(pfo.EnergyU, dtype = np.double)
        self.energyV = np.array(pfo.EnergyV, dtype = np.double)
        self.energyW = np.array(pfo.EnergyW, dtype = np.double)
        self.nHitsPfoU = pfo.nHitsPfoU
        self.nHitsPfoV = pfo.nHitsPfoV
        self.nHitsPfoW = pfo.nHitsPfoW
        self.nHitsMatchU = pfo.nHitsMatchU
        self.nHitsMatchV = pfo.nHitsMatchV
        self.nHitsMatchW = pfo.nHitsMatchW
        self.nHitsMcpU = pfo.nHitsMcpU
        self.nHitsMcpV = pfo.nHitsMcpV
        self.nHitsMcpW = pfo.nHitsMcpW
        self.fileName = fileName

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
        return self.IsShower(self.monteCarloPDGW)
    def IsShowerV(self):
        return self.IsShower(self.monteCarloPDGV)
    def IsShowerU(self):
        return self.IsShower(self.monteCarloPDGU)
    def IsShower(self, pdgCode):
        if pdgCode != 0:
            return 1 if abs(pdgCode) in (11, 22) else 0
        else:
            return -1

    def IsTrackW(self):
        return self.IsTrack(self.monteCarloPDGW)
    def IsTrackV(self):
        return self.IsTrack(self.monteCarloPDGV)
    def IsTrackU(self):
        return self.IsTrack(self.monteCarloPDGU)
    def IsTrack(self, pdgCode):
        if pdgCode != 0:
            return 1 if abs(pdgCode) not in (11, 22) else 0
        else:
            return -1

    def TrueParticleW(self):
        return self.TrueParticle(self.monteCarloPDGW)
    def TrueParticleV(self):
        return self.TrueParticle(self.monteCarloPDGV)
    def TrueParticleU(self):
        return self.TrueParticle(self.monteCarloPDGU)
    def TrueParticle(self, pdgCode):
            switcher = {
                    11: "Electron",
                    12: "Electron Neutrino",
                    13: "Muon",
                    14: "Muon Neutrino",
                    15: "Tau",
                    16: "Tau Neutrino",
                    22: "Photon",
                    111: "Neutral Pion",
                    211: "Pion",
                    311: "Neutral Kaon",
                    321: "Kaon",
                    411: "D",
                    421: "Neutral D",
                    511: "Neutral B",
                    521: "B",
                    2212: "Proton",
                    2112: "Neutron",
                    2224: "Doubly Delta",
                    2214: "Delta",
                    2114: "Neutral Delta",
                    1114: "Negative Delta",
                    9221132: "Theta",
                    9331122: "Phi"
                    }
            return  ("Anti-" if pdgCode < 0 else "") + switcher.get(abs(pdgCode), "Unknown")

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
