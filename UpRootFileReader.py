# This module is used to transcribe PFO data from a ROOT file (stored as vectors) into python arrays.
import uproot as up
import pandas as pd
import numpy as np

# This class defines the data associated with each PFO.
class PfoClass(object):

    wireCoordErr = 0.3  # Sets wire coord error to 3 millimetres.

    def __init__(self, pfo):
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
        self.nHitsU = len(pfo.DriftCoordU)
        self.nHitsV = len(pfo.DriftCoordV)
        self.nHitsW = len(pfo.DriftCoordW)

    # These change how the PFO is printed to the screen
    def __str__(self):
        return "{PFO eventID=%d pfoID=%d}" % (self.eventId, self.pfoId)

    def __unicode__(self):
        return str(self)

    def __repr__(self):
        return str(self)

    def TrueTypeU(self):
        if self.monteCarloPDGU != 0:
            return 1 if abs(self.monteCarloPDGU) in (11, 22) else 0
        else:
            return -1

    def TrueTypeV(self):
        if self.monteCarloPDGV != 0:
            return 1 if abs(self.monteCarloPDGV) in (11, 22) else 0
        else:
            return -1

    def TrueTypeW(self):
        if self.monteCarloPDGW != 0:
            return 1 if abs(self.monteCarloPDGW) in (11, 22) else 0
        else:
            return -1

    # Uses the PDG codes of all wire plane to check if the PFO truly is a track or shower
    def TrueTypeCombined(self):
        absPDGs = [self.TrueTypeU(), self.TrueTypeV(), self.TrueTypeW()]
        # Is a shower if more than two views have majority of hits from electrons.
        if absPDGs.count(1) > 1:
            return 1
        # Is a track if more than two views have majority of hits from other particle types.
        elif absPDGs.count(0) == 0:
            return 0
        # Two conflicting PDGs.
        elif absPDGs.count(-1) == 1:
            return -1
        # Only a PDG from one view. We just use that PDG.
        elif absPDGs.count(-1) == 2:
            return absPDGs.count(1)
        # No PDG info available
        else:
            return -1


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
        PfoBeingRead = PfoClass(pfo)  # Inputing the variables read from the ROOT file into the class to create the PfoObject.
        if currentEventId == pfo.EventId:
            AddPfoToEvent(eventPfos, PfoBeingRead)
        else:
            events.append(eventPfos)
            eventPfos = [PfoBeingRead]
            currentEventId = pfo.EventId

    # The for loop does not append the last event to the array
    events.append(eventPfos)
    return events
