import BaseConfig as bc
import os
import glob
import UpRootFileReader as rdr
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random as rnd
import pandas as pd
import DataSampler as ds
import PfoGraphicalAnalyserConfig as cfg
from UpRootFileReader import MicroBooneGeo

# From given axes limits, calculate new limits that give a square region.
# The square region encloses the old (rectangular) region and shares the same centre point.
def GetSquareRegionAxesLimits(xMin, xMax, yMin, yMax):
    xSpan = (xMax - xMin) / 2
    ySpan = (yMax - yMin) / 2
    maxSpan = max(xSpan, ySpan)
    xMid = xMin + xSpan
    yMid = yMin + ySpan
    return xMid - maxSpan, xMid + maxSpan, yMid - maxSpan, yMid + maxSpan

def DisplayPfo(pfo, wireView = "W", additionalInfo = None, showTitle = True):
    # Setting variables to be plotted.
    if wireView == "U":
        x = pfo.driftCoordU
        y = pfo.wireCoordU
        xerr = pfo.driftCoordErrU / 2
        yerr = np.repeat(pfo.wireCoordErr / 2, pfo.nHitsPfoU)
        energy = pfo.energyU
        purity = pfo.PurityU()
        completeness = pfo.CompletenessU()
        vertexDriftCoord = pfo.vertexU[0]
        vertexWireCoord = pfo.vertexU[1]
        wireRange = MicroBooneGeo.WireRangeU
        deadZones = ()
    if wireView == "V":
        x = pfo.driftCoordV
        y = pfo.wireCoordV
        xerr = pfo.driftCoordErrV / 2
        yerr = np.repeat(pfo.wireCoordErr / 2, pfo.nHitsPfoV)
        energy = pfo.energyV
        purity = pfo.PurityV()
        completeness = pfo.CompletenessV()
        vertexDriftCoord = pfo.vertexV[0]
        vertexWireCoord = pfo.vertexV[1]
        wireRange = MicroBooneGeo.WireRangeV
        deadZones = ()
    if wireView == "W":
        x = pfo.driftCoordW
        y = pfo.wireCoordW
        xerr = pfo.driftCoordErrW / 2
        yerr = np.repeat(pfo.wireCoordErr / 2, pfo.nHitsPfoW)
        energy = pfo.energyW
        purity = pfo.PurityW()
        completeness = pfo.CompletenessW()
        vertexDriftCoord = pfo.vertexW[0]
        vertexWireCoord = pfo.vertexW[1]
        wireRange = MicroBooneGeo.RangeZ
        deadZones = MicroBooneGeo.DeadZonesW

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.set_aspect('equal', 'box')
    colourList = [(1, 0, 0), (0, 0, 1)]
    energyMap = matplotlib.colors.LinearSegmentedColormap.from_list('energyMap', colourList, N=1024)

    # Plot variables
    sc = ax.scatter(x, y, s=20, c=energy, cmap=energyMap, zorder=3)
    clb = plt.colorbar(sc)
    clb.set_label('Energy as Ionisation Charge')
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', mew=0, zorder=0, c='black')
    ax.plot(vertexDriftCoord, vertexWireCoord, marker = 'X', color = 'green', markersize = 15)

    # Plot detector region and dead zones
    ax.autoscale(False)
    driftSpan = MicroBooneGeo.RangeX[1] - MicroBooneGeo.RangeX[0]
    wireSpan = wireRange[1] - wireRange[0]
    for zone in deadZones:
        ax.add_patch(plt.Rectangle((MicroBooneGeo.RangeX[0], zone[0]), driftSpan, zone[1] - zone[0], alpha=0.15))
    ax.add_patch(plt.Rectangle((MicroBooneGeo.RangeX[0], wireRange[0]), driftSpan, wireSpan, fill=False, linewidth=2))

    # Set axes limits so that the region is a square
    xMin, xMax, yMin, yMax = GetSquareRegionAxesLimits(*ax.get_xlim(), *ax.get_ylim())
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Axes and labels
    if additionalInfo is not None:
        additionalInfoStr = '\n'.join([('%s = %.2f' % info).rstrip('0').rstrip('.') for info in additionalInfo.items()])
        props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.5}
        ax.text(0.57, 0.98, additionalInfoStr, transform=ax.transAxes, va='top', ha='right', position=(1,1), bbox=props)

    if showTitle:
        plt.title(
            '%s\nEventId = %d, PfoId = %d, Hierarchy = %d\n%s (%s), Purity = %.2f, Completeness = %.2f' %
            (pfo.fileName,
            pfo.eventId, pfo.pfoId, pfo.hierarchyTier,
            pfo.TrueParticle(), 'Track' if pfo.IsShower()==0 else 'Shower', purity, completeness)
        )
    plt.xlabel('DriftCoord%s (cm)' % wireView)
    plt.ylabel('WireCoord%s (cm)' % wireView)
    plt.tight_layout()
    plt.savefig(bc.figureFolderFull + '/%s, EventId %d, PfoId %d.svg' % (pfo.fileName, pfo.eventId, pfo.pfoId), format='svg', dpi=1200)
    plt.show()

def RandomPfoView(filePaths):
    rnd.shuffle(filePaths)
    for filePath in filePaths:
        events = rdr.ReadRootFile(filePath)
        rnd.shuffle(events)
        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.mcPdgCode == 0:
                    continue
                DisplayPfo(pfo, cfg.view)

def SelectivePfoView(filePaths, dfPfoData, pfoFilters):
    nameToPathDict = FileNameToFilePath(filePaths)
    if len(pfoFilters) > 0:
        dfPfoData = dfPfoData.query(' and '.join(pfoFilters))
    nPFOs = len(dfPfoData)
    dfPfoData = dfPfoData.query('fileName in @nameToPathDict')
    nPFOsAvailable = len(dfPfoData)
    print("Found %d matching PFOs, %d have calohit data available." % (nPFOs, nPFOsAvailable))
    dfPfoData = dfPfoData.sample(frac=1).reset_index(drop=True) # Shuffle so that we see different PFOs each time
    for index, pfoData in dfPfoData.iterrows():
        filePath = nameToPathDict[pfoData.fileName]
        pfo = rdr.ReadPfoFromRootFile(filePath, pfoData.eventId, pfoData.pfoId)
        DisplayPfo(pfo, cfg.view, pfoData[cfg.additionalInfo], cfg.showTitle)


def FileNameToFilePath(filePaths):
    nameToPathDict = {}
    for filePath in filePaths:
        nameToPathDict[os.path.basename(filePath)] = filePath
    return nameToPathDict

if __name__ == "__main__":
    plt.rcParams.update(cfg.plotStyle)
    filePaths =  glob.glob(cfg.rootFileDirectory + '/**/*.root', recursive=True)
    if cfg.useDataSample:
        ds.LoadPfoData()
        dfPfoData = ds.GetFilteredPfoData(*cfg.dataSample.values())
        SelectivePfoView(filePaths, dfPfoData, cfg.filters)
    else:
        print(filePaths)
        RandomPfoView(filePaths)
    print('\nFinished!')


