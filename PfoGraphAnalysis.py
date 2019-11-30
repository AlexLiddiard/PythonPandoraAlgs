import os
import glob
import UpRootFileReader as rdr
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random as rnd
import pandas as pd

myTestArea = "/home/alexliddiard/Desktop/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'

usePickleFile = True
filters = ({'name': 'likelihood', 'min': 0, 'max': 0.87},
           {'name': 'absPdgCode', 'min': 11, 'max': 11})
additionalInfo = ['F0a', 'F1a', 'F2a', 'F2b', 'F2c', 'F2d', 'F2e', 'likelihood']

# Microboone Geometry stuff
class MicroBooneGeo:
    SpanX = 256.35
    SpanW = 1036.8
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

# From given axes limits, calculate new limits that give a square region.
# The square region encloses the old (rectangular) region and shares the same centre point.
def GetSquareRegionAxesLimits(xMin, xMax, yMin, yMax):
    xSpan = (xMax - xMin) / 2
    ySpan = (yMax - yMin) / 2
    maxSpan = max(xSpan, ySpan)
    xMid = xMin + xSpan
    yMid = yMin + ySpan
    return xMid - maxSpan, xMid + maxSpan, yMid - maxSpan, yMid + maxSpan

def DisplayPfo(pfo, additionalInfo = None):
    # Setting variables to be plotted.
    x = pfo.driftCoordW
    y = pfo.wireCoordW
    xerr = pfo.driftCoordErrW / 2
    yerr = np.repeat(pfo.wireCoordErr / 2, pfo.nHitsPfoW)

    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')
    colorList = [(1, 0, 0), (0, 0, 1)]
    energyMap = matplotlib.colors.LinearSegmentedColormap.from_list('energyMap', colorList, N=1024)

    # Plot variables
    sc = ax.scatter(x, y, s=20, c=pfo.energyW, cmap=energyMap, zorder=3)
    clb = plt.colorbar(sc)
    clb.set_label('Energy as Ionisation Charge', fontsize = 15)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', mew=0, zorder=0, c='black')
    ax.plot(pfo.vertex[0], pfo.vertex[2], marker = 'X', color = 'green', markersize = 15)

    # Plot detector region and dead zones
    ax.autoscale(False)
    for zone in MicroBooneGeo.DeadZonesW:
        ax.add_patch(plt.Rectangle((0, zone[0]), MicroBooneGeo.SpanX, zone[1] - zone[0], alpha=0.15))
    ax.add_patch(plt.Rectangle((0, 0), MicroBooneGeo.SpanX, MicroBooneGeo.SpanW, fill=False, linewidth=2))

    # Set axes limits so that the region is a square
    xMin, xMax, yMin, yMax = GetSquareRegionAxesLimits(*ax.get_xlim(), *ax.get_ylim())
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)

    # Axes and labels
    if additionalInfo is not None:
        additionalInfoStr = '\n'.join([('%s = %.2f' % info).rstrip('0').rstrip('.') for info in additionalInfo.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.75, 0.97, additionalInfoStr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)


    plt.title('%s\nEventId = %d, PfoId = %d, Hierarchy = %d\n%s (%s), Purity = %.2f, Completeness = %.2f' %
              (pfo.fileName,
               pfo.eventId, pfo.pfoId, pfo.heirarchyTier,
               pfo.TrueParticleW(), 'Track' if pfo.IsShowerW()==0 else 'Shower', pfo.PurityW(), pfo.CompletenessW()),
               fontsize=20)
    plt.xlabel('DriftCoordW (cm)', fontsize = 15)
    plt.ylabel('WireCoordW (cm)', fontsize = 15)

    plt.show()

def RandomPfoView(filePaths):
    rnd.shuffle(filePaths)
    for filePath in filePaths:
        events = rdr.ReadRootFile(filePath)
        rnd.shuffle(events)
        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.monteCarloPDGW == 0:
                    continue;
                DisplayPfo(pfo)

def SelectivePfoView(filePaths, dfPfoFeatureData, filters):
    for filterer in filters:
        dfFilter = (dfPfoFeatureData[filterer['name']] >= filterer['min']) & (dfPfoFeatureData[filterer['name']] <= filterer['max'])
        dfPfoFeatureData = dfPfoFeatureData[dfFilter]

    nameToPathDict = FileNameToFilePath(filePaths)
    dfFilter = dfPfoFeatureData['fileName'].isin(nameToPathDict)
    dfPfoFeatureData = dfPfoFeatureData[dfFilter]

    for index, pfoData in dfPfoFeatureData.iterrows():
        filePath = nameToPathDict[pfoData.fileName]
        pfo = rdr.ReadPfoFromRootFile(filePath, pfoData.eventId, pfoData.pfoId)
        DisplayPfo(pfo, pfoData[additionalInfo])


def FileNameToFilePath(filePaths):
    nameToPathDict = {}
    for filePath in filePaths:
        nameToPathDict[os.path.basename(filePath)] = filePath
    return nameToPathDict


if __name__ == "__main__":
    filePaths =  glob.glob(rootFileDirectory + '/**/*.root', recursive=True)
    if usePickleFile:
        dfPfoFeatureData = pd.read_pickle(inputPickleFile)
        SelectivePfoView(filePaths, dfPfoFeatureData, filters)
    else:
        RandomPfoView(filePaths)
    print("Done")


