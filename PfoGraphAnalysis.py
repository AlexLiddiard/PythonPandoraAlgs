import os
import UpRootFileReader as rdr
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import random as rnd

minHits = 2

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

if __name__ == "__main__":
    directory = input("Enter a folder path containing ROOT files: ")
    #directory = "/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/"
    fileList = os.listdir(directory)
    rnd.shuffle(fileList)
    for fileName in fileList:
        events = rdr.ReadRootFile(os.path.join(directory, fileName))
        rnd.shuffle(events)
        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.nHitsW < minHits or pfo.TrueTypeW() == -1:
                    continue


                # Setting variables to be plotted.
                x = pfo.driftCoordW
                y = pfo.wireCoordW
                xerr = pfo.driftCoordErrW / 2
                yerr = np.repeat(pfo.wireCoordErr / 2, pfo.nHitsW)

                fig = plt.figure(figsize=(13,10))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_aspect(aspect = 1)
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
                ax.add_patch(plt.Rectangle((0, 0), MicroBooneGeo.SpanX, MicroBooneGeo.SpanW, fill=False))

                # Axes and labels
                plt.title('%s\nEventId = %d, PfoId = %d, Hierarchy = %d, %s (%s)' %(fileName ,pfo.eventId, pfo.pfoId, pfo.heirarchyTier, pfo.TrueParticleW(), 'Track' if pfo.TrueTypeW()==0 else 'Shower'), fontsize=20)
                plt.xlabel('DriftCoordW (cm)', fontsize = 15)
                plt.ylabel('WireCoordW (cm)', fontsize = 15)

                plt.show()

