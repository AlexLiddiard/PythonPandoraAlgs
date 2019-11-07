import os
import UpRootFileReader as rdr
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

minHits = 10

if __name__ == "__main__":
    #directory = input("Enter a folder path containing ROOT files: ")
    directory = "/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/"
    fileList = os.listdir(directory)

    for fileName in fileList:
        events = rdr.ReadRootFile(os.path.join(directory, fileName))
        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.nHitsW == 0:
                    continue
                
                
                #Setting variables to be plotted.
                x = pfo.driftCoordW
                y = pfo.wireCoordW
                xerr = pfo.driftCoordErrW
                yerr = np.repeat(pfo.wireCoordErr, pfo.nHitsW)
                 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect(aspect = 1)
                fig.set_dpi(200)
                colorList = [(1, 0, 0), (0, 0, 1)]
                energyMap = matplotlib.colors.LinearSegmentedColormap.from_list('energyMap', colorList, N=1024)

                sc = ax.scatter(x, y, s=20, c=pfo.energyW, cmap=energyMap, zorder=3)
                
                clb = plt.colorbar(sc)
                clb.set_label('Energy as Ionisation Charge', fontsize = 15)
                
                ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', mew=0, zorder=0, c='black')        
                
                plt.title('EventId = %d, PfoId = %d, Hierarchy = %d, %s' %(pfo.eventId, pfo.pfoId, pfo.heirarchyTier, 'Track' if pfo.TrueTypeW()==0 else 'Shower'), fontsize=20)
                plt.xlabel('DriftCoordW (cm)', fontsize = 15)
                plt.ylabel('WireCoordW (cm)', fontsize = 15)
                ax.plot(pfo.vertex[0], pfo.vertex[2], marker = 'X', color = 'green', markersize = 15)
                
                
                plt.show()
