import os
import UpRootFileReader as rdr
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors
from PythonTools.mpl_interaction import figure_pz

minHits = 10

if __name__ == "__main__":
    #directory = input("Enter a folder path containing ROOT files: ")
    directory = "/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/"
    fileList = os.listdir(directory)

    print("EventId\tPfoId\tType\t[Features]")
    for fileName in fileList:
        events = rdr.ReadRootFile(os.path.join(directory, fileName))
        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.nHitsW == 0:
                    continue
                
                x = pfo.driftCoordW
                y = pfo.wireCoordW
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect(aspect = 1)
                fig.set_dpi(200)
                colorList = [(1, 0, 0), (0, 0, 1)]
                energyMap = matplotlib.colors.LinearSegmentedColormap.from_list('energyMap', colorList, N=1024)
                
                ax.scatter(x, y, c=pfo.energyW, s = 100, cmap = energyMap)
                
                
                plt.show()
                input()