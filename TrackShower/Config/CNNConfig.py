import BaseConfig as bc
import numpy as np
import DataSamplerConfig as dsc

rootFileDirectory =  bc.pythonFolderFull + "/ROOT Files/BNBNuOnly"
outputFolder = bc.analysisFolderFull + "/SVGData"

# {"View": (driftSpan, wireSpan)}
imageSpan = {
    "U": (100, 100), 
    "V": (100, 100),
    "W": (100, 100),
}

maxEnergy = 5000
logMaxEnergy = np.log(maxEnergy)

preFilters = {
    "training": {
        "general": (
            'abs(mcPdgCode) != 2112',
            #'maxCoordX3D >= @MicroBooneGeo.RangeX[0] + 10',
            'minCoordX3D <= @MicroBooneGeo.RangeX[1] - 15',
            #'maxCoordY3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
            #'maxCoordZ3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 20',
        ),
    }
}

dsc.ProcessFilters(preFilters)