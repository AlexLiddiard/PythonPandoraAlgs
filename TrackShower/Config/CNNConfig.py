import BaseConfig as bc
import numpy as np

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