import BaseConfig as bc
import numpy as np
import DataSamplerConfig as dsc

rootFileDirectory =  bc.pythonFolderFull + "/ROOT Files"
outputFolder = bc.analysisFolderFull + "/SVGData"

# {"View": (driftSpan, wireSpan)}
imageSizePixels = (224, 224)
imageSpan = {
    "U": (100, 100), 
    "V": (100, 100),
    "W": (100, 100),
}

maxEnergyDensity = 1500 / (0.3 * 0.3)
