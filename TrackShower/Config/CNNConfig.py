import BaseConfig as bc
import numpy as np
import DataSamplerConfig as dsc

rootFileDirectory =  bc.pythonFolderFull + "/ROOT Files/OneFile"
outputFolder = bc.analysisFolderFull + "/SVGData"

# {"View": (driftSpan, wireSpan)}
imageSpan = {
    "U": (100, 100), 
    "V": (100, 100),
    "W": (100, 100),
}

maxEnergyDensity = 1500 / (0.3 * 0.3)

preRequisites = {
    "training":[
        'abs(mcPdgCode) != 2112',
        'minCoordX3D <= @MicroBooneGeo.RangeX[1] - 15',
        'minCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
        'minCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 20',
        'nHitsU>=20',
        'nHitsV>=20',
        'nHitsW>=20',
        'purityU>=0.8',
        'completenessU>=0.5',
    ],
    "performance":[
        'minCoordX3D <= @MicroBooneGeo.RangeX[1] - 15',
        'minCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
        'minCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 20',
        'nHitsU>=20',
        'nHitsV>=20',
        'nHitsW>=20',
    ]
}


preRequisites["training"] = dsc.CombineFilters(preRequisites["training"], "and")
preRequisites["performance"] = dsc.CombineFilters(preRequisites["performance"], "and")

trainingRatio = 0.5