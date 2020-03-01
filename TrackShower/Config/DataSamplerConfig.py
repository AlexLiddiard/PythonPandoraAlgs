import BaseConfig as bc

############################################## DATA SAMPLER CONFIGURATION ##################################################

# "DataSourceName": (data start fraction, data end fraction)
#  where e.g. data start fraction = (data start position) / (# of samples)
dataSources = {
    "training": {
        "BNBNuOnly": (0, 0.5),
        "BNBNuOnly_400-800": (0, 1),
        #"BNBNuOnly_0-400": (0, 1),
    },
    "performance": {
        "BNBNuOnly": (0.5, 1),
        #"BNBNuOnly_400-800": (0, 1),
        #"BNBNuOnly_0-400": (0, 1),
    }
}

preFilters = {
    "training": {
        "general": [
            'abs(mcPdgCode) != 2112',
            #'maxCoordX3D >= @MicroBooneGeo.RangeX[0] + 10',
            'minCoordX3D <= @MicroBooneGeo.RangeX[1] - 15',
            #'maxCoordY3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
            #'maxCoordZ3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 20',
            'nHitsU>=20',
            'nHitsV>=20',
            'nHitsW>=20',
            'nHits3D>=20',
        ],
        "U": [
            '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
            '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
            #'nHitsU>=50',
        ],
        "V": [
            '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
            '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
            #'nHitsV>=50',
        ],
        "W": [
            '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
            '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
            #'nHitsW>=50',
        ],
        "3D":
        [
            '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
            '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
            '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
            '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
            '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
            '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
            #'nHits3D >= 50',
        ]
    },

    "performance": {
        "general": [
            #'maxCoordX3D >= @MicroBooneGeo.RangeX[0] + 15',
            'minCoordX3D <= @MicroBooneGeo.RangeX[1] - 15',
            #'maxCoordY3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
            #'maxCoordZ3D >= @MicroBooneGeo.RangeY[0] + 20',
            'minCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 20',
            #'nHitsU >= 20 and nHitsV>=20 and nHits3D>=20',
            #'nHitsU>=20 and nHitsV >= 20 and nHitsW>=20 and nHits3D>=20',
            'nHitsU>=20',
            'nHitsV>=20',
            'nHitsW>=20',
            'nHits3D>=20',
            #"nHitsU + nHitsV + nHitsW >= 100",
            #'nHitsU>0 and nHitsV>0 and nHitsW>0',
            #'completenessU > 0.9',
            #'purityU > 0.9',
            #'completenessV > 0.9',
            #'purityV > 0.9',
            #'completenessW > 0.9',
            #'purityW > 0.9',
        ],
        "U": [
            #'purityU>=0.8',
            #'completenessU>=0.8',
            #'nHitsU>=20',
        ],
        "V": [
            #'purityV>=0.8',
            #'completenessV>=0.8',
            #'nHitsV>=20',
        ],
        "W": [
            #'purityW>=0.8',
            #'completenessW>=0.8',
            #'nHitsW>=20',
        ],
        "3D":
        [
            #'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
            #'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
            #'nHits3D>=20',
        ]
    }
}