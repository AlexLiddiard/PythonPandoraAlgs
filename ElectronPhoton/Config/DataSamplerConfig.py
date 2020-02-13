import BaseConfig as bc

############################################## DATA SAMPLING CONFIGURATION ##################################################

# "DataSourceName": (data start fraction, data end fraction)
#  where e.g. data start fraction = (data start position) / (# of samples)
dataSources = {
    "training": {
        #"BNBNuOnly2000": (0, 0.5),
        #"BNBNuOnly": (0, 0.5),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 0),
        "Pi0_0-1000": (0, 0.5),
        "NCDelta": (0, 0.5),
        "eminus_0-4000": (0, 0.5),
    },
    "performance": {
        #"BNBNuOnly2000": (0.5, 1),
        #"BNBNuOnly": (0, 1),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 1),
        "Pi0_0-1000": (0.5, 1),
        "NCDelta": (0.5, 1),
        "eminus_0-4000": (0.5, 1),
    }
}

preFilters = {
    "training": {
        "general": (
            'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
            'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
            'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
            'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
            'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
            'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
        ),
        "U": (
            'purityU>=0.8',
            'completenessU>=0.5',
            #'nHitsU>=50',
        ),
        "V": (
            'purityV>=0.8',
            'completenessV>=0.5',
            #'nHitsV>=50',
        ),
        "W": (
            'purityW>=0.8',
            'completenessW>=0.5',
            #'nHitsW>=50',
        ),
        "3D":
        (
            'purityU>=0.8',
            'completenessU>=0.5',
            'purityW>=0.8',
            'completenessW>=0.5',
            'purityW>=0.8',
            'completenessW>=0.5',
            #'nHits3D >= 50',
        )
    },

    "performance": {
        "general": (
            'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
            'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
            'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
            'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
            'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
            'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
            'nHitsU>=20 and nHitsV >= 20 and nHitsW>=20 and nHits3D>=20',
            #'mcHierarchyTier == 1 or (abs(mcPdgCode) == 22 and mcHierarchyTier <= 2)',
            #'mcNuanceCode == 1000', # ????????????????????????
            #'mcNuanceCode == 1001', #CCQE
            #'mcNuanceCode == 1002', #NCQE
            #'mcNuanceCode in (1003, 1004, 1005)', #CCRES
            #'mcNuanceCode in (1006, 1007, 1008, 1009)', #NCRES
            #'mcNuanceCode == 1091', #CCDIS
            #'mcNuanceCode == 1092', #NCDIS
            #'mcNuanceCode == 1096', #NCCOH
            #'mcNuanceCode == 1097', #CCCOH
            #'mcNuanceCode == 1098', #e-Nu_e scatter
            #'mcNuanceCode not in (1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1091, 1092, 1096, 1097)', #Other
            #"nHitsU + nHitsV + nHitsW >= 100",
            'nHitsU>=0',
        ),
        "U": (
            #'purityU>=0.8',
            #'completenessU>=0.8',
            'nHitsU>=0',
        ),
        "V": (
            #'purityV>=0.8',
            #'completenessV>=0.8',
            'nHitsV>=0',
        ),
        "W": (
            #'purityW>=0.8',
            #'completenessW>=0.8',
            'nHitsW>=0',
        ),
        "3D":
        (
            #'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
            #'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
            'nHits3D>=0',
        )
    }
}

############################################## CONFIGURATION PROCESSING ##################################################
def ProcessFilters(filterClasses):
    for filterClass in filterClasses:
        for filter in filterClasses[filterClass]:
            filterClasses[filterClass][filter] = ' and '.join(filterClasses[filterClass][filter])
        viewFilters = [filterClasses[filterClass][x] for x in ["U", "V", "W", "3D"]]
        filterClasses[filterClass]["union"] = "(" + ") or (".join(viewFilters) + ")"
        filterClasses[filterClass]["intersection"] = "(" + ") and (".join(viewFilters) + ")"

dataSources["all"] = []
for dataSource in dataSources.values():
    dataSources["all"] += list(dataSource)
dataSources["all"] = dict.fromkeys(dataSources["all"])
ProcessFilters(preFilters)