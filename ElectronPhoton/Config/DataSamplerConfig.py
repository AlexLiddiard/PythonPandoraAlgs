import BaseConfig as bc

############################################## DATA SAMPLING CONFIGURATION ##################################################

# "DataSourceName": (data start fraction, data end fraction)
#  where e.g. data start fraction = (data start position) / (# of samples)
dataSources = {
    "training": {
        "BNBNuOnly2000": (0, 0.5),
        #"BNBNuOnly": (0, 0.5),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 1)
    },
    "performance": {
        "BNBNuOnly2000": (0.5, 1),
        #"BNBNuOnly": (0.5, 1),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 1)
    }
}

preFilters = {
    "training": {
        "general": (
            'abs(mcPdgCode) != 2112',
            #'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
            #'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
            #'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
            #'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
            #'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
            #'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
        ),
        "U": (
            '(abs(mcPdgCode) == 11 and purityU>=0.8) or (abs(mcPdgCode) == 22 and purityU>=0.8)',
            '(abs(mcPdgCode) == 11 and completenessU>=0.5) or (abs(mcPdgCode) == 22 and completenessU>=0.5)',
            #'nHitsU>=50',
        ),
        "V": (
            '(abs(mcPdgCode) == 11 and purityV>=0.8) or (abs(mcPdgCode) == 22 and purityV>=0.8)',
            '(abs(mcPdgCode) == 11 and completenessV>=0.5) or (abs(mcPdgCode) == 22 and completenessV>=0.5)',
            #'nHitsV>=50',
        ),
        "W": (
            '(abs(mcPdgCode) == 11 and purityW>=0.8) or (abs(mcPdgCode) == 22 and purityW>=0.8)',
            '(abs(mcPdgCode) == 11 and completenessW>=0.5) or (abs(mcPdgCode) == 22 and completenessW>=0.5)',
            #'nHitsW>=50',
        ),
        "3D":
        (
            '(abs(mcPdgCode) == 11 and purityU>=0.8) or (abs(mcPdgCode) == 22 and purityU>=0.8)',
            '(abs(mcPdgCode) == 11 and purityV>=0.8) or (abs(mcPdgCode) == 22 and purityV>=0.8)',
            '(abs(mcPdgCode) == 11 and purityW>=0.8) or (abs(mcPdgCode) == 22 and purityW>=0.8)',
            '(abs(mcPdgCode) == 11 and completenessU>=0.5) or (abs(mcPdgCode) == 22 and completenessU>=0.5)',
            '(abs(mcPdgCode) == 11 and completenessV>=0.5) or (abs(mcPdgCode) == 22 and completenessV>=0.5)',
            '(abs(mcPdgCode) == 11 and completenessW>=0.5) or (abs(mcPdgCode) == 22 and completenessW>=0.5)',
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
            'nHitsU>=20 and nHitsV >= 20 and nHitsW>=20 and nHits3D>=20'
            #"nHitsU + nHitsV + nHitsW >= 100"
        ),
        "U": (
            #'purityU>=0.8',
            #'completenessU>=0.8',
            'nHitsU>=20',
        ),
        "V": (
            #'purityV>=0.8',
            #'completenessV>=0.8',
            'nHitsV>=20',
        ),
        "W": (
            #'purityW>=0.8',
            #'completenessW>=0.8',
            'nHitsW>=20',
        ),
        "3D":
        (
            #'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
            #'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
            'nHits3D>=20',
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