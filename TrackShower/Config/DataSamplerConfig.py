import BaseConfig as bc

############################################## DATA SAMPLING CONFIGURATION ##################################################

# "DataSourceName": (data start fraction, data end fraction)
#  where e.g. data start fraction = (data start position) / (# of samples)
dataSources = {
    "training": {
        "BNBNuOnly": (0, 0.5),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 1)
    },
    "performance": {
        "BNBNuOnly": (0.5, 1),
        #"BNBNuOnly400-800": (0, 1),
        #"BNBNuOnly0-400": (0, 1)
    }
}

performanceDataSources = {
    "BNBNuOnly": (0.5, 1),
}

trainingFraction = 0.5
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
            '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
            '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
            #'nHitsU>=50',
        ),
        "V": (
            '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
            '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
            #'nHitsV>=50',
        ),
        "W": (
            '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
            '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
            #'nHitsW>=50',
        ),
        "3D":
        (
            '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
            '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
            '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
            '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
            '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
            '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
            #'nHits3D >= 50',
        )
    },

    "performance": {
        "general": (
            'abs(mcPdgCode) != 2112',
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