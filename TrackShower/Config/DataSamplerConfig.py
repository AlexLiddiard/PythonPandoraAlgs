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
        #"BNBNuOnly_0-400": (0, 1)
    }
}

preFilters = {
    "training": {
        "general": (
            'abs(mcPdgCode) != 2112',
            #'minCoordX3D >= @MicroBooneGeo.RangeX[0] + 10',
            #'maxCoordX3D <= @MicroBooneGeo.RangeX[1] - 10',
            #'minCoordY3D >= @MicroBooneGeo.RangeY[0] + 20',
            #'maxCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
            #'minCoordZ3D >= @MicroBooneGeo.RangeY[0] + 10',
            #'maxCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 10',
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
            #'minCoordX3D >= @MicroBooneGeo.RangeX[0] + 10',
            #'maxCoordX3D <= @MicroBooneGeo.RangeX[1] - 10',
            #'minCoordY3D >= @MicroBooneGeo.RangeY[0] + 20',
            #'maxCoordY3D <= @MicroBooneGeo.RangeY[1] - 20',
            #'minCoordZ3D >= @MicroBooneGeo.RangeY[0] + 10',
            #'maxCoordZ3D <= @MicroBooneGeo.RangeZ[1] - 10',
            'nHitsU>=20 and nHitsV >= 20 and nHitsW>=20 and nHits3D>=20',
            #"nHitsU + nHitsV + nHitsW >= 100"
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
def CombineFilters(filterList, logicalOperator):
    combinedFilter = ""
    for filter in filterList:
        if filter != "":
            combinedFilter += " %s %s" % (logicalOperator, filter) if combinedFilter != "" else filter
    return combinedFilter

def ProcessFilters(filterClasses):
    for filterClass in filterClasses:
        for filter in filterClasses[filterClass]:
            filterClasses[filterClass][filter] = ' and '.join(filterClasses[filterClass][filter])
        viewFilters = [filterClasses[filterClass][x] for x in ["U", "V", "W", "3D"]]
        filterClasses[filterClass]["union"] = CombineFilters(viewFilters, "or")
        filterClasses[filterClass]["intersection"] = CombineFilters(viewFilters, "and")

dataSources["all"] = []
for dataSource in dataSources.values():
    dataSources["all"] += list(dataSource)
dataSources["all"] = dict.fromkeys(dataSources["all"])
ProcessFilters(preFilters)