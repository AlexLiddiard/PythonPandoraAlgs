import BaseConfig as bc

############################################## GET FEATURE DATA CONFIGURATION ##################################################

# "dataName": "ROOT file directory"
dataSources = {
    "BNBNuOnly": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/BNBNuOnly",
    "BNBNuOnly_0-400": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/BNBNuOnly_0-400",
    "BNBNuOnly_400-800": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/BNBNuOnly_400-800",
}

algorithmNames = [
    #"GeneralInfo",
    #"LinearRegression",
    "HitBinning",
    #"ChainCreation",
    #"AngularSpan",
    #"PCAnalysis",
    #"ChargeStdMeanRatio",
    #"BraggPeak",
    #"MoliereRadius"
]

calculateView = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True,
}
