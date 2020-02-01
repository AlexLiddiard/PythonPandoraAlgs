import BaseConfig as bc

############################################## GET FEATURE DATA CONFIGURATION ##################################################

rootFileDirectory = bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/BNBNuOnly"
outputDataName = "BNBNuOnly2000"

algorithmNames = [
    #"GeneralInfo",
    #"InitialdEdx",
    #"AngularSpan",
    "SegmentedPca",
]

calculateViews = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True
}
