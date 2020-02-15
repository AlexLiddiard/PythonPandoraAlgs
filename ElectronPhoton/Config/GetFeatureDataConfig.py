import BaseConfig as bc

############################################## GET FEATURE DATA CONFIGURATION ##################################################

rootFileDirectory = bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/eminus_0-4000"
outputDataName = "eminus_0-4000"

algorithmNames = [
    "GeneralInfo",
    "TotalCharge",
    "AngularSpan",
    "TestDeDx1",
    "TestDeDx2",
    "VertexSeparation",
    "NewInitialDeDx",
]

calculateViews = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True
}
