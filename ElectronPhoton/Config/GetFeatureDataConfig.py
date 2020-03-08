import BaseConfig as bc

############################################## GET FEATURE DATA CONFIGURATION ##################################################

# "dataName": "ROOT file directory"
dataSources = {
    "eminus_0-4000": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/eminus_0-4000",
    "Pi0_0-1000": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/Pi0_0-1000",
    "NCDelta": bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/NCDelta",
}

algorithmNames = [
    "GeneralInfo",
    "TotalCharge",
    #"AngularSpan",
    "TestDeDx1",
    "TestDeDx2",
    #"VertexSeparation",
    "NewInitialDeDx",
    "AnglesFromAxes",
]

calculateView = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True,
}
