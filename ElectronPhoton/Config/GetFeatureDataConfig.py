from PythonPandoraAlgs import BaseConfig as bc

rootFileDirectory = bc.myTestArea + "/PythonPandoraAlgs/ROOT Files/BNBNuOnly"
outputDataName = "BNBNuOnly2000"

algorithmNames = (
    "GeneralInfo",
    "LinearRegression",
    "HitBinning",
    "ChainCreation",
    "AngularSpan",
    "PCAnalysis",
    "ChargedHitBinning",
    "ChargeStdMeanRatio",
    "BraggPeak",
    "MoliereRadius"
)

calculateViews = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True
}
