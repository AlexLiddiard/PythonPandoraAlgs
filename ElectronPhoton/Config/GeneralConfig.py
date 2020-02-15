############################################## GENERAL CONFIGURATION ##################################################

classNames = ("Electron", "Photon")
classQueries = (
    "abs(mcPdgCode) == 11", 
    "abs(mcPdgCode) == 22"
)
random_state = 201746973
additionalImportPaths = ["TrackShower/FeatureAlgs"]

############################################## CONFIGURATION PROCESSING ##################################################
classes = {}
for (className, classQuery)  in zip(classNames, classQueries):
    classes[className] = classQuery
classes["all"] = " or ".join(classes.values())