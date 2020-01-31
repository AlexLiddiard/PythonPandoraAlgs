############################################## GENERAL CONFIGURATION ##################################################

classNames = ("photon", "electron")
classQueries = ("abs(mcPdgCode)==22", "abs(mcPdgCode==11)")
random_state = 201746973


############################################## CONFIGURATION PROCESSING ##################################################
classes = {}
for (className, classQuery)  in zip(classNames, classQueries):
    classes[className] = classQuery
classes["all"] = " or ".join(classes.values())