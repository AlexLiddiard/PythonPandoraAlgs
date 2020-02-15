############################################## GENERAL CONFIGURATION ##################################################

classNames = ("track", "shower")
classQueries = ("isShower==0", "isShower==1")
random_state = 201746973
additionalImportPaths = []

############################################## CONFIGURATION PROCESSING ##################################################
classes = {}
for (className, classQuery)  in zip(classNames, classQueries):
    classes[className] = classQuery
classes["all"] = " or ".join(classes.values())