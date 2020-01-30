import PythonPandoraAlgs.DataSampler as ds
import numpy as np

############################################## GENERAL CONFIGURATION ##################################################

myTestArea = "/home/tomalex/Pandora/"
pythonFolder = "/PythonPandoraAlgs/"
classNames = ("track", "shower")
classQueries = ("isShower==0", "isShower==1")
random_state = 201746973


classes = {}
for (className, classQuery)  in zip(classNames, classQueries):
    classes[className] = classQuery