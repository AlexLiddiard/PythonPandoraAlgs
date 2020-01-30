import PythonPandoraAlgs.DataSampler as ds
import numpy as np

############################################## GENERAL CONFIGURATION ##################################################

myTestArea = "/home/tomalex/Pandora/"
pythonFolder = "/PythonPandoraAlgs/"
classNames = ("track", "shower")
classQueries = {
    "track": "isShower==0",
    "shower": "isShower==1"
}
random_state = 201746973