import PythonPandoraAlgs.DataSampler as ds
import numpy as np

############################################## GENERAL CONFIGURATION ##################################################

classNames = ("photon", "electron")
classQueries = ("isElectron==0", "isElectron==1")
random_state = 201746973

classes = {}
for (className, classQuery)  in zip(classNames, classQueries):
    classes[className] = classQuery