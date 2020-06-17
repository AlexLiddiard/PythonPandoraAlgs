# PythonPandoraAlgs
All our Python code do to with Pandora algorithms.

Orientation:

Base config is where the separation you wish to use e.g. track-shower or electron-photon, is specified (these are the analysis folders to be used). It also allows the test area filepath to be defined.

After specifying the analysis folder to be used go to the analysis folder. Inside each analysis folder is a config file. This configures settings to be used by each algorithm in the main /PythonPandoraAlgs area.

In /PythonPandoraAlgs, running GetFeatureData.py uses information in the config file of the analysis folder to find root files and read them. It then outputs relevant data to pickle files in the analysis folder. This includes the calculation of PFO features.

Features are calculated using the /FeatureAlgs in the analysis folder. These are where the algorithms we used to distinguish particles are.

Running FeatureAnalyser presents the results of individual feature algorithms.

Running BDT calculator outputs a pickle file containing specified BDTs from the BDT config in the analysis folder. This used an XGBoost BDT with some sampling and iterative imputation techniques included. Similarly there is a Likelihood calculator.

There is also an MVCNNImageGenerator, which creates the images used by the MVCNN.

Running the MVCNNTrainer trains the MVCNN on some training images. Then MVCNNTester uses the trained MVCNN to identify PFOs.

The DataSampler is included in a lot of these programs to select PFOs for training and testing.


