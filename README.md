# PythonPandoraAlgs
All our Python code do to with Pandora algorithms.

## Orientation

#### BaseConfig.py
Base config is where the separation you wish to use e.g. track-shower or electron-photon, is specified (these are the analysis folders to be used). It also allows the test area filepath to be defined.

#### GetFeatureData.py
Runs feature algorithms on PFO data, which is in the "ROOT Files" folder. Algorithms are in the FeatureAlgs folder, inside the analysis folder. These are the algorithms we used to distinguish particles. It outputs relevant data to pickle files in the PickleData folder, inside the analysis folder. 

#### DataSampler.py
Creates the PFO data samples used for the analysis. It is included in a lot of the programs to select PFOs for training and testing.

#### LikelihoodCalculator.py
Trains and validates a likelihood predictor, stores results in the PickleData folder.

#### BDTCalculator.py
Trains and validates a Boosted Decision Tree predictor, in the PickleData folder. This uses an XGBoost BDT with some sampling and iterative imputation techniques included. Similarly there is a Likelihood calculator.

#### MVCNNImageGenerator.py
Uses calohit data to creates 2D images of particle tracks/showers. Stores images in the SVGData folder.

#### MVCNNTrainer.py
Trains a multi-view convolutional neural network on some training images stored in SVGData. Stores the results in the TrainedModels folder.

#### MVCNNTester.py
Validates a multi-view convolutional network stored in TrainedModels, stores the results in the PickleData folder.

#### PfoGraphicalAnalyser.py
Gives a graphical view of PFOs, and provides other useful information. This aided algorithm development.

#### FeatureAnalyser.py
Presents the results of individual feature algorithms as a series of graphs/histograms. Uses the data stored by GetFeatureData.py.

#### PredictorAnalyser.py
Presents the results of Likelihood/BDT/MVCNN calculations as a series of graphs/histograms. Uses the data stored by LikelihoodCalculator.py, BDTCalculator.py, and MVCNNTester.py.

#### OpenPickledFigure.py
Graphs and histograms are saved as pickle files to Figures folder inside the analysis folder. This program opens these files. Can be modified to make changes to figures.

## Configuration
Inside each analysis folder is a config folder containing a configuration file for each of the programs above. It also contains:

#### GeneralConfig.py
Defines the separation classes, and some other shared options.

#### AlgorithmConfig.py
Contains any parameters for feature algorithms.

## Libraries
Also some libraries imported by the programs:
#### UpRootFileReader.py
Reads PFOs from ROOT files and converts them to our Python version of PFOs. Also contains particle type and MicroBooNE geometry info.

#### HistoSynthesis.py
Used to create fancy histogram plots.

#### PFOVertexing.py
Our attempt at an algorithm for calculating the vertex of a PFO.
