import numpy as np

############################################## LIKELIHOOD CALCULATOR CONFIGURATION ##################################################

features = [
    #{'name': 'TotalChargeU', 'algorithmName': 'TotalCharge', 'pdfBins': np.linspace(0, 100000, num = 50)},
    #{'name': 'TotalChargeV', 'algorithmName': 'TotalCharge', 'pdfBins': np.linspace(0, 100000, num = 50)},
    #{'name': 'TotalChargeW', 'algorithmName': 'TotalCharge', 'pdfBins': np.linspace(0, 100000, num = 50)},
    #{'name': 'TotalCharge3D', 'algorithmName': 'TotalCharge', 'pdfBins': np.linspace(0, 100000, num = 50)},
    #{'name': 'AngleFromX_3D', 'algorithmName': 'AnglesFromAxes', 'pdfBins': np.linspace(0, np.pi, num = 50)},
    #{'name': 'AngleFromY_3D', 'algorithmName': 'AnglesFromAxes', 'pdfBins': np.linspace(0, np.pi, num = 50)},
    #{'name': 'AngleFromZ_3D', 'algorithmName': 'AnglesFromAxes', 'pdfBins': np.linspace(0, np.pi, num = 50)},
    #{'name': 'VertexSepU', 'algorithmName': 'VertexSeparation'},
    #{'name': 'VertexSepV', 'algorithmName': 'VertexSeparation'},
    #{'name': 'VertexSepW', 'algorithmName': 'VertexSeparation'},
    #{'name': 'VertexSep3D', 'algorithmName': 'VertexSeparation'},
    {'name': 'TestDeDxU', 'algorithmName': 'TestDeDx1', 'pdfBins': np.linspace(0, 1500, num=50)},
    {'name': 'TestDeDxV', 'algorithmName': 'TestDeDx1', 'pdfBins': np.linspace(0, 1500, num=50)},
    {'name': 'TestDeDxW', 'algorithmName': 'TestDeDx1', 'pdfBins': np.linspace(0, 1500, num=50)},
    {'name': 'TestDeDx3D', 'algorithmName': 'TestDeDx1', 'pdfBins': np.linspace(0, 2000, num=50)},
    {'name': 'TestDeDx2U', 'algorithmName': 'TestDeDx2', 'pdfBins': np.linspace(0, 1000, num=50)},
    {'name': 'TestDeDx2V', 'algorithmName': 'TestDeDx2', 'pdfBins': np.linspace(0, 1000, num=50)},
    {'name': 'TestDeDx2W', 'algorithmName': 'TestDeDx2', 'pdfBins': np.linspace(0, 1000, num=50)},
    {'name': 'TestDeDx2_3D', 'algorithmName': 'TestDeDx2', 'pdfBins': np.linspace(0, 2000, num=50)},
    {'name': 'dedxU', 'algorithmName': 'NewInitialDeDx', 'pdfBins': np.linspace(0, 1200, num=50)},
    {'name': 'dedxV', 'algorithmName': 'NewInitialDeDx', 'pdfBins': np.linspace(0, 1200, num=50)},
    {'name': 'dedxW', 'algorithmName': 'NewInitialDeDx', 'pdfBins': np.linspace(0, 1200, num=50)},
]