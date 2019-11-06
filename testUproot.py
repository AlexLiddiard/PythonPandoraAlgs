#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:10:52 2019

@author: jack
"""

import uproot as up
import pandas as pd

file = up.open("/home/jack/Documents/Pandora/PythonPandoraAlgs/Pandora_Events_0a0f9054-bdee-4535-bd7f-2cc94620910d.root")
tree = file["PFOs"].pandas.df(flatten=False)

for index, branch in tree.iterrows():
    print(branch.get("Vertex[0]"))
    break