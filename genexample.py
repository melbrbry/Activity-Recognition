#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:46:15 2018

@author: elbarbari
"""

import pickle

vid = [[1, 8, 2],
       [7, 9, 12, 14]]

file = "./dataset/train/fI12XNNqldA"
with open(file, 'wb') as filehandle:  
    video = pickle.dump(vid, filehandle)