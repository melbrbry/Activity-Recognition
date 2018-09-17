#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:19:57 2018

@author: elbarbari
"""

import os
import pickle
import utilities as ut
import numpy as np
from shutil import copy
from random import shuffle

noOfObjs = 39
noOfActivities = 10
videos_path = './activity-splitted dataset/'
class2id = {'Blowing leaves': 0, 'Cutting the grass': 1, 'Fixing the roof': 2,
           'Mowing the lawn': 3, 'Painting fence': 4, 'Raking leaves': 5,
           'Roof shingle removal': 6, 'Shoveling snow': 7, 'Spread mulch': 8,
            'Trimming branches or hedges': 9}
        
def to_hot(arr):
    alho = []
    for a in arr:
        ho = [0 for i in range(noOfObjs)]
        for i in a:
            ho[i] = 1
        alho.append(ho)
    return alho
    
def collect_and_reformat(directory):
    videos = []
    for file in os.listdir(directory):
        if not file in ['data', 'labels']:
            video = ut.parser(directory+file)
            video = to_hot(video)
            videos.append(video)
            
    desFile = directory + 'data'
    with open(desFile, 'wb') as filehandle:  
        pickle.dump(videos, filehandle)

def to_1hot(labels):
    ret = []
    for label in labels:
        hot = np.zeros(noOfActivities)
        hot[label]=1
        ret.append(hot)
    return ret
    
def collect_labels(directory):
    labels = []
    for file in os.listdir(directory):
        if not file in ['data', 'labels']:
            flag = False
            for subDir in os.listdir(videos_path):
                for vid in os.listdir(videos_path+subDir):
                    if vid == file:
                        labels.append(class2id[subDir])
                        flag = True
            if not flag:
                print("Vid not found!")
    labels = to_1hot(labels)
    desFile = directory + 'labels'
    with open(desFile, 'wb') as filehandle:  
        pickle.dump(labels, filehandle)

def collect_and_resplit(directory, train_ratio, val_ratio, test_ratio):
    for sub_dir in os.listdir(directory):
        no_of_videos = len(os.listdir(directory+sub_dir))
        all_vid = os.listdir(directory+sub_dir)
        shuffle(all_vid)
        for step, vid in enumerate(all_vid):
            src = directory + sub_dir + "/" + vid
            if step < train_ratio * no_of_videos:
                copy(src, './dataset/train/')
            if step >= train_ratio * no_of_videos \
                and step < (train_ratio+val_ratio) *no_of_videos:
                copy(src, './dataset/val/')
            if step >= (train_ratio+val_ratio) *no_of_videos:
                copy(src, './dataset/test/')
            
                    
    
def main():
#    collect_and_resplit("./activity-splitted dataset/", train_ratio=0.7,
#                        val_ratio=0.15, test_ratio=0.15)
    collect_and_reformat("./dataset/train/")
    collect_and_reformat("./dataset/val/")
    collect_and_reformat("./dataset/test/")
    collect_labels("./dataset/train/")
    collect_labels("./dataset/val/")
    collect_labels("./dataset/test/")
    print("preprocessing done!")

main()
