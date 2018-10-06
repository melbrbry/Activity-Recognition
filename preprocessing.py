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
videos_path = './activity-splitted JSON dataset/'
class2id = {'Blowing leaves': 0, 'Cutting the grass': 1, 'Fixing the roof': 2,
           'Mowing the lawn': 3, 'Painting fence': 4, 'Raking leaves': 5,
           'Roof shingle removal': 6, 'Shoveling snow': 7, 'Spread mulch': 8,
            'Trimming branches or hedges': 9}
        
def build_vid(ids_per_frame, confs_per_frame, rois_per_frame, areas_per_frame, centers_per_frame):
    vid = []
    for step_frame, frame_ids in enumerate(ids_per_frame):
        frame_features = []
        for object_id in range(noOfObjs):
            object_index = -1
            for i, it in enumerate(frame_ids):
                if it==object_id:
                    object_index = i
                    break
            if object_index == -1:
                for i in range(8):
                    frame_features.append(0)
            else:
                frame_features.append(confs_per_frame[step_frame][object_index])
                for i in range(4):
                    frame_features.append(rois_per_frame[step_frame][object_index][i])
                for i in range(2):
                    frame_features.append(max(0, centers_per_frame[step_frame][object_index][i]))
                frame_features.append(areas_per_frame[step_frame][object_index])                
        vid.append(frame_features)
    return vid
    
def collect_and_reformat(directory):
    videos = []
    for step, file in enumerate(os.listdir(directory)):
        if not file in ['data', 'labels']:
            ids_per_frame, confs_per_frame, rois_per_frame, areas_per_frame, centers_per_frame = ut.parser(directory+file)
            video = build_vid(ids_per_frame, confs_per_frame, rois_per_frame, areas_per_frame, centers_per_frame)
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
            for subDir in os.listdir(videos_path):
                for vid in os.listdir(videos_path+subDir):
                    if vid == file:
                        labels.append(class2id[subDir])
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
                copy(src, './JSON dataset/train/')
            if step >= train_ratio * no_of_videos \
                and step < (train_ratio+val_ratio) *no_of_videos:
                copy(src, './JSON dataset/val/')
            if step >= (train_ratio+val_ratio) *no_of_videos:
                copy(src, './JSON dataset/test/')

def test():    
    desFile = "./JSON dataset/train/data"
    with open(desFile, 'rb') as filehandle:  
        print(pickle.load(filehandle)[0][1])
    
def main():
    collect_and_resplit("./activity-splitted JSON dataset/", train_ratio=0.8,
                        val_ratio=0.1, test_ratio=0.1)
    collect_and_reformat("./JSON dataset/train/")
    collect_and_reformat("./JSON dataset/val/")
    collect_and_reformat("./JSON dataset/test/")
    collect_labels("./JSON dataset/train/")
    collect_labels("./JSON dataset/val/")
    collect_labels("./JSON dataset/test/")
    print("preprocessing done!")
    test()

if __name__ == '__main__':
    main()
