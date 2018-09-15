#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:24:27 2018

@author: elbarbari
"""
import os
import random
import numpy as np

class2id = {'label1':0, 'label2':1, 'label3':2}

def get_train_batches(model_obj):
    batch_size = model_obj.config.batch_size
    all_ids = []
    for i in range(len(model_obj.train_data)):
        all_ids.append(i)
    random.shuffle(all_ids)
#    print(all_ids)
    no_of_full_batches = int(len(all_ids)/batch_size)
    batches_of_ids = []
    for i in range(no_of_full_batches):
        one_batch_ids = all_ids[i*batch_size:(i+1)*batch_size]
        batches_of_ids.append(one_batch_ids)
    random.shuffle(batches_of_ids)
#    print(batches_of_ids)
    return batches_of_ids

def train_data_iterator(model_obj):    
    batches_of_ids = get_train_batches(model_obj)
    for one_batch_of_ids in batches_of_ids:
#        print(one_batch_of_ids)
        data, labels, sequence_length = get_batch_ph_train_data(model_obj, one_batch_of_ids)
#        print(labels)
        yield (data, labels, sequence_length)

def get_batch_ph_train_data(model_obj, one_batch_ids):
    batch_size = model_obj.config.batch_size
    max_frames_in_batch, sequence_length = get_max_frames(one_batch_ids, model_obj.train_data)
    frame_dim = model_obj.config.frame_dim
    noOfActivities = model_obj.config.no_of_activities
    data = -1.0*np.ones((batch_size, max_frames_in_batch, frame_dim))
    labels = np.zeros((batch_size,noOfActivities))
    for i, id in enumerate(one_batch_ids):
        frame_len = len(model_obj.train_data[id])
        data[i][:frame_len] = model_obj.train_data[id]
        labels[i] = model_obj.train_labels[id]
    return data, labels, sequence_length

def get_number_of_objects():
    return len(class2id)

def parser(file):
    f = open(file)
    lines = f.readlines()
    arr = []
    for line in lines:
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace('\n', '')
        line = line.replace('  ', ' ')
        array = line.split(' ')
        try:
            intArray = [int(elements) for elements in array]
        except Exception:
            intArray = []
        arr.append(intArray[1:])
    return arr
    
def get_max_frames(one_batch_ids, train_data):
    batch = []
    sequence_length = []
    for id in one_batch_ids:
        batch.append(train_data[id])
        sequence_length.append(len(train_data[id]))
    mx=0
    for vid in batch:
        mx = max(mx, len(vid))
    return mx, sequence_length