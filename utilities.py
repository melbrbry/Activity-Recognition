#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:24:27 2018

@author: elbarbari
"""
import random
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import pickle as pk



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
        data, labels, sequence_length = get_batch_ph_data(model_obj, one_batch_of_ids, mode="train")
#        print(labels)
        yield (data, labels, sequence_length)

def get_batch_ph_data(model_obj, one_batch_ids, mode="train"):
    batch_size = len(one_batch_ids)
    if mode=="train":
        d_data = model_obj.train_data
        d_labels = model_obj.train_labels
    if mode=="val":
        d_data = model_obj.val_data
        d_labels = model_obj.val_labels
    max_frames_in_batch, sequence_length = get_max_frames(one_batch_ids, d_data)
    frame_dim = model_obj.config.frame_dim
    noOfActivities = model_obj.config.no_of_activities
    data = -1.0*np.ones((batch_size, max_frames_in_batch, frame_dim))
    labels = np.zeros((batch_size,noOfActivities))
    for i, id in enumerate(one_batch_ids):
        frame_len = len(d_data[id])
        data[i][:frame_len] = d_data[id]
        labels[i] = d_labels[id]
    return data, labels, sequence_length

def parser(file):
    ids_per_frame, confs_per_frame, rois_per_frame = [], [], []
    with open(file, "rb") as fp:   
        vid = pk.load(fp)
        for frame in range(len(vid)):
            ids_per_frame.append(vid[frame]['class_ids'])
            confs_per_frame.append(vid[frame]['scores'])
            rois_per_frame.append(vid[frame]['rois'])
    return ids_per_frame, confs_per_frame, rois_per_frame
            
    
def get_max_frames(one_batch_ids, data):
    batch = []
    sequence_length = []
    for id in one_batch_ids:
        batch.append(data[id])
        sequence_length.append(len(data[id]))
    mx=0
    for vid in batch:
        mx = max(mx, len(vid))
    return mx, sequence_length

def print_array(array):
    for r in array:
        print(r)

def log(log_message):
    with open("log.txt", "a") as log_file:
        log_file.write(datetime.strftime(datetime.today(),
                    "%Y-%m-%d %H:%M:%S") + ": " + log_message)
        log_file.write("\n")

def plot_performance(model_dir):

    train_accuracies = pickle.load(open("%s/metrics/train_accuracies"\
                % model_dir, "rb"))
    val_accuracies = pickle.load(open("%s/metrics/val_accuracies"\
                % model_dir, "rb"))
    train_epochs_losses = pickle.load(open("%s/losses/train_epochs_losses" % model_dir, "rb"))
    val_epochs_losses = pickle.load(open("%s/losses/val_epochs_losses" % model_dir, "rb"))
    
    plt.figure(1)
    plt.plot(train_epochs_losses, "k^")
    plt.plot(train_epochs_losses, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/plots/train_epochs_losses.png" % model_dir)

    plt.figure(2)
    plt.plot(val_epochs_losses, "k^")
    plt.plot(val_epochs_losses, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/plots/val_epochs_losses.png" % model_dir)
    
    plt.figure(3)
    plt.plot(train_accuracies, "k^")
    plt.plot(train_accuracies, "k")
    plt.ylabel("train accuracy")
    plt.xlabel("epoch")
    plt.title("train accuracy per epoch")
    plt.savefig("%s/plots/train_accuracies.png" % model_dir)
    
    plt.figure(4)
    plt.plot(val_accuracies, "k^")
    plt.plot(val_accuracies, "k")
    plt.ylabel("val accuracy")
    plt.xlabel("epoch")
    plt.title("val accuracy per epoch")
    plt.savefig("%s/plots/val_accuracies.png" % model_dir)
    
