import pickle
import time
import numpy as np
import torch
import csv 
import os
import random
import logging
import shutil
import torch.nn.functional as F
import json
import math
import argparse

from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as sklearn_cm
import seaborn as sns

from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange


def load_pickle(result_dir, filename):
    with open(os.path.join(result_dir, filename), 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def makedir_if_not_exist(specified_dir):
    if not os.path.exists(specified_dir):
        os.makedirs(specified_dir)

        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#Mar23
def get_slope_and_intercept(column_values, return_value = 'w'):
    
    num_timesteps = len(column_values)
    print('num_timesteps: {}'.format(num_timesteps))
    tvec_T = np.linspace(0, 1, num_timesteps) #already asserted len(column_values) = 10
    tdiff_T = tvec_T - np.mean(tvec_T)
    
    w = np.inner(column_values - np.mean(column_values), tdiff_T) / np.sum(np.square(tdiff_T))
    b = np.mean(column_values) - w * np.mean(tvec_T)
    
    if return_value == 'w':
        return w
    
    elif return_value == 'b':
        return b
    
    else:
        raise Exception("invalid return_value")
        
        
def featurize(sub_feature_array, classification_task='four_class'):
    
    num_data = sub_feature_array.shape[0]
    num_features = sub_feature_array.shape[2]
    
    assert num_features == 8 #8 features
    
    transformed_sub_feature_array = []
    for i in range(num_data):
        this_chunk_data = sub_feature_array[i]
        this_chunk_column_means = np.mean(this_chunk_data, axis=0)
        this_chunk_column_stds = np.std(this_chunk_data, axis=0)
        this_chunk_column_slopes = np.array([get_slope_and_intercept(this_chunk_data[:,i], 'w') for i in range(num_features)])
        this_chunk_column_intercepts = np.array([get_slope_and_intercept(this_chunk_data[:,i], 'b') for i in range(num_features)])
        
        this_chunk_transformed_features = np.concatenate([this_chunk_column_means, this_chunk_column_stds, this_chunk_column_slopes, this_chunk_column_intercepts])
        
        transformed_sub_feature_array.append(this_chunk_transformed_features)
    
    return np.array(transformed_sub_feature_array)


def plot_confusion_matrix(predictions, true_labels, figure_labels, save_dir, filename):
    
    sns.set(color_codes=True)
    sns.set(font_scale=1.4)
    
    plt.figure(1, figsize=(8,5))
    plt.title('Confusion Matrix')
    
    data = sklearn_cm(true_labels, predictions)
    ax = sns.heatmap(data, annot=True, fmt='d', cmap='Blues')
    
    ax.set_xticklabels(figure_labels)
    ax.set_yticklabels(figure_labels)
    ax.set(ylabel='True Label', xlabel='Predicted Label')
    ax.set_ylim([4, 0])
    
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    

def save_training_curves_FixedTrainValSplit(figure_name, result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy=None, epoch_validation_accuracy = None, epoch_test_accuracy = None):
    
    fig = plt.figure(figsize=(15, 8))
    
    ax_1 = fig.add_subplot(1,4,1)
    ax_1.plot(range(len(epoch_train_loss)), epoch_train_loss, label='epoch_train_loss')
    
    if epoch_train_accuracy is not None:
        ax_2 = fig.add_subplot(1,4,2, sharex = ax_1)
        ax_2.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy, label='epoch_train_accuracy')
        ax_2.legend()
        
    if epoch_validation_accuracy is not None:
        ax_3 = fig.add_subplot(1,4,3, sharex = ax_1)
        ax_3.plot(range(len(epoch_validation_accuracy)), epoch_validation_accuracy, label='epoch_validation_accuracy')
        ax_3.legend()
    
    if epoch_test_accuracy is not None:
        ax_4 = fig.add_subplot(1,4,4)
        ax_4.plot(range(len(epoch_test_accuracy)), epoch_test_accuracy, label='epoch_test_accuracy')
        ax_4.legend()
    
    ax_1.legend()
        
    figure_save_path = os.path.join(result_save_subject_trainingcurvedir, figure_name)
    plt.savefig(figure_save_path)
    plt.close()
    

def save_training_curves_FixedTrainValSplit_overlaid(figure_name, result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy=None, epoch_validation_accuracy = None, epoch_test_accuracy = None):
    
    fig = plt.figure(figsize=(15, 8))
    
    ax_1 = fig.add_subplot(1,2,1)
    ax_1.plot(range(len(epoch_train_loss)), epoch_train_loss, label='epoch_train_loss')
    
    ax_2 = fig.add_subplot(1,2,2)
    ax_2.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy, label='epoch_train_accuracy')
    ax_2.plot(range(len(epoch_validation_accuracy)), epoch_validation_accuracy, label='epoch_validation_accuracy')
    ax_2.plot(range(len(epoch_test_accuracy)), epoch_test_accuracy, label='epoch_test_accuracy')

    ax_2.legend()
    
    ax_1.legend()
        
    figure_save_path = os.path.join(result_save_subject_trainingcurvedir, figure_name)
    plt.savefig(figure_save_path)
    plt.close()
    
def Offline_write_performance_info_FixedTrainValSplit(model_state_dict, result_save_subject_resultanalysisdir, highest_validation_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #write performance to file
    file_writer.write('highest validation accuracy: {}\n'.format(highest_validation_accuracy))
    #write model parameters to file
    file_writer.write('Model parameters:\n')
    
    if model_state_dict != 'NA':
        total_elements = 0
        for name, tensor in model_state_dict.items():
            file_writer.write('layer {}: {} parameters\n'.format(name, torch.numel(tensor)))
            total_elements += torch.numel(tensor)
        file_writer.write('total elemets in this model: {}'.format(total_elements))
    else:
        file_writer.write('total elemets in this model NA, sklearn model')
    
    file_writer.close()    

def Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model_state_dict, result_save_subject_resultanalysisdir, result_save_dict):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #write performance to file
    for key, value in result_save_dict.items():
        if key == 'bestepoch_val_accuracy':
            file_writer.write('highest validation accuracy: {}\n'.format(value))
        else:
            file_writer.write(key + ': {}\n'.format(value))

    #write model parameters to file
    file_writer.write('Model parameters:\n')
    
    if model_state_dict != 'NA':
        total_elements = 0
        for name, tensor in model_state_dict.items():
            file_writer.write('layer {}: {} parameters\n'.format(name, torch.numel(tensor)))
            total_elements += torch.numel(tensor)
        file_writer.write('total elemets in this model: {}'.format(total_elements))
    else:
        file_writer.write('total elemets in this model NA, sklearn model')
    
    file_writer.close()  

#Aug19
def write_performance_info_FixedTrainValSplit(model_state_dict, result_save_subject_resultanalysisdir, highest_validation_accuracy, corresponding_test_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #write performance to file
    file_writer.write('highest validation accuracy: {}\n'.format(highest_validation_accuracy))
    file_writer.write('corresponding test accuracy: {}\n'.format(corresponding_test_accuracy))
    #write model parameters to file
    file_writer.write('Model parameters:\n')
    
    if model_state_dict != 'NA':
        total_elements = 0
        for name, tensor in model_state_dict.items():
            file_writer.write('layer {}: {} parameters\n'.format(name, torch.numel(tensor)))
            total_elements += torch.numel(tensor)
        file_writer.write('total elemets in this model: {}'.format(total_elements))
    else:
        file_writer.write('total elemets in this model NA, sklearn model')
    
    file_writer.close()

def write_performance_info_FixedTrainValSplit_1(model_state_dict, result_save_subject_resultanalysisdir, combination_test_accuracy, corresponding_test_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #write performance to file
    file_writer.write('combination_test_accuracy: {}\n'.format(combination_test_accuracy))
    file_writer.write('highest validation accuracy corresponding test accuracy: {}\n'.format(corresponding_test_accuracy))
    #write model parameters to file
    file_writer.write('Model parameters:\n')
    
    if model_state_dict != 'NA':
        total_elements = 0
        for name, tensor in model_state_dict.items():
            file_writer.write('layer {}: {} parameters\n'.format(name, torch.numel(tensor)))
            total_elements += torch.numel(tensor)
        file_writer.write('total elemets in this model: {}'.format(total_elements))
    else:
        file_writer.write('total elemets in this model NA, sklearn model')
    
    file_writer.close()


def write_initial_test_accuracy(result_save_subject_resultanalysisdir, initial_test_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'initial_test_accuracy.txt'), 'w')
    
    #write performance to file
    file_writer.write('initial test accuracy: {}\n'.format(initial_test_accuracy))
    
    file_writer.close()

def write_program_time(result_save_subject_resultanalysisdir, time_in_seconds):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'program_time.txt'), 'w')
    
    #write performance to file
    file_writer.write('program_time: {} seconds \n'.format(round(time_in_seconds,2)))
    
    file_writer.close()

def write_inference_time(result_save_subject_resultanalysisdir, time_in_seconds):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'inference_time.txt'), 'w')
    
    #write performance to file
    file_writer.write('program_time: {} seconds \n'.format(round(time_in_seconds,2)))
    
    file_writer.close()

    
#Aug13
def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()

        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_fea(model, optimizer, criterion, train_loader, device):
    model.train()
    
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, _ = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()

        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_fea_MMDContrastive_targetcls_iter(model, optimizer, criterion, source_loader, target_loader, memoryBank_source, memoryBank_target, device, cons_beta=0.01):
    # based on the contrastive learning based method in:
    # D. Zhang, H. Li and J. Xie, Unsupervised and semi-supervised domain adaptation networks considering both global knowledge and prototype-based local class information for Motor Imagery Classification. Neural Networks (2024), doi: https://doi.org/10.1016/j.neunet.2024.106497. 
    
    model.train()
    # Convert memoryBank_source to torch tensor and move to device
    memoryBank_source = torch.from_numpy(memoryBank_source).to(device)
    memoryBank_target = torch.from_numpy(memoryBank_target).to(device)

    # considering that data in the source is much larger than that in target, so we use itertools for target data expansion 
    target_loader = itertools.cycle(target_loader)

    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        # check target_data size
        if len(target_data) != len(source_data):
            _batch_size = min(len(target_data), len(source_data))
            source_data = source_data[0:_batch_size,:,:]
            source_labels = source_labels[0:_batch_size]
            target_data = target_data[0:_batch_size,:,:]
            target_labels = target_labels[0:_batch_size]
        
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        #cls_loss = criterion(source_output, source_labels)
        
        cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        # Calculate MMD loss
        mmd_loss = mmd_loss_func(source_features, target_features)

        # calculate the Source contrastive loss
        _source_innerdot = torch.einsum('bct,nct->bn', source_features, memoryBank_source)
        source_cons_loss = criterion(_source_innerdot, source_labels)
        # calculate the Interactive contrastive loss for the Target
        _target_innerdot = torch.einsum('bct,nct->bn', target_features, memoryBank_source)
        target_cons_loss = criterion(_target_innerdot, target_labels)

        # Total loss is the sum of classification loss and MMD loss
        #loss = cls_loss + mmd_loss + source_cons_loss + target_cons_loss
        loss_transfer = mmd_loss + source_cons_loss + target_cons_loss
        loss = cls_loss + cons_beta * loss_transfer

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_fea_MMDContrastive_targetcls_iter_2(model, optimizer, criterioncls, criterioncons, criterionMMD, source_loader, target_loader, memoryBank_source, memoryBank_target, device, cons_beta=0.01):
    # based on the contrastive learning based method in:
    # D. Zhang, H. Li and J. Xie, Unsupervised and semi-supervised domain adaptation networks considering both global knowledge and prototype-based local class information for Motor Imagery Classification. Neural Networks (2024), doi: https://doi.org/10.1016/j.neunet.2024.106497. 
    # using the implementation of MMDLoss from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py

    model.train()
    # Convert memoryBank_source to torch tensor and move to device
    memoryBank_source = torch.from_numpy(memoryBank_source).to(device)
    memoryBank_target = torch.from_numpy(memoryBank_target).to(device)

    # considering that data in the source is much larger than that in target, so we use itertools for target data expansion 
    target_loader = itertools.cycle(target_loader)

    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        # check target_data size
        if len(target_data) != len(source_data):
            _batch_size = min(len(target_data), len(source_data))
            source_data = source_data[0:_batch_size,:,:]
            source_labels = source_labels[0:_batch_size]
            target_data = target_data[0:_batch_size,:,:]
            target_labels = target_labels[0:_batch_size]
        
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        #cls_loss = criterion(source_output, source_labels)
        
        cls_loss = criterioncls(source_output, source_labels) + criterioncls(target_output, target_labels)

        # Calculate MMD loss, the implementation from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
        b, c, t = source_features.shape  # needs a flattening
        mmd_loss = criterionMMD(source_features.view(b, -1), target_features.view(b, -1))  

        # calculate the Source contrastive loss
        _source_innerdot = torch.einsum('bct,nct->bn', source_features, memoryBank_source)
        source_cons_loss = criterioncons(_source_innerdot, source_labels)
        # calculate the Interactive contrastive loss for the Target
        _target_innerdot = torch.einsum('bct,nct->bn', target_features, memoryBank_source)
        target_cons_loss = criterioncons(_target_innerdot, target_labels)

        # Total loss is the sum of classification loss and MMD loss
        #loss = cls_loss + mmd_loss + source_cons_loss + target_cons_loss
        loss_transfer = mmd_loss + source_cons_loss + target_cons_loss
        loss = cls_loss + cons_beta * loss_transfer

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch


def train_one_epoch_fea_selfpace(model, optimizer, criterion, train_loader, device, lambda_pace, lambda_pace_new):
    model.train()
    
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        output_batch, _ = model(data_batch)
        
        #calculate loss per sample
        loss_per_sample = criterion(output_batch, labels_batch)
        
        #calculate weights for each sample based on their loss
        weights = loss_per_sample.detach()
        weights = torch.where(weights < lambda_pace_new, (torch.log(weights + torch.tensor([2-lambda_pace], device=device))/torch.log(torch.tensor([2-lambda_pace], device=device))), torch.zeros_like(weights))   # using the lambda_pace to calculate the weight v_i and use the lambda_pace_new to choose samples#calculate weighted loss
        #weights = torch.where(weights < lambda_pace_new, weights, torch.zeros_like(weights))   # using the lambda_pace to calculate the weight v_i and use the lambda_pace_new to choose samples#calculate weighted loss
        loss = (loss_per_sample * weights).sum()
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()

        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch


def train_one_epoch_fea_distillation(model, optimizer, criterion, sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader, device, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((oldcls_data_batch, oldcls_fea_batch), (newcls_fea_batch, newcls_labels_batch), (newdata_data_batch, newdata_label_bacth)) in enumerate(zip(sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader)):
        oldcls_data_batch = oldcls_data_batch.to(device) 
        oldcls_fea_batch = oldcls_fea_batch.to(device) 
        newcls_fea_batch = newcls_fea_batch.to(device)
        newcls_labels_batch = newcls_labels_batch.to(device)
        newdata_data_batch = newdata_data_batch.to(device)
        newdata_label_bacth = newdata_label_bacth.to(device)

        _, oldcls_data_batch_fea = model(oldcls_data_batch)
        _, newdata_data_batch_fea = model(newdata_data_batch)

        # caculate the distillation loss
        oldcls_feadistill_loss = criterion(oldcls_fea_batch, oldcls_data_batch_fea)
        
        newdata_MMD_loss = mmd_loss_func(newdata_data_batch_fea, newcls_fea_batch)
        
        # sum the loss
        loss = alpha * newdata_MMD_loss + (1 - alpha) * oldcls_feadistill_loss

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_logit_distillation(model, optimizer, criterion, train_loader, old_data_loader, device, T=2, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((data_batch, labels_batch), (old_data_batch, old_labels_batch)) in enumerate(zip(train_loader, old_data_loader)):
        data_batch = data_batch.to(device) 
        labels_batch = labels_batch.to(device) 
        old_data_batch = old_data_batch.to(device)
        old_labels_batch = old_labels_batch.to(device)

        output_batch, _ = model(data_batch)
        old_output_batch, _ = model(old_data_batch)

        # 计算新的类别数据的交叉熵损失函数
        ce_loss = criterion(output_batch, labels_batch)

        # 计算旧的类别数据的蒸馏损失函数
        soft_target = F.softmax(old_output_batch / T, dim=1)
        distillation_loss = F.kl_div(F.log_softmax(old_labels_batch / T, dim=1), soft_target, reduction='batchmean') * (T**2)

        # 将两部分的损失函数按照一定的比例相加
        loss = alpha * ce_loss + (1 - alpha) * distillation_loss

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_label_distillation(model, optimizer, criterion, train_loader, old_data_loader, device, T=2, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((data_batch, labels_batch), (old_data_batch, old_labels_batch)) in enumerate(zip(train_loader, old_data_loader)):
        data_batch = data_batch.to(device) 
        labels_batch = labels_batch.to(device) 
        old_data_batch = old_data_batch.to(device)
        old_labels_batch = old_labels_batch.to(device)

        output_batch, _ = model(data_batch)
        old_output_batch, _ = model(old_data_batch)

        # 计算新的类别数据的交叉熵损失函数
        ce_loss = criterion(output_batch, labels_batch)

        # 计算旧的类别数据的蒸馏损失函数
        #soft_target = F.softmax(old_output_batch / T, dim=1)
        #distillation_loss = F.kl_div(F.log_softmax(old_labels_batch / T, dim=1), soft_target, reduction='batchmean') * (T**2)
        distillation_loss = criterion(old_output_batch, old_labels_batch)

        # 将两部分的损失函数按照一定的比例相加
        loss = alpha * ce_loss + (1 - alpha) * distillation_loss

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_logitlabel_distillation(model, optimizer, criterion, train_loader, old_datalogit_loader, old_datalabel_loader, device, T=2, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((data_batch, labels_batch), (old_data_batch, old_logits_batch), (old_datalabel_batch, old_labels_batch)) in enumerate(zip(train_loader, old_datalogit_loader, old_datalabel_loader)):
        data_batch = data_batch.to(device) 
        labels_batch = labels_batch.to(device) 
        old_data_batch = old_data_batch.to(device)
        old_logits_batch = old_logits_batch.to(device)
        old_datalabel_batch = old_datalabel_batch.to(device)
        old_labels_batch = old_labels_batch.to(device)

        output_batch, _ = model(data_batch)
        old_output_batch, _ = model(old_data_batch)
        old_output_label, _ = model(old_datalabel_batch)

        # 计算新的类别数据的交叉熵损失函数
        ce_loss = criterion(output_batch, labels_batch)

        # 计算旧的类别数据的蒸馏损失函数
        soft_target = F.softmax(old_output_batch / T, dim=1)
        distillation_loss_logit = F.kl_div(F.log_softmax(old_logits_batch / T, dim=1), soft_target, reduction='batchmean') * (T**2)
        distillation_loss_label = criterion(old_output_label, old_labels_batch)

        # 将两部分的损失函数按照一定的比例相加
        loss = alpha * ce_loss + (1 - alpha) * (distillation_loss_logit + distillation_loss_label)

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_logitlabel_distillation_GME(model, optimizer, criterion, train_loader, old_datalogit_loader, old_datalabel_loader, device, T=2, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((data_batch, labels_batch), (old_data_batch, old_logits_batch), (old_datalabel_batch, old_labels_batch)) in enumerate(zip(train_loader, old_datalogit_loader, old_datalabel_loader)):
        data_batch = data_batch.to(device) 
        labels_batch = labels_batch.to(device) 
        old_data_batch = old_data_batch.to(device)
        old_logits_batch = old_logits_batch.to(device)
        old_datalabel_batch = old_datalabel_batch.to(device)
        old_labels_batch = old_labels_batch.to(device)

        output_batch, _ = model(data_batch)
        old_output_batch, _ = model(old_data_batch)
        old_output_label, _ = model(old_datalabel_batch)

        # 计算新的类别数据的交叉熵损失函数
        ce_loss = criterion(output_batch, labels_batch)

        # 计算旧的类别数据的蒸馏损失函数
        soft_target = F.softmax(old_output_batch / T, dim=1)
        distillation_loss_logit = F.kl_div(F.log_softmax(old_logits_batch / T, dim=1), soft_target, reduction='batchmean') * (T**2)
        distillation_loss_label = criterion(old_output_label, old_labels_batch)

        # 将两部分的损失函数按照一定的比例相加
        loss = alpha * ce_loss + (1 - alpha) * (distillation_loss_logit + distillation_loss_label)

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        ce_loss = alpha * ce_loss
        ce_loss.backward(retain_graph=True)

        # Get the gradients of the ce_loss
        ce_grad = [param.grad.clone() for param in model.parameters() if param.grad is not None]

        optimizer.zero_grad()
        distillation_loss = (1 - alpha) * (distillation_loss_logit + distillation_loss_label)
        distillation_loss.backward(retain_graph=True)

        # Get the gradients of the distillation_loss
        distillation_grad = [param.grad.clone() for param in model.parameters() if param.grad is not None]

        # Calculate the new gradient
        new_grad = []
        for g1, g2 in zip(ce_grad, distillation_grad):
            # Ensure the new gradient has an acute angle with the two old gradients
            # Minimize the Euclidean distance between the new gradient and the two old gradients
            new_g = (g1 + g2) / 2
            # Ensure the new gradient has the same shape as the original gradient
            if new_g.shape == g1.shape:
                new_grad.append(new_g)
            else:
                #print(f"Shape mismatch: {new_g.shape} vs {g1.shape}")
                new_grad.append(g1)  # Use the original gradient if the shapes do not match

        # Update the gradients of the model parameters
        for param, new_g in zip(model.parameters(), new_grad):
            if param.grad is not None and param.grad.shape == new_g.shape:
                param.grad = new_g
        
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch



def train_one_epoch_fealogitlabel_distillation(model, optimizer, criterion, train_loader, old_datalogit_loader, old_datalabel_loader, sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader, device, T=2, alpha=0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((data_batch, labels_batch), (old_data_batch, old_logits_batch), (old_datalabel_batch, old_labels_batch), (oldcls_data_batch, oldcls_fea_batch), (newcls_fea_batch, newcls_labels_batch), (newdata_data_batch, newdata_label_bacth)) \
        in enumerate(zip(train_loader, old_datalogit_loader, old_datalabel_loader, sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader)):
        
        data_batch = data_batch.to(device) 
        labels_batch = labels_batch.to(device) 
        old_data_batch = old_data_batch.to(device)
        old_logits_batch = old_logits_batch.to(device)
        old_datalabel_batch = old_datalabel_batch.to(device)
        old_labels_batch = old_labels_batch.to(device)
        oldcls_data_batch = oldcls_data_batch.to(device) 
        oldcls_fea_batch = oldcls_fea_batch.to(device) 
        newcls_fea_batch = newcls_fea_batch.to(device)
        newcls_labels_batch = newcls_labels_batch.to(device)
        newdata_data_batch = newdata_data_batch.to(device)
        newdata_label_bacth = newdata_label_bacth.to(device)

        output_batch, _ = model(data_batch)
        old_output_batch, _ = model(old_data_batch)
        old_output_label, _ = model(old_datalabel_batch)
        _, oldcls_data_batch_fea = model(oldcls_data_batch)
        _, newdata_data_batch_fea = model(newdata_data_batch)

        # 计算新的类别数据的交叉熵损失函数
        ce_loss = criterion(output_batch, labels_batch)

        # 计算旧的类别数据的蒸馏损失函数
        soft_target = F.softmax(old_output_batch / T, dim=1)
        distillation_loss_logit = F.kl_div(F.log_softmax(old_logits_batch / T, dim=1), soft_target, reduction='batchmean') * (T**2)
        distillation_loss_label = criterion(old_output_label, old_labels_batch)

        # caculate the distillation loss
        oldcls_feadistill_loss = F.mse_loss(oldcls_fea_batch, oldcls_data_batch_fea)
        
        newdata_MMD_loss = mmd_loss_func(newdata_data_batch_fea, newcls_fea_batch)
        
        # 将两部分的损失函数按照一定的比例相加
        loss = alpha * ce_loss + (1 - alpha) * (distillation_loss_logit + distillation_loss_label) + alpha * newdata_MMD_loss + (1 - alpha) * oldcls_feadistill_loss

        loss_avg.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_fea_momentum(model, optimizer, criterion, train_loader, device, alpha=0.7):
    model.train()
    
    # save the parameters of the model before updating
    theta_0 = {name: param.clone() for name, param in model.named_parameters()}

    loss_avg = RunningAverage()

    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, _ = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()

        #perform parameters update
        optimizer.step()
    
    # updating the model with the weighted average method  
    #print("updating the model with the alpha value: {}".format(alpha))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(alpha * theta_0[name] + (1 - alpha) * param)

    average_loss_this_epoch = loss_avg()

    return average_loss_this_epoch

def compute_kernel(x, y):
    return torch.exp(-torch.norm(x-y)**2 / 2)

def compute_mmd(x, y):
    min_size = min(x.size(0), y.size(0))
    x = x[:min_size]
    y = y[:min_size]
    xx = compute_kernel(x, x)
    yy = compute_kernel(y, y)
    xy = compute_kernel(x, y)
    return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)


def compute_mmd_avg(x, y):
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    return compute_kernel(x_mean, y_mean)

def mmd_loss_func(source_features, target_features):
    return compute_mmd(source_features, target_features)

def mmd_loss_func_avg(source_features, target_features):
    return compute_mmd_avg(source_features, target_features)

def train_one_epoch_MMD(model, optimizer, criterion, source_loader, target_loader, device):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        # Calculate MMD loss
        mmd_loss = mmd_loss_func(source_features, target_features)

        # Total loss is the sum of classification loss and MMD loss
        loss = cls_loss + mmd_loss

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_MMDavg(model, optimizer, criterion, source_loader, target_loader, device):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        # Calculate MMD loss
        mmd_loss = mmd_loss_func_avg(source_features, target_features)

        # Total loss is the sum of classification loss and MMD loss
        loss = cls_loss + mmd_loss

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_MMD_Weights(model, optimizer, criterion, source_loader, target_loader, device, accuracy_per_class):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Calculate weights for each sample based on its class's accuracy
        source_weights = torch.tensor([1 - accuracy_per_class[label] for label in source_labels]).to(device)
        target_weights = torch.tensor([1 - accuracy_per_class[label] for label in target_labels]).to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        #cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        cls_loss = (criterion(source_output, source_labels) * source_weights).mean() + (criterion(target_output, target_labels) * target_weights).mean()

        # Calculate MMD loss for each class
        #mmd_loss = mmd_loss_func(source_features, target_features)
        mmd_loss = 0
        for class_id in range(len(accuracy_per_class)):
            source_features_class = source_features[source_labels == class_id]
            target_features_class = target_features[target_labels == class_id]
            mmd_loss += (1 - accuracy_per_class[class_id]) * mmd_loss_func_avg(source_features_class, target_features_class)
            #mmd_loss += mmd_loss_func_avg(source_features_class, target_features_class)

        # Total loss is the sum of classification loss and MMD loss
        loss = cls_loss + mmd_loss

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def label_weights(label):
    if label == 0:
        return 1.0
    else:
        return 10.0

def train_one_epoch_MMD_Weights1(model, optimizer, criterion, source_loader, target_loader, device, accuracy_per_class):
    model.train()
    
    loss_avg = RunningAverage()
    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Calculate weights for each sample based on its class's accuracy
        source_weights = torch.tensor([label_weights(label) * (1 - accuracy_per_class[label]) for label in source_labels]).to(device)
        target_weights = torch.tensor([label_weights(label) * (1 - accuracy_per_class[label]) for label in target_labels]).to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        #cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        cls_loss = (criterion(source_output, source_labels) * source_weights).mean() + (criterion(target_output, target_labels) * target_weights).mean()

        # Calculate MMD loss for each class
        #mmd_loss = mmd_loss_func(source_features, target_features)
        mmd_loss = 0
        for class_id in range(len(accuracy_per_class)):
            source_features_class = source_features[source_labels == class_id]
            target_features_class = target_features[target_labels == class_id]
            mmd_loss += label_weights(class_id) * (1 - accuracy_per_class[class_id]) * mmd_loss_func_avg(source_features_class, target_features_class)
            #mmd_loss += mmd_loss_func_avg(source_features_class, target_features_class)

        # Total loss is the sum of classification loss and MMD loss
        loss = cls_loss + mmd_loss

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def compute_total_accuracy_per_class(accuracies_per_class_iterations):
    # Initialize a dictionary to store the total accuracy and count for each class
    accuracy_dict = {}

    # Iterate over the accuracies_per_class_iterations list
    for ground_truth_label, predict_accu in accuracies_per_class_iterations:
        # If the class is not in the dictionary, add it
        if ground_truth_label not in accuracy_dict:
            accuracy_dict[ground_truth_label] = {'total_accuracy': 0, 'count': 0}

        # Update the total accuracy and count for the class
        accuracy_dict[ground_truth_label]['total_accuracy'] += predict_accu
        accuracy_dict[ground_truth_label]['count'] += 1

    # Compute the total accuracy for each class and store it in a list
    accuracy_per_class_iteration = [accuracy_dict[i]['total_accuracy'] / (accuracy_dict[i]['count']) for i in sorted(accuracy_dict.keys())]
 
    return accuracy_per_class_iteration

def train_one_epoch_MMD_Momentum(model, optimizer, criterion, source_loader, target_loader, device, alpha = 0.7):
    model.train()
    
    loss_avg = RunningAverage()

    # save the parameters of the model before updating
    theta_0 = {name: param.clone() for name, param in model.named_parameters()}

    for i, ((source_data, source_labels), (target_data, target_labels)) in enumerate(zip(source_loader, target_loader)):
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        target_data = target_data.to(device)
        target_labels = target_labels.to(device)

        # Forward pass
        source_output, source_features = model(source_data)
        target_output, target_features = model(target_data)

        # Calculate classification loss
        cls_loss = criterion(source_output, source_labels) + criterion(target_output, target_labels)

        # Calculate MMD loss
        mmd_loss = mmd_loss_func(source_features, target_features)

        # Total loss is the sum of classification loss and MMD loss
        loss = cls_loss + mmd_loss

        # Update running average of the loss
        loss_avg.update(loss.item())

        # Clear previous gradients
        optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Perform parameters update
        optimizer.step()

    # updating the model with the weighted average method  
    print("updating the model with the alpha value: {}".format(alpha))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(alpha * theta_0[name] + (1 - alpha) * param)

    average_loss_this_epoch = loss_avg()
    
    return average_loss_this_epoch

def train_update(model, optimizer, criterion, train_loader, device, alpha = 0.5):
    model.train()
    
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()

        # save the parameters of the model before updating
        theta_0 = {name: param.clone() for name, param in model.named_parameters()}

        #perform parameters update
        optimizer.step()

        # updating the model with the weighted average method  
        print("updating the model with the alpha value: {}".format(alpha))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(alpha * theta_0[name] + (1 - alpha) * param)
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def data_transform(data_batch):
    # transpose the batch data from tufts dataset to the input form of fNIRS-preT
    x = torch.transpose(data_batch, 1,2)  # [b, 150, 8]->[b, 8, 150]
    x = torch.stack(list(x.chunk(2, dim=1)), dim=0)
    x = torch.transpose(x, 0,1)  # [b, 8, 150]->[b, 2, 4, 150]
    x = torch.stack(list(x.chunk(2, dim=2)), dim=0)
    x = torch.transpose(x, 0,1)
    x = torch.transpose(x, 1,2)  # [b, 2, 4, 150]->[b, 2, 2, 2, 150]
    x = torch.flatten(x, 3,4)  # [b, 2, 2, 2, 150]->[b, 2, 2, 300]
    
    return x

def train_one_epoch_fNIRS_T(model, optimizer, criterion, train_loader, device, epoch):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    flooding_level = [0.40, 0.38, 0.35]
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_transform(data_batch)  # transform the data form to the input of fNIRS-preT
        #  input shape of fNIRS-preT is [B, 2, fNIRS channels, sampling points]

        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_Ours_T(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        #  input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:  # pretrain 10; train/finetune 30
            b = flooding_level[0]
        elif epoch < 50:  # pretrain 30; train/finetune 50
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_Ours_T_chunk(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        #  input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, output_chunk = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch[:, 0])
        loss_chunk = criterion(output_chunk, labels_batch[:, 1])
        loss_all = loss + 0.1*loss_chunk

        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss_all = (loss_all - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss_all.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss_all.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_Ours_T_finetune(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        #  input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, output_chunk = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch[:, 0])
        # loss_chunk = criterion(output_chunk, labels_batch[:, 1])
        loss_all = loss #+ 0.5 * loss_chunk

        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss_all = (loss_all - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss_all.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss_all.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_Ours_T_Norm(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        # normalization of the input batch
        """
        data_batch_mean = data_batch.mean(dim=1).unsqueeze(dim=1)
        data_batch_std = data_batch.std(dim=1).unsqueeze(dim=1)
        data_batch = (data_batch-data_batch_mean)/data_batch_std
        """
        data_batch_max = torch.max(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch_min = torch.min(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch = (data_batch - data_batch_min)/(data_batch_max-data_batch_min)

        #  input shape of ours-T is [B, patches, patch_size]
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch


def train_one_epoch_Ours_T_Cluster(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        # normalization of the input batch
        """
        data_batch_mean = data_batch.mean(dim=1).unsqueeze(dim=1)
        data_batch_std = data_batch.std(dim=1).unsqueeze(dim=1)
        data_batch = (data_batch-data_batch_mean)/data_batch_std
        """
        """
        data_batch_max = torch.max(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch_min = torch.min(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch = (data_batch - data_batch_min)/(data_batch_max-data_batch_min)
        """
        # randomly select the indexes from the data_batch and labels_batch  
        b,_,_ = data_batch.shape
        index = random.sample(range(b), int(b/2))
        index_left = np.delete(np.arange(b), index)
        data_batch_0 = torch.index_select(data_batch, 0, torch.LongTensor(index))
        data_batch_1 = torch.index_select(data_batch, 0, torch.LongTensor(index_left))
        labels_batch_0 = torch.index_select(labels_batch, 0, torch.LongTensor(index))
        labels_batch_1 = torch.index_select(labels_batch, 0, torch.LongTensor(index_left))

        #  input shape of ours-T is [B, patches, patch_size]
        data_batch_0 = data_batch_0.to(device) #put inputs to device
        data_batch_1 = data_batch_1.to(device)
        labels_batch_0 = labels_batch_0.to(device)
        labels_batch_1 = labels_batch_1.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_cls_0, output_batch_0 = model(data_batch_0)
        output_cls_1, output_batch_1 = model(data_batch_1)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        clu_loss = 1/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
            -2/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum()
        cls_loss = criterion(output_batch_0, labels_batch_0) + criterion(output_batch_1, labels_batch_1)
        loss = 0.001*clu_loss + cls_loss  # clustering and classification loss
        # the loss weight is about from 0.001 to 0.01 

        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def add_noise(input_batch, channel_num, step_num, channels=8, time_steps=150):
    b,_,_ = input_batch.shape
    input_batch = rearrange(input_batch, 'b c h -> b h c')  # [b, 8, 150]
    # input_batch = input_batch.numpy()  # tranform to numpy
    select_channels = random.sample(range(channels), int(channel_num))
    select_steps = random.sample(range(time_steps), int(step_num))  # select time and channels
    for i in range(b):
        for select_channel in select_channels:
            mean = input_batch[i, select_channel, :].mean()
            std = input_batch[i, select_channel, :].std()
            input_batch[i, select_channel, torch.tensor(select_steps)] = \
                input_batch[i, select_channel, torch.tensor(select_steps)] + \
                    (torch.randn(len(select_steps))*std)

    input_batch = rearrange(input_batch, 'b c h -> b h c')  # back to [b, 150, 8]
    return input_batch 


def train_one_epoch_Ours_T_ClusterNoise(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        # normalization of the input batch
        """
        data_batch_mean = data_batch.mean(dim=1).unsqueeze(dim=1)
        data_batch_std = data_batch.std(dim=1).unsqueeze(dim=1)
        data_batch = (data_batch-data_batch_mean)/data_batch_std
        """
        """
        data_batch_max = torch.max(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch_min = torch.min(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch = (data_batch - data_batch_min)/(data_batch_max-data_batch_min)
        """
        # randomly select the indexes from the data_batch and labels_batch  
        b,_,_ = data_batch.shape
        index = random.sample(range(b), int(b/2))
        index_left = np.delete(np.arange(b), index)
        data_batch_0 = torch.index_select(data_batch, 0, torch.LongTensor(index))
        data_batch_1 = torch.index_select(data_batch, 0, torch.LongTensor(index_left))
        labels_batch_0 = torch.index_select(labels_batch, 0, torch.LongTensor(index))
        labels_batch_1 = torch.index_select(labels_batch, 0, torch.LongTensor(index_left))
        # randomly add some noise to the input data_batch 
        data_batch_2 = add_noise(data_batch_0, channel_num=4, step_num=30)
        data_batch_3 = add_noise(data_batch_1, channel_num=4, step_num=30)

        #  input shape of ours-T is [B, patches, patch_size]
        data_batch_0 = data_batch_0.to(device) #put inputs to device
        data_batch_1 = data_batch_1.to(device)
        data_batch_2 = data_batch_2.to(device) #put inputs to device
        data_batch_3 = data_batch_3.to(device)

        labels_batch_0 = labels_batch_0.to(device)
        labels_batch_1 = labels_batch_1.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_cls_0, output_batch_0 = model(data_batch_0)
        output_cls_1, output_batch_1 = model(data_batch_1)
        
        output_cls_2, output_batch_2 = model(data_batch_2)
        output_cls_3, output_batch_3 = model(data_batch_3)

        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        clu_loss = 2/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
            -2/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum()

        clu_loss_noise = 2/b * ( ((output_cls_0-output_cls_3)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
            + 2/b * ( ((output_cls_1-output_cls_2)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
            - 2/b * ( ((output_cls_0-output_cls_3)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum() \
            - 2/b * ( ((output_cls_1-output_cls_2)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum()
        
        cls_loss = criterion(output_batch_0, labels_batch_0) + criterion(output_batch_1, labels_batch_1)
        cls_loss_noise = criterion(output_batch_2, labels_batch_0) + criterion(output_batch_3, labels_batch_1)
        #loss = 0.001*clu_loss + cls_loss + 0.001 * clu_loss_noise + cls_loss_noise # clustering and classification loss
        loss = 0.001 * clu_loss + cls_loss + 0.001 * clu_loss_noise + 0.0 * cls_loss_noise # clustering and classification loss
        # the loss weight of clu_ is about from 0.001 to 0.01 

        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def train_one_epoch_Ours_T_ClusterNoise_1(model, optimizer, criterion, train_loader, device, epoch, flooding_level = [0.40, 0.38, 0.35]):
    # this func is used for the training of fNIRS-T model with label-smoothing and flooding trick 
    
    model.train()
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features]) in the 30s, sequence _length=150， num_features=8
        #labels: tensor on cpu, torch.Size([batch_size])
        
        # normalization of the input batch
        """
        data_batch_mean = data_batch.mean(dim=1).unsqueeze(dim=1)
        data_batch_std = data_batch.std(dim=1).unsqueeze(dim=1)
        data_batch = (data_batch-data_batch_mean)/data_batch_std
        """
        """
        data_batch_max = torch.max(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch_min = torch.min(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch = (data_batch - data_batch_min)/(data_batch_max-data_batch_min)
        """
        noise_standard = [2,2,2, 4,4, 6]
        # if epoch is less than 10, we make the clustering without noise(if epoch%2!=0), else cluster with noise(if epoch%2==0)
        if epoch < 12:
            b,_,_ = data_batch.shape # randomly select the indexes from the data_batch and labels_batch  
            index = random.sample(range(b), int(b/2))
            index_left = np.delete(np.arange(b), index)
            data_batch_0 = torch.index_select(data_batch, 0, torch.LongTensor(index))
            data_batch_1 = torch.index_select(data_batch, 0, torch.LongTensor(index_left))
            labels_batch_0 = torch.index_select(labels_batch, 0, torch.LongTensor(index))
            labels_batch_1 = torch.index_select(labels_batch, 0, torch.LongTensor(index_left))
            
            if (epoch%2):
                # randomly add some noise to the input data_batch 
                data_batch_2 = add_noise(data_batch_0, channel_num= noise_standard[int(epoch/2)], step_num=15*noise_standard[int(epoch/2)])
                data_batch_3 = add_noise(data_batch_1, channel_num= noise_standard[int(epoch/2)], step_num=15*noise_standard[int(epoch/2)])
            
            #  input shape of ours-T is [B, patches, patch_size]
            data_batch_0 = data_batch_0.to(device) #put inputs to device
            data_batch_1 = data_batch_1.to(device)
            
            labels_batch_0 = labels_batch_0.to(device)
            labels_batch_1 = labels_batch_1.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

            #forward pass
            #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
            output_cls_0, output_batch_0 = model(data_batch_0)
            output_cls_1, output_batch_1 = model(data_batch_1)
            
            #calculate loss
            #loss: tensor (scalar) on gpu, torch.Size([])
            clu_loss = 2/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
                -2/b * ( ((output_cls_0-output_cls_1)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum()
            loss = 0.001 * clu_loss

            if epoch%2 :
                
                data_batch_2 = data_batch_2.to(device) #put inputs to device
                data_batch_3 = data_batch_3.to(device)
                output_cls_2, output_batch_2 = model(data_batch_2)
                output_cls_3, output_batch_3 = model(data_batch_3)

                clu_loss_noise = 2/b * ( ((output_cls_0-output_cls_3)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
                    + 2/b * ( ((output_cls_1-output_cls_2)**2).sum(dim=1) * (labels_batch_0==labels_batch_1).float()).sum() \
                    - 2/b * ( ((output_cls_0-output_cls_3)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum() \
                    - 2/b * ( ((output_cls_1-output_cls_2)**2).sum(dim=1) * (labels_batch_0!=labels_batch_1).float()).sum()
                
                #cls_loss = criterion(output_batch_0, labels_batch_0) + criterion(output_batch_1, labels_batch_1)
                #cls_loss_noise = criterion(output_batch_2, labels_batch_0) + criterion(output_batch_3, labels_batch_1)
                #loss = 0.001*clu_loss + cls_loss + 0.001 * clu_loss_noise + cls_loss_noise # clustering and classification loss
                loss = 0.001 * clu_loss_noise  # clustering and classification loss
                # the loss weight of clu_ is about from 0.001 to 0.01 

        elif(epoch >= 12):
            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)
            output_cls, output_batch = model(data_batch)
            cls_loss = criterion(output_batch, labels_batch)
            loss = cls_loss

        # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
        if epoch < 30:
            b = flooding_level[0]
        elif epoch < 50:
            b = flooding_level[1]
        else:
            b = flooding_level[2]
        
        # flooding
        loss = (loss - b).abs() + b

        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch

def eval_model_fNIRST(model, eval_loader, device):
    
    # evaluation for the fNIRS-preT models with an input of [B, 2, fNIRS channels, sampling points]
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        data_batch = data_transform(data_batch)  # transform the data form to the input of fNIRS-preT
        # input shape of fNIRS-preT is [B, 2, fNIRS channels, sampling points]

        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

def eval_model_OursT(model, eval_loader, device):
    
    # evaluation for the ours-T models with an input of [B, patches, patch_size]
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        # input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array


def eval_model_OursT_Cluster(model, eval_loader, device):
    
    # evaluation for the ours-T models with an input of [B, patches, patch_size]
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        # input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        _, output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

def eval_model_OursT_chunk(model, eval_loader, device):
    
    # evaluation for the ours-T models with an input of [B, patches, patch_size]
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        # input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, _ = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch[:,0].data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch[:,0].data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

def eval_model_OursT_Norm(model, eval_loader, device):
    
    # evaluation for the ours-T models with an input of [B, patches, patch_size]
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        # normalization of the input batch
        """
        data_batch_mean = data_batch.mean(dim=1).unsqueeze(dim=1)
        data_batch_std = data_batch.std(dim=1).unsqueeze(dim=1)
        data_batch = (data_batch-data_batch_mean)/data_batch_std
        """
        data_batch_max = torch.max(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch_min = torch.min(data_batch, dim=1)[0].unsqueeze(dim=1)
        data_batch = (data_batch - data_batch_min)/(data_batch_max-data_batch_min)

        # input shape of ours-T is [B, patches, patch_size]

        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array


def eval_model_confusion_matrix(model, eval_loader, device):
    
    # the model will not only return the accuracy, but also return the confusion matrix and the accuracy of each class
    #set the model to evaluation mode
    model.eval()
    
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
            labels_array = labels_batch.data.cpu().numpy()
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0)
            
    class_predictions_array = probabilities_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
    
    # Compute confusion matrix
    conf_matrix = sklearn_cm(labels_array, class_predictions_array)
    
    # Compute per-class accuracies
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return accuracy, class_predictions_array, labels_array, probabilities_array, conf_matrix, class_accuracies

def eval_model_confusion_matrix_fea(model, eval_loader, device):
    
    # the model will not only return the accuracy, but also return the confusion matrix and the accuracy of each class
    #set the model to evaluation mode
    model.eval()
    
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        output_batch, _ = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
            labels_array = labels_batch.data.cpu().numpy()
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0)
            
    class_predictions_array = probabilities_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
    
    # Compute confusion matrix
    conf_matrix = sklearn_cm(labels_array, class_predictions_array)
    
    # Compute per-class accuracies
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return accuracy, class_predictions_array, labels_array, probabilities_array, conf_matrix, class_accuracies

def eval_model(model, eval_loader, device):
    
    #reference: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/evaluate.py
    #set the model to evaluation mode
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

def eval_model_fea(model, eval_loader, device):
    
    #reference: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/evaluate.py
    #set the model to evaluation mode
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, _ = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

def eval_model_fea_loss(model, eval_loader, criterion, device):
    
    #reference: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/evaluate.py
    #set the model to evaluation mode
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    loss_avg = RunningAverage()

    for data_batch, labels_batch in eval_loader:#test_loader
        # print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device)
        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch, _ = model(data_batch)
        
        loss = criterion(output_batch, labels_batch)
        loss_avg.update(loss.item())

        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    average_loss_this_epoch = loss_avg()
    return accuracy, class_predictions_array, labels_array, probabilities_array, average_loss_this_epoch

def eval_model_fea_exemplars(model, eval_loader, device, m):
    model.eval()
    output_feas_exemplars = []
    output_label_exemplars = []

    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        output_batch, output_fea = model(data_batch)

        # 计算特征向量的均值(全局均值)
        mean_fea = torch.mean(output_fea, dim=0)

        # 初始化 exemplars
        exemplars = torch.empty((0,output_fea.shape[-2],output_fea.shape[-1]), device=device)
        output_feas_exemplars = torch.empty((0,data_batch.shape[-2],data_batch.shape[-1]), device=device)
        exemplar_labels = []
        exemplar_sum = torch.zeros(mean_fea.shape).to(device)

        for i in range(m):

            # 计算每一个样本加入exemplar之后的均值和全局均值之间的距离
            distances = torch.norm(mean_fea.unsqueeze(0) - (output_fea + exemplar_sum.unsqueeze(0))/(i + 1), dim=(1,2))

            # 选择距离最小的样本
            index = torch.argmin(distances).item()
            # 更新 exemplars
            exemplars = torch.cat([exemplars, output_fea[index].unsqueeze(0)], dim=0)
            output_feas_exemplars = torch.cat([output_feas_exemplars, data_batch[index].unsqueeze(0)], dim=0)
            exemplar_labels.append(labels_batch[index].item())

            # 移除已选择的样本
            output_fea = torch.cat([output_fea[:index], output_fea[index+1:]])
            data_batch = torch.cat([data_batch[:index], data_batch[index+1:]])
            labels_batch = torch.cat([labels_batch[:index], labels_batch[index+1:]])

            # 计算当前 exemplars 的sum数值，用于后面更新均值用的
            exemplar_sum = torch.sum(exemplars, dim=0)

        output_feas_exemplars = output_feas_exemplars.cpu().numpy()
        output_label_exemplars = np.array(exemplar_labels)

    return output_feas_exemplars, output_label_exemplars

def eval_model_fea_exemplars_distillation(model, eval_loader, device, m):
    
    model.eval()

    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        output_logits, output_fea = model(data_batch)

        # 计算特征向量的均值(全局均值)
        mean_fea = torch.mean(output_fea, dim=0)

        # 初始化 exemplars
        exemplars = torch.empty((0,output_fea.shape[-2],output_fea.shape[-1]), device=device)
        output_feas_exemplars = torch.empty((0,data_batch.shape[-2],data_batch.shape[-1]), device=device)
        output_logits_exemplars = torch.empty((0, output_logits.shape[-1]), device=device)
        exemplar_sum = torch.zeros(mean_fea.shape).to(device)

        for i in range(m):

            # 计算每一个样本加入exemplar之后的均值和全局均值之间的距离
            distances = torch.norm(mean_fea.unsqueeze(0) - (output_fea + exemplar_sum.unsqueeze(0))/(i + 1), dim=(1,2))

            # 选择距离最小的样本
            index = torch.argmin(distances).item()
            # 更新 exemplars
            exemplars = torch.cat([exemplars, output_fea[index].unsqueeze(0)], dim=0)
            output_feas_exemplars = torch.cat([output_feas_exemplars, data_batch[index].unsqueeze(0)], dim=0)
            output_logits_exemplars = torch.cat([output_logits_exemplars, output_logits[index].unsqueeze(0)], dim=0)

            # 移除已选择的样本
            output_fea = torch.cat([output_fea[:index], output_fea[index+1:]])
            output_logits = torch.cat([output_logits[:index], output_logits[index+1:]])
            data_batch = torch.cat([data_batch[:index], data_batch[index+1:]])
            labels_batch = torch.cat([labels_batch[:index], labels_batch[index+1:]])

            # 计算当前 exemplars 的sum数值，用于后面更新均值用的
            exemplar_sum = torch.sum(exemplars, dim=0)

        output_feas_exemplars = output_feas_exemplars.cpu().numpy()
        output_logits_exemplars = output_logits_exemplars.detach().cpu().numpy()

    return output_feas_exemplars, output_logits_exemplars

def eval_model_fea_exemplars_distillation_label(model, eval_loader, device, m):
    
    model.eval()

    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        output_logits, output_fea = model(data_batch)

        # 计算特征向量的均值(全局均值)
        mean_fea = torch.mean(output_fea, dim=0)

        # 初始化 exemplars
        exemplars = torch.empty((0,output_fea.shape[-2],output_fea.shape[-1]), device=device)
        output_feas_exemplars = torch.empty((0,data_batch.shape[-2],data_batch.shape[-1]), device=device)
        output_logits_exemplars = torch.empty((0, output_logits.shape[-1]), device=device)
        exemplar_sum = torch.zeros(mean_fea.shape).to(device)
        exemplar_labels = []

        for i in range(m):

            # 计算每一个样本加入exemplar之后的均值和全局均值之间的距离
            distances = torch.norm(mean_fea.unsqueeze(0) - (output_fea + exemplar_sum.unsqueeze(0))/(i + 1), dim=(1,2))

            # 选择距离最小的样本
            index = torch.argmin(distances).item()
            # 更新 exemplars
            exemplars = torch.cat([exemplars, output_fea[index].unsqueeze(0)], dim=0)
            output_feas_exemplars = torch.cat([output_feas_exemplars, data_batch[index].unsqueeze(0)], dim=0)
            output_logits_exemplars = torch.cat([output_logits_exemplars, output_logits[index].unsqueeze(0)], dim=0)
            exemplar_labels.append(labels_batch[index].item())

            # 移除已选择的样本
            output_fea = torch.cat([output_fea[:index], output_fea[index+1:]])
            output_logits = torch.cat([output_logits[:index], output_logits[index+1:]])
            data_batch = torch.cat([data_batch[:index], data_batch[index+1:]])
            labels_batch = torch.cat([labels_batch[:index], labels_batch[index+1:]])

            # 计算当前 exemplars 的sum数值，用于后面更新均值用的
            exemplar_sum = torch.sum(exemplars, dim=0)

        output_feas_exemplars = output_feas_exemplars.cpu().numpy()
        output_logits_exemplars = output_logits_exemplars.detach().cpu().numpy()
        output_label_exemplars = np.array(exemplar_labels)

    return output_feas_exemplars, output_logits_exemplars, output_label_exemplars

def eval_model_fea_exemplars_distillation_datafea_logitlabel(model, eval_loader, device, m):
    
    model.eval()

    for data_batch, labels_batch in eval_loader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)
        output_logits, output_fea = model(data_batch)

        # 计算特征向量的均值(全局均值)
        mean_fea = torch.mean(output_fea, dim=0)

        # 初始化 exemplars
        exemplars = torch.empty((0,output_fea.shape[-2],output_fea.shape[-1]), device=device)
        output_data_exemplars = torch.empty((0, data_batch.shape[-2], data_batch.shape[-1]), device=device)
        output_feas_exemplars = torch.empty((0, output_fea.shape[-2], output_fea.shape[-1]), device=device)
        output_logits_exemplars = torch.empty((0, output_logits.shape[-1]), device=device)
        exemplar_sum = torch.zeros(mean_fea.shape).to(device)
        exemplar_labels = []

        for i in range(m):

            # 计算每一个样本加入exemplar之后的均值和全局均值之间的距离
            distances = torch.norm(mean_fea.unsqueeze(0) - (output_fea + exemplar_sum.unsqueeze(0))/(i + 1), dim=(1,2))

            # 选择距离最小的样本
            index = torch.argmin(distances).item()
            # 更新 exemplars
            exemplars = torch.cat([exemplars, output_fea[index].unsqueeze(0)], dim=0)
            
            output_data_exemplars = torch.cat([output_data_exemplars, data_batch[index].unsqueeze(0)], dim=0)
            output_feas_exemplars = torch.cat([output_feas_exemplars, output_fea[index].unsqueeze(0)], dim=0)
            output_logits_exemplars = torch.cat([output_logits_exemplars, output_logits[index].unsqueeze(0)], dim=0)
            exemplar_labels.append(labels_batch[index].item())

            # 移除已选择的样本
            output_fea = torch.cat([output_fea[:index], output_fea[index+1:]])
            output_logits = torch.cat([output_logits[:index], output_logits[index+1:]])
            data_batch = torch.cat([data_batch[:index], data_batch[index+1:]])
            labels_batch = torch.cat([labels_batch[:index], labels_batch[index+1:]])

            # 计算当前 exemplars 的sum数值，用于后面更新均值用的
            exemplar_sum = torch.sum(exemplars, dim=0)

        output_data_exemplars = output_data_exemplars.cpu().numpy()
        output_feas_exemplars = output_feas_exemplars.detach().cpu().numpy()
        output_logits_exemplars = output_logits_exemplars.detach().cpu().numpy()
        output_label_exemplars = np.array(exemplar_labels)

    return output_data_exemplars, output_feas_exemplars, output_logits_exemplars, output_label_exemplars

def eval_model_fea_classPrototypes(model, source_train_loader_prototypes, target_train_loader_prototypes, device, classes=3):
    # generate the class prototypes referring the works in:
    # D. Zhang, H. Li and J. Xie, Unsupervised and semi-supervised domain adaptation networks considering both global knowledge and prototype-based local class information for Motor Imagery Classification. Neural Networks (2024), doi: https://doi.org/10.1016/j.neunet.2024.106497. 
    model.eval()
    feas_source = None 
    labels_source = None
    feas_target = None
    labels_target = None
    memoryBank_source = None
    memoryBank_target = None

    # collect data from each batch of source data
    with torch.no_grad(): 
        for data_batch_source, labels_batch_source in source_train_loader_prototypes:
            
            data_batch_source = data_batch_source.to(device)
            labels_batch_source = labels_batch_source.to(device)
            _, fea_batch_source = model(data_batch_source)
            
            if feas_source is None:
                feas_source = fea_batch_source.detach().cpu().numpy()
            else:
                feas_source = np.concatenate((feas_source, fea_batch_source.detach().cpu().numpy()), axis=0)  # using no gradient data to calculate the prototypes
            
            if labels_source is None:
                labels_source = labels_batch_source.detach().cpu().numpy()
            else:
                labels_source = np.concatenate((labels_source, labels_batch_source.detach().cpu().numpy()), axis = 0)
        
        # calculate the prototypes of the source data 
        for class_idx in range(classes):
            # collect data from each class 
            indices_source = np.where(labels_source==class_idx)[0]
            _class_feas_source = feas_source[indices_source]
            # calculate the prototypes and store them in the memory bank
            _class_prototype_source = np.mean(_class_feas_source, axis=0)
            _class_prototype_source = np.expand_dims(_class_prototype_source, axis=0)

            if memoryBank_source is None:
                memoryBank_source = _class_prototype_source
            else:
                memoryBank_source = np.concatenate((memoryBank_source, _class_prototype_source),axis=0)

        for data_batch_target, labels_batch_target in target_train_loader_prototypes:
            
            data_batch_target = data_batch_target.to(device)
            labels_batch_target = labels_batch_target.to(device)
            _, fea_bacth_target = model(data_batch_target)

            if feas_target is None:
                feas_target = fea_bacth_target.detach().cpu().numpy()
            else:
                feas_target = np.concatenate((feas_target, fea_bacth_target.detach().cpu().numpy()), axis=0)  # using no gradient data to calculate the prototypes
            
            if labels_target is None:
                labels_target = labels_batch_target.detach().cpu().numpy()
            else:
                labels_target = np.concatenate((labels_target, labels_batch_target.detach().cpu().numpy()), axis = 0)
        # calculate the prototypes of the target data 
        for class_idx in range(classes):
            # collect data from each class 
            indices_target = np.where(labels_target==class_idx)[0]
            _class_feas_target = feas_target[indices_target]
            # calculate the prototypes and store them in the memory bank
            _class_prototype_target = np.mean(_class_feas_target, axis=0)
            _class_prototype_target = np.expand_dims(_class_prototype_target, axis=0)

            if memoryBank_target is None:
                memoryBank_target = _class_prototype_target
            else:
                memoryBank_target = np.concatenate((memoryBank_target, _class_prototype_target),axis=0)
    
    return memoryBank_source, memoryBank_target



class EarlyStopping(object):
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 1):
        """
        :param monitor: 要监测的指标，只有传入指标字典才会生效
        :param mode: 监测指标的模式,min 或 max
        :param patience: 最大容忍次数

        example:

        ```python
        # Initialize
        earlystopping = EarlyStopping(mode='max', patience=5)

        # call
        if earlystopping(val_accuracy):
           return;

        # save checkpoint

        state = {
            'model': model,
            'earlystopping': earlystopping.state_dict(),
            'optimizer': optimizer
        }

        torch.save(state, 'checkpoint.pth')

        checkpoint = torch.load('checkpoint.pth')
        earlystopping.load_state_dict(checkpoint['earlystopping'])
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.__value = -math.inf if mode == 'max' else math.inf
        self.__times = 0

    def state_dict(self) -> dict:
        """:保存状态，以便下次加载恢复
        torch.save(state_dict, path)
        """
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'value': self.__value,
            'times': self.__times
        }

    def load_state_dict(self, state_dict: dict):
        """:加载状态
        :param state_dict: 保存的状态
        """
        self.monitor = state_dict['monitor']
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self.__value = state_dict['value']
        self.__times = state_dict['times']

    def reset(self):
        """:重置次数
        """
        self.__times = 0

    def __call__(self, metrics) -> bool:
        """
        :param metrics: 指标字典或数值标量
        :return: 返回bool标量，True表示触发终止条件
        """
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        if (self.mode == 'min' and metrics <= self.__value) or (
                self.mode == 'max' and metrics >= self.__value):
            self.__value = metrics
            self.__times = 0
        else:
            self.__times += 1
        if self.__times >= self.patience:
            return True
        return False

def softmax(X):
    """
    np softmax
    """
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)

    return s

class RunningAverage():
    '''
    A class that maintains the running average of a quantity
    
    Usage example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)

    

def save_dict_to_json(d, json_path):
    """Saves dict of floats in josn file
    
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
    
    

def save_checkpoint(state, is_best, checkpoint):
    """Save model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves checkpoint + 'best.pth.tar'
    
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    
    else:
        print("Checkpoint Directory exists!")
    
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. 
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    
    return checkpoint
    
    
def write_model_info(model_state_dict, result_save_path, file_name):
    temp_file_name = os.path.join(result_save_path, file_name)
    
    auto_file = open(temp_file_name, 'w')
    total_elements = 0
    for name, tensor in model_state_dict.items():
        total_elements += torch.numel(tensor)
        auto_file.write('\t Layer {}: {} elements \n'.format(name, torch.numel(tensor)))

        #print('\t Layer {}: {} elements'.format(name, torch.numel(tensor)))
    auto_file.write('\n total elemets in this model state_dict: {}\n'.format(total_elements))
    #print('\n total elemets in this model state_dict: {}\n'.format(total_elements))
    auto_file.close()
    

def bootstrapping(candidate_subjects, lookup_table, num_bootstrap_samples=5000, upper_percentile=97.5, lower_percentile=2.5):
    '''
    To generate 1 bootstrap sample, first sample the 71 subjects to include for this bootstrap sample.
    Then for each included subject, sample the chunk to calculate accuracy for this subject
    '''
    
    rng = np.random.RandomState(0)
    
    num_candidate_subjects = len(candidate_subjects)
    
    bootstrap_accuracy_list = [] # each element is the accuracy of this bootstrap sample (the average accuracy of the selected subjects with their selected chunks in this bootstrap sample)
    for i in range(num_bootstrap_samples):
        print('sample: {}'.format(i))
        #sample the subjects to include for this sample
        subject_location_ix = np.array(range(num_candidate_subjects))
        bootstrap_subject_location_ix = rng.choice(subject_location_ix, num_candidate_subjects, replace=True)
        bootstrap_subjects = candidate_subjects[bootstrap_subject_location_ix]
#         print('subject to include for this sample: {}'.format(bootstrap_subjects))
        
        #for each selected subject, independently resample the chunks to include (as the test set for this subject)
        subject_accuracies = []
        for subject_id in bootstrap_subjects:
            #load the test predictions (for the selected hyper setting) of this subject, and the corresponding true labels
            ResultSaveDict_this_subject_path = lookup_table.loc[lookup_table['subject_id']==subject_id].experiment_folder.values[0]
            ResultSaveDict_this_subject = load_pickle(ResultSaveDict_this_subject_path, 'predictions/result_save_dict.pkl')

            TestLogits_this_subject = ResultSaveDict_this_subject['bestepoch_test_logits']
            TrueLabels_this_subject = ResultSaveDict_this_subject['bestepoch_test_class_labels']

            #bootstrap the chunks to include for this subject (at this bootstrap sample)
            chunk_location_ix = np.array(range(len(TrueLabels_this_subject)))
            bootstrap_chunk_location_ix = rng.choice(chunk_location_ix, len(TrueLabels_this_subject), replace=True)
            bootstrap_chunks_logits = TestLogits_this_subject[bootstrap_chunk_location_ix]
            bootstrap_chunks_true_labels = TrueLabels_this_subject[bootstrap_chunk_location_ix]

            accuracy_this_subject = (bootstrap_chunks_logits.argmax(1) == bootstrap_chunks_true_labels).mean()*100

            subject_accuracies.append(accuracy_this_subject)
        
        
        average_accuracy_this_bootstrap_sample = np.mean(np.array(subject_accuracies))
        bootstrap_accuracy_list.append(average_accuracy_this_bootstrap_sample)
    
    bootstrap_accuracy_array = np.array(bootstrap_accuracy_list)
    
    accuracy_upper_percentile = np.percentile(bootstrap_accuracy_array, upper_percentile)
    accuracy_lower_percentile = np.percentile(bootstrap_accuracy_array, lower_percentile)
    accuracy_median = np.percentile(bootstrap_accuracy_array, 50)

#     return accuracy_upper_percentile, accuracy_lower_percentile, bootstrap_accuracy_df
    return accuracy_upper_percentile, accuracy_lower_percentile, accuracy_median

def accuracy_iteration_plot(predict_accuracies, save_dir, filename='accuracies_iterations_plot.png'):
    # 创建一个新的图形
    plt.figure()

    # 绘制精度随着迭代次数的变化
    plt.plot(predict_accuracies)

    # 添加标题和标签
    plt.title('Accuracy over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图形
    plt.savefig(os.path.join(save_dir, filename))

    # 关闭图形以释放内存
    plt.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
import os
import pandas as pd

def accuracy_save2csv(predict_accuracies, save_dir, filename='predict_accuracies.csv'):
    # 创建一个 DataFrame
    df = pd.DataFrame(predict_accuracies, columns=['Accuracy'])

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存 DataFrame 到 csv 文件
    df.to_csv(os.path.join(save_dir, filename), index=False)


def save_results_online(class_predictions_array, labels_array, result_save_subject_resultanalysisdir):
    # 定义保存结果的路径
    results_file = os.path.join(result_save_subject_resultanalysisdir, 'results_online.csv')

    # 检查文件是否存在，如果不存在则创建一个新的DataFrame
    if not os.path.exists(results_file):
        results_df = pd.DataFrame(columns=['Prediction', 'Label'])
    else:
        results_df = pd.read_csv(results_file)

    # 添加新的预测结果和标签
    new_result = pd.DataFrame({'Prediction': class_predictions_array, 'Label': labels_array})
    results_df = results_df.append(new_result, ignore_index=True)

    # 保存结果
    results_df.to_csv(results_file, index=False)

def save_accuracies_per_class_online(args_dict, file_name):
    # save the three values: 
    # args_dict.accuracies_per_class_iterations.append([motor_class, predict_accu/100])
    # args_dict.predict_accuracies.append(predict_accu)
    # args_dict.accuracies_per_class.append(accuracy_per_class)
    # the file_name is in the form of .pkl file
    file_name = os.path.join(file_name, 'accuracies_per_class_online.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump({
            'accuracies_per_class_iterations': args_dict.accuracies_per_class_iterations,
            'predict_accuracies': args_dict.predict_accuracies,
            'accuracies_per_class': args_dict.accuracies_per_class,
            'accuracies_per_class_iterations_Rest':args_dict.accuracies_per_class_iterations_Rest
        }, f)

def load_accuracies_per_class_online(file_name):
    # load the three values: 
    # args_dict.accuracies_per_class_iterations.append([motor_class, predict_accu/100])
    # args_dict.predict_accuracies.append(predict_accu)
    # args_dict.accuracies_per_class.append(accuracy_per_class)
    # the file_name is in the form of .pkl file
    file_name = os.path.join(file_name, 'accuracies_per_class_online.pkl')
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data 

def calculate_accuracy_per_class_online(result_save_subject_resultanalysisdir, best_validation_class_accuracy):
    # 定义结果文件的路径
    results_file = os.path.join(result_save_subject_resultanalysisdir, 'results_online.csv')

    # 读取结果文件
    results_df = pd.read_csv(results_file)

    # 获取所有的类别
    classes = np.sort(results_df['Label'].unique())

    # 初始化一个列表来存储每一类的准确率
    accuracy_per_class_iter = [0] * len(best_validation_class_accuracy)

    # 对每一类进行循环
    for cls in classes:
        # 获取这一类的所有预测结果和实际标签
        predictions = results_df[results_df['Label'] == cls]['Prediction']
        labels = results_df[results_df['Label'] == cls]['Label']

        # 计算这一类的准确率
        accuracy = np.mean(predictions == labels)
        
        # 将准确率添加到列表中
        accuracy_per_class_iter[cls] = accuracy

    # 计算每一类的平均准确率
    average_accuracy_per_class = [(a + b) / 2 for a, b in zip(accuracy_per_class_iter, best_validation_class_accuracy)]

    return average_accuracy_per_class

def save_best_validation_class_accuracy_offline(best_validation_class_accuracy, summary_save_dir):
    # 定义结果文件的路径
    results_file = os.path.join(summary_save_dir, 'best_validation_class_accuracy.csv')

    # 创建一个DataFrame并保存到CSV文件
    df = pd.DataFrame(best_validation_class_accuracy, columns=['Accuracy'])
    df.to_csv(results_file, index=False)

def load_best_validation_class_accuracy_offline(summary_save_dir):
    # 定义结果文件的路径
    results_file = os.path.join(summary_save_dir, 'best_validation_class_accuracy.csv')

    # 从CSV文件中读取数据
    df = pd.read_csv(results_file)

    # 将数据转换为列表
    best_validation_class_accuracy = df['Accuracy'].tolist()

    return best_validation_class_accuracy

def load_best_validation_path_offline(summary_save_dir):
    # 定义结果文件的路径
    path_file = os.path.join(summary_save_dir, 'best_validation_model.txt')

    # 从文件中读取best_validation_path
    with open(path_file, 'r') as f:
        best_validation_path = f.readline().strip()

    return best_validation_path

def accuracy_perclass_save2csv(accuracy_per_class_iters, save_dir, filename='predict_perclass_accuracies.csv'):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建完整的文件路径
    file_path = os.path.join(save_dir, filename)

    # 将数据转换为DataFrame并保存为.csv文件
    df = pd.DataFrame(accuracy_per_class_iters)
    df.to_csv(file_path, index=False, header=False)

def accuracy_perclass_iteration_plot(accuracy_per_class_iters, save_dir, filename='accuracies_perclass_iterations_plot.png'):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建完整的文件路径
    file_path = os.path.join(save_dir, filename)

    # 将数据转换为列表并绘制图像
    plt.figure(figsize=(10, 6))
    for i in range(len(accuracy_per_class_iters[0])):
        accuracies = [row[i] for row in accuracy_per_class_iters]
        plt.plot(accuracies, label=f'Class {i}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Class over Iterations')
    plt.legend()
    plt.savefig(file_path)
    plt.close()

class MultiClassFocalLoss(nn.Module):
    def __init__(self, device, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
from torch.nn.functional import one_hot

class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    <https://arxiv.org/abs/2204.12511>
    """
    def __init__(self, num_classes=1000, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
    def forward(self, output, target):
        ce = self.criterion(output, target)
        pt = one_hot(target, num_classes=self.num_classes) * self.softmax(output)
        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()
    
class MMD_loss(nn.Module):
    # from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
    # the implementation of mmd loss from Jingdong Wang, a Senior Researcher at Microsoft Research Asia (MSRA).
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss