import os
import sys
import numpy as np
import torch
import torch.nn as nn

import time
import argparse
import re

from easydict import EasyDict as edict
from tqdm import trange

from helpers.models import EEGNetTest
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit

#for personal model, save the test prediction of each cv fold
def Online_train_classifier(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name = args_dict.sub_name
    Offline_folder_path = args_dict.Offline_folder_path
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch = args_dict.n_epoch
    order = args_dict.order
    motor_class = args_dict.motor_class
    data = args_dict.data
    session = args_dict.session
    trial = args_dict.trial
    
    model_to_use = EEGNetTest
    # orders
    if order == 1:
        print('*********Online Predicting***********')
    if order == 2:
        print('*********Online Updating***********')
    
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        #device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
    
    # get the lr and dropout value of the restore_file
    filename = restore_file
    match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", filename)
    if match:
        lr = float(match.group(1))
        dropout = float(match.group(2))
        # print(f"lr={lr}, dropout={dropout}")
    else:
        print("No match found.")
    
    #create model
    model = model_to_use(dropout=dropout).to(device)
    
    #reload weights from restore_file is specified
    if restore_file != 'None':
        restore_path = os.path.join(Online_result_save_rootdir, sub_name, restore_file, 'checkpoint/online_model.statedict')  # using the name online_model.statedict for all the online manipulations
        print('loading checkpoint: {}'.format(restore_path))
        model.load_state_dict(torch.load(restore_path, map_location=device)) 

    if order == 1:
        # order 1: recognize the MI class online without feedback 
        
        sub_val_feature_array = [data[0:-1,:]]
        sub_val_label_array = np.array([motor_class])
        group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)
        cv_val_batch_size = 1
        sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False) 
        _, class_predictions_array, _, _ = eval_model(model, sub_cv_val_loader, device)
        print(class_predictions_array)
        return class_predictions_array
    
    if order == 2:
        # order 2: update the MI recognition model online 
        
        train_list, train_label, scores = Online_read_csv(Online_folder_path,session,trial)
        sub_trian_set = brain_dataset(train_list, train_label)
        sub_cv_train_loader = torch.utils.data.DataLoader(sub_trian_set, batch_size=train_list.shape[0], shuffle=False) 
        
        start_time = time.time()
        experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
        print(experiment_name)
        #derived arg
        result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name, experiment_name)
        result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')

        makedir_if_not_exist(result_save_subjectdir)
        makedir_if_not_exist(result_save_subject_checkpointdir)
        
        #create model
        model = model_to_use(dropout=dropout).to(device)

        #create criterion and optimizer
        criterion = nn.NLLLoss() #for EEGNet and DeepConvNet, use nn.NLLLoss directly, which accept integer labels
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        #training loop
        epoch_train_loss = []
        epoch_train_accuracy = []

        for epoch in trange(n_epoch, desc='1-fold cross validation'):
            average_loss_this_epoch = train_one_epoch(model, optimizer, criterion, sub_cv_train_loader, device)
            train_accuracy, _, _ , _ = eval_model(model, sub_cv_train_loader, device)

            epoch_train_loss.append(average_loss_this_epoch)
            epoch_train_accuracy.append(train_accuracy)
            
        #update the model
        # torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.statedict'))
        
        #save the model
        torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'online_model.statedict'))
        print(os.path.join(result_save_subject_checkpointdir, 'online_model.statedict') + " model saved")
        end_time = time.time()
        total_time = end_time - start_time
        write_program_time(os.path.join(Online_result_save_rootdir, sub_name), total_time)    
    

"""
if __name__=='__main__':
    
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
    parser.add_argument('--sub_name', default='Jyt', type=str, help='name of the subject')
    parser.add_argument('--folder_path', default='./DataCollected_' + 'Jyt', help="folder to the dataset")
    parser.add_argument('--windows_num', default=149, type=int, help='number of windows')
    parser.add_argument('--proportion', default=0.8, type=float, help='proportion of the training set of the whole dataset')
    parser.add_argument('--result_save_rootdir', default='./Offline_experiments', help="Directory containing the dataset")
    parser.add_argument('--restore_file', default='None', help="xxx.statedict")
    parser.add_argument('--n_epoch', default=64, type=int, help="number of epoch")
    
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    sub_name = args.sub_name
    folder_path  = args.folder_path
    windows_num = args.windows_num
    proportion = args.proportion
    result_save_rootdir = args.result_save_rootdir
    restore_file = args.restore_file
    n_epoch = args.n_epoch
    
    #sanity check:
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))
    print('sub_name: {}, type: {}'.format(sub_name, type(sub_name)))
    print('folder_path: {}, type: {}'.format(folder_path, type(folder_path)))
    print('windows_num: {}, type: {}'.format(windows_num, type(windows_num)))
    print('proportion: {}, type: {}'.format(proportion, type(proportion)))
    print('result_save_rootdir: {}, type: {}'.format(result_save_rootdir, type(result_save_rootdir)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch: {} type: {}'.format(n_epoch, type(n_epoch)))
   
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.sub_name = sub_name
    args_dict.folder_path = folder_path
    args_dict.windows_num = windows_num
    args_dict.proportion = proportion
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.restore_file = restore_file
    args_dict.n_epoch = n_epoch

    seed_everything(seed)
    Online_train_classifier(args_dict)
"""
