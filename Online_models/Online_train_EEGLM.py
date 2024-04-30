import os
import sys
import numpy as np
import torch
import torch.nn as nn

import time
import argparse
import re
import shutil

from easydict import EasyDict as edict
from tqdm import trange

from helpers.models import EEGNetTest, ConvEncoderResBN, ConvEncoderClsFea, ResEncoderfinetune, ConvEncoder3_ClsFeaTL
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, \
    train_one_epoch, train_update, eval_model, eval_model_confusion_matrix_fea, save_training_curves_FixedTrainValSplit, \
        write_performance_info_FixedTrainValSplit, write_program_time, train_one_epoch_MMD_Weights, train_one_epoch_MMD, eval_model_fea, train_one_epoch_fea, \
            compute_total_accuracy_per_class, softmax, save_results_online, calculate_accuracy_per_class_online, load_best_validation_class_accuracy_offline,\
            load_best_validation_path_offline
        
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix

#for personal model, save the test prediction of each cv fold
def Online_train_classifierLM(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    sub_name_offline = args_dict.sub_name_offline
    sub_name_online = args_dict.sub_name_online
    Offline_folder_path = args_dict.Offline_folder_path
    Online_folder_path = args_dict.Online_folder_path
    windows_num = args_dict.windows_num
    proportion = args_dict.proportion
    Offline_result_save_rootdir = args_dict.Offline_result_save_rootdir
    Online_result_save_rootdir = args_dict.Online_result_save_rootdir
    restore_file = args_dict.restore_file
    n_epoch_online = args_dict.n_epoch_online
    order = args_dict.order
    motor_class = args_dict.motor_class
    data = args_dict.data
    session = args_dict.session
    trial = args_dict.trial
    batch_size_online = args_dict.batch_size_online
    best_validation_path = args_dict.best_validation_path
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    use_pretrain = args_dict.use_pretrain
    trial_pre = args_dict.trial_pre
    trial_pre_target = args_dict.trial_pre_target
    window = args_dict.winodw
    restore_path_encoder = args_dict.restore_path_encoder
    restore_path_encoder_output = args_dict.restore_path_encoder_output
    device = args_dict.device
    lr = args_dict.lr
    dropout = args_dict.dropout
    best_validation_class_accuracy = args_dict.best_validation_class_accuracy
    sub_train_feature_array_offline = args_dict.sub_train_feature_array_offline
    sub_train_label_array_offline = args_dict.sub_train_label_array_offline
    sub_val_feature_array_offline = args_dict.sub_val_feature_array_offline
    sub_val_label_array_offline = args_dict.sub_val_label_array_offline

    # orders
    if order == 1:
        print('*********Online Predicting***********')
    if order == 2:
        print('*********Online Updating***********')
    
    #create model
    if use_pretrain:
        encoder_to_use = ConvEncoderResBN(in_features=32, encoder_h=512)
        encoder_to_use_output = ConvEncoderClsFea(output_h=128, dropout=dropout)
    else:
        encoder_to_use_output = ConvEncoder3_ClsFeaTL(in_features=32, output_h=128, dropout=dropout)
    
    restore_file = best_validation_path
    #reload weights from restore_file is specified
    if restore_file != 'None':
        # load the model
        if use_pretrain:
            encoder_to_use.load(filename=restore_path_encoder)  
            encoder_to_use.freeze_features(unfreeze=unfreeze_encoder_online)  # We freeze the model parameters
            encoder_to_use_output.load(filename=restore_path_encoder_output)
            #print('loading checkpoint encoder: {}'.format(restore_path_encoder))
            #print('loading checkpoint encoder: {}'.format(restore_path_encoder_output))
        else:
            encoder_to_use_output.load(filename=restore_path_encoder_output)
            #print('loading checkpoint encoder: {}'.format(restore_path_encoder_output))
    
    if use_pretrain:
        model_to_use = ResEncoderfinetune(encoder=encoder_to_use, encoder_output=encoder_to_use_output).double()
        model = model_to_use.to(device)
    else:
        model_to_use = encoder_to_use_output.double()

    model = model_to_use.to(device)
    #derived arg
    experiment_name = best_validation_path
    result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name_online, experiment_name)
    result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
    result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
    result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
    result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

    makedir_if_not_exist(result_save_subjectdir)
    makedir_if_not_exist(result_save_subject_checkpointdir)
    makedir_if_not_exist(result_save_subject_predictionsdir)
    makedir_if_not_exist(result_save_subject_resultanalysisdir)
    makedir_if_not_exist(result_save_subject_trainingcurvedir)

    if order == 1:
        # order 1: recognize the MI class online without feedback 
        sub_val_feature_array = [data[0:-1,:]]
        sub_val_label_array = np.array([motor_class])
        group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)
        cv_val_batch_size = 1
        sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False) 
        _, class_predictions_array, labels_array, probabilities_array, _, _ = eval_model_confusion_matrix_fea(model, sub_cv_val_loader, device)
        #print(class_predictions_array)
        probabilities_array = softmax(probabilities_array.reshape((1,-1)))
        probabilities_label = probabilities_array[0, labels_array[0]]
        save_results_online(class_predictions_array, labels_array, result_save_subject_resultanalysisdir)

        return class_predictions_array, probabilities_label
    
    if order == 2:
        # order 2: update the MI recognition model online
        # combine the datasets
        combined_feature_array = np.concatenate((sub_train_feature_array_offline, sub_val_feature_array_offline), axis=0)
        combined_label_array = np.concatenate((sub_train_label_array_offline, sub_val_label_array_offline), axis=0)
        # combine the data from the previous trials
        train_list, train_label, scores = Online_read_csv(Online_folder_path,session,trial)
        #args_dict.data_online_store.append(train_list)
        #args_dict.label_online_store.append(train_label)
        args_dict.data_online_store = np.concatenate((args_dict.data_online_store, train_list), axis=0)
        args_dict.label_online_store = np.concatenate((args_dict.label_online_store, train_label), axis=0)

        data_online_store = args_dict.data_online_store
        label_online_store = args_dict.label_online_store
        
        combined_feature_array = np.concatenate((combined_feature_array, data_online_store.reshape(-1, data_online_store.shape[-2], data_online_store.shape[-1])), axis=0)
        combined_label_array = np.concatenate((combined_label_array, label_online_store.reshape(-1)), axis=0).astype(np.int64)
        print("training dataset volume: {}".format(combined_feature_array.shape[0]))
        #for trial_idx in range(1, trial+1):
        #    train_list, train_label, scores = Online_read_csv(Online_folder_path,session,trial_idx)
        #    combined_feature_array = np.concatenate((combined_feature_array, train_list), axis=0)
        #    combined_label_array = np.concatenate((combined_label_array, train_label), axis=0)
        
        # choose the trial_pre samples as the new training set 
        unique_labels = np.unique(combined_label_array)
        sub_train_feature_update = []
        sub_train_label_update = []
        for label in unique_labels:
            indices = np.where(combined_label_array == label)[0]
            selected_indices = indices[-trial_pre:]
            sub_train_feature_update.append(combined_feature_array[selected_indices])
            sub_train_label_update.append(combined_label_array[selected_indices])
        sub_train_feature_update = np.concatenate(sub_train_feature_update, axis=0)
        sub_train_label_update = np.concatenate(sub_train_label_update, axis=0)
        
        if (trial + 1) % 5 == 0:
            # Split the updated training set into source and target sets
            sub_train_feature_update_source = []
            sub_train_label_update_source = []
            sub_train_feature_update_target = []
            sub_train_label_update_target = []
            for label in unique_labels:
                indices = np.where(sub_train_label_update == label)[0]
                target_indices = indices[-trial_pre_target:]
                source_indices = list(set(indices) - set(target_indices))
                sub_train_feature_update_target.append(sub_train_feature_update[target_indices])
                sub_train_label_update_target.append(sub_train_label_update[target_indices])
                sub_train_feature_update_source.append(sub_train_feature_update[source_indices])
                sub_train_label_update_source.append(sub_train_label_update[source_indices])
            sub_train_feature_update_source = np.concatenate(sub_train_feature_update_source, axis=0)
            sub_train_label_update_source = np.concatenate(sub_train_label_update_source, axis=0)
            sub_train_feature_update_target = np.concatenate(sub_train_feature_update_target, axis=0)
            sub_train_label_update_target = np.concatenate(sub_train_label_update_target, axis=0)
            
            # Form the new training sets
            source_train_set = brain_dataset(sub_train_feature_update_source, sub_train_label_update_source)
            target_train_set = brain_dataset(sub_train_feature_update_target, sub_train_label_update_target)
            source_train_loader = torch.utils.data.DataLoader(source_train_set, batch_size=batch_size_online, shuffle=True)
            target_train_loader = torch.utils.data.DataLoader(target_train_set, batch_size=batch_size_online, shuffle=True)
            criterion = nn.CrossEntropyLoss(reduction='none')
            #_n_epoch_online = 8
        else:
            # form the new training set
            sub_trian_set = brain_dataset(sub_train_feature_update, sub_train_label_update)
            sub_cv_train_loader = torch.utils.data.DataLoader(sub_trian_set, batch_size=batch_size_online, shuffle=True)
            criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
        # print(experiment_name)
        

        #create criterion and optimizer
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        #training loop
        best_val_accuracy = 0.0
        is_best = False

        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_validation_accuracy = []
        
        result_save_dict = dict()

        _n_epoch_online = int(2 * np.exp((trial-5)/25))

        for epoch in trange(_n_epoch_online, desc='1-fold cross validation'):
            if (trial + 1) % 5 == 0:
                accuracy_per_class_iter = calculate_accuracy_per_class_online(result_save_subject_resultanalysisdir, best_validation_class_accuracy)
                #print(accuracy_per_class_iter)
                average_loss_this_epoch = train_one_epoch_MMD_Weights(model, optimizer, criterion, source_train_loader, target_train_loader, device, accuracy_per_class_iter)
                #average_loss_this_epoch = train_one_epoch_MMD(model, optimizer, criterion, source_train_loader, target_train_loader, device)
                val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, target_train_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, target_train_loader, device)
            else:
                average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, sub_cv_train_loader, device)
                val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_cv_train_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, sub_cv_train_loader, device)

            epoch_train_loss.append(average_loss_this_epoch)
            epoch_train_accuracy.append(train_accuracy)
            epoch_validation_accuracy.append(val_accuracy)

            #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
            if (trial + 1) % 5 == 0:
                if accuracy_per_class[1] > 0.33 or accuracy_per_class[2] > 0.33:
                    is_best = val_accuracy >= best_val_accuracy
            else:
                is_best = val_accuracy >= best_val_accuracy
            
            if is_best:
                best_val_accuracy = val_accuracy

                #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.statedict'))
                if use_pretrain:
                    encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder.pt'))
                    encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))
                else:
                    encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'best_model_encoder_output.pt'))

                result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                for cls_i in range(accuracy_per_class.shape[0]):
                    result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]

        #save training curve 
        save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

        #save the model at last epoch
        #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
        if use_pretrain:
            encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
            encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
        else:
            encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
        
        #save result_save_dict
        save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
        
        #write performance to txt file
        Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
        end_time = time.time()
        total_time = end_time - start_time
        write_program_time(os.path.join(Online_result_save_rootdir, sub_name_online), total_time)
    

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
