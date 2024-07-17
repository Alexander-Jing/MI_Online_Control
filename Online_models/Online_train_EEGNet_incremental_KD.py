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

from helpers.models import EEGNetFea, ConvEncoderResBN, ConvEncoderClsFea, ResEncoderfinetune, ConvEncoder3_ClsFeaTL, ConvEncoder3ResBN, ConvEncoder_OutputClsFeaTL
from helpers.brain_data import Offline_read_csv, brain_dataset, Online_read_csv, Online_read_csv_selection, Online_read_csv_selection_session
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, \
    train_one_epoch, train_update, eval_model, eval_model_confusion_matrix_fea, save_training_curves_FixedTrainValSplit, \
        write_performance_info_FixedTrainValSplit, write_program_time, train_one_epoch_MMD_Weights, train_one_epoch_MMD, eval_model_fea, train_one_epoch_fea, \
            compute_total_accuracy_per_class, softmax, save_results_online, calculate_accuracy_per_class_online, load_best_validation_class_accuracy_offline,\
            load_best_validation_path_offline, eval_model_fea_exemplars_distillation_datafea_logitlabel, MultiClassFocalLoss, train_one_epoch_logitlabel_distillation, train_one_epoch_fea_distillation,\
            accuracy_save2csv, accuracy_iteration_plot, accuracy_perclass_save2csv, accuracy_perclass_iteration_plot
        
from helpers.utils import Offline_write_performance_info_FixedTrainValSplit, Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix

#for personal model, save the test prediction of each cv fold
def Online_train_classifierEEGNet_incremental_KD(args_dict):
    
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
    samples_online = args_dict.samples_online
    best_validation_path = args_dict.best_validation_path
    unfreeze_encoder_offline = args_dict.unfreeze_encoder_offline
    unfreeze_encoder_online = args_dict.unfreeze_encoder_online
    use_pretrain = args_dict.use_pretrain
    trial_pre = args_dict.trial_pre
    trial_pre_target = args_dict.trial_pre_target
    window = args_dict.winodw
    restore_path_encoder = args_dict.restore_path_encoder
    device = args_dict.device
    lr = args_dict.lr
    dropout = args_dict.dropout
    best_validation_class_accuracy = args_dict.best_validation_class_accuracy
    sub_train_feature_array_offline = args_dict.sub_train_feature_array_offline
    sub_train_label_array_offline = args_dict.sub_train_label_array_offline
    sub_val_feature_array_offline = args_dict.sub_val_feature_array_offline
    sub_val_label_array_offline = args_dict.sub_val_label_array_offline
    update_trial = args_dict.update_trial
    alpha_distill = args_dict.alpha_distill
    update_wholeModel = args_dict.update_wholeModel
    total_trials = args_dict.total_trials
    data_preprocessing = args_dict.data_preprocessing 
    channel_list = args_dict.channel_list
    target_channel_list = args_dict.target_channel_list
    session_manual = args_dict.session_manual
    session_manual_id = args_dict.session_manual_id

    # orders
    if order == 1:
        print('*********Online Predicting***********')
    if order == 2:
        print('*********Online Updating***********')
    
    #create model
    if data_preprocessing:
        model = EEGNetFea(feature_size=len(target_channel_list)+1, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
    else:
        model = EEGNetFea(feature_size=len(channel_list), num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=dropout)
 
    restore_file = best_validation_path
    
    #reload weights from restore_file is specified
    if restore_file != 'None':
        # load the model only in the first trial or use the session_manual (select a session manually) 
        if trial==1:  
            model.load_state_dict(torch.load(restore_path_encoder))
            model = model.to(device)
            print("initially load model: ", restore_path_encoder) 
        if session_manual and trial%update_wholeModel == 1:
            model.load_state_dict(torch.load(restore_path_encoder))
            model = model.to(device)
            print("initially load model: ", restore_path_encoder)

    model = model.to(device)

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
        # order 1: recognize the MI class online without online updating the model, this is for the online scenario in the real interaction 
        sub_val_feature_array = [data[0:-1,:]]  # take care, we only recorded the 0 to end-1 row of data 
        sub_val_label_array = np.array([motor_class])
        group_val_set = brain_dataset(sub_val_feature_array, sub_val_label_array)
        cv_val_batch_size = 1
        sub_cv_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False) 
        if trial > 1: 
            if not session_manual:
                load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session))
                if not os.path.exists(load_model_path_encoder):
                    load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session-1))
                model.load_state_dict(torch.load(load_model_path_encoder))
                model = model.to(device)
                print("load model parameter: ", load_model_path_encoder)
            if session_manual and trial%update_wholeModel!=1:
                load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session))
                if not os.path.exists(load_model_path_encoder):
                    load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session-1))
                model.load_state_dict(torch.load(load_model_path_encoder))
                model = model.to(device)
                print("load model parameter: ", load_model_path_encoder)

        
        predict_accu, class_predictions_array, labels_array, probabilities_array, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_cv_val_loader, device)
        #print(class_predictions_array)
        probabilities_array = softmax(probabilities_array.reshape((1,-1)))
        probabilities_label = probabilities_array[0, labels_array[0]]
        save_results_online(class_predictions_array, labels_array, result_save_subject_resultanalysisdir)
        if motor_class != 0.0:
            args_dict.accuracies_per_class_iterations.append([motor_class, predict_accu/100])
            args_dict.predict_accuracies.append(predict_accu)
            args_dict.accuracies_per_class.append(accuracy_per_class)
        else:
            args_dict.accuracies_per_class_iterations_Rest.append([motor_class, predict_accu/100])

        return class_predictions_array, probabilities_label, probabilities_array

    if order == 2:
        # order 2: update the MI recognition model online
        # combine the datasets
        combined_feature_array = np.concatenate((sub_train_feature_array_offline, sub_val_feature_array_offline), axis=0)
        combined_label_array = np.concatenate((sub_train_label_array_offline, sub_val_label_array_offline), axis=0)
        # combine the data from the previous trials
        #train_list, train_label, scores = Online_read_csv(Online_folder_path,session,trial)
        train_list, train_label, scores = Online_read_csv_selection(Online_folder_path,session,trial,samples_online)

        #args_dict.data_online_store.append(train_list)
        #args_dict.label_online_store.append(train_label)
        
        if session_manual and trial%update_wholeModel==1:
            session_list, session_label, _ = Online_read_csv_selection_session(Online_folder_path, session_manual_id)
            args_dict.data_online_store = np.concatenate((args_dict.data_online_store, session_list), axis=0)
            args_dict.label_online_store = np.concatenate((args_dict.label_online_store, session_label), axis=0)
            print("manual mode, collect {} samples before session {}, trial {}".format((args_dict.data_online_store).shape[0], session_manual_id, trial))


        # the session based data selection will be added here
        data_online_store = args_dict.data_online_store
        label_online_store = args_dict.label_online_store
        
        #_data_online_store = data_online_store.reshape(-1, data_online_store.shape[-2], data_online_store.shape[-1])
        #_label_online_store = label_online_store.reshape(-1)

        # combine the training datasets with the online existing data
        combined_feature_array = np.concatenate((combined_feature_array, data_online_store), axis=0)
        combined_label_array = np.concatenate((combined_label_array, label_online_store), axis=0).astype(np.int64)
        print("training dataset volume already existed: {}".format(combined_feature_array.shape[0]))
        
        if (trial) % update_wholeModel == 1:
            print("********** Online mean-of-exemplars generation trial: {} ***********".format(trial))
            # load the model
            if trial > 1:
                load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session))
                if not os.path.exists(load_model_path_encoder):
                    load_model_path_encoder = os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session-1))
                model.load_state_dict(torch.load(load_model_path_encoder))
                model = model.to(device)
            
            unique_labels = np.unique(combined_label_array)
            # generate the exemplar of each class, including the 0(rest), 1,2(MI)
            for label in unique_labels:
                # for old classes, generating the exemplars class
                if label == 0.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    Rest_output_data_exemplars, Rest_output_feas_exemplars, Rest_output_logits_exemplars, Rest_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)

                if label == 1.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    MI1_output_data_exemplars, MI1_output_feas_exemplars, MI1_output_logits_exemplars, MI1_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)
                     
                if label == 2.0:
                    indices = np.where(combined_label_array == label)[0]
                    selected_indices_exemplars = indices
                    sub_train_feature_exemplars = combined_feature_array[selected_indices_exemplars]
                    sub_train_label_exemplars = combined_label_array[selected_indices_exemplars]
                    _sub_exemplars = brain_dataset(sub_train_feature_exemplars, sub_train_label_exemplars)
                    sub_exemplars = torch.utils.data.DataLoader(_sub_exemplars, batch_size=sub_train_feature_exemplars.shape[0], shuffle=False)
                    MI2_output_data_exemplars, MI2_output_feas_exemplars, MI2_output_logits_exemplars, MI2_output_label_exemplars = eval_model_fea_exemplars_distillation_datafea_logitlabel(model, sub_exemplars, device, trial_pre)
            
            print("mean-of-exemplars generated trial: {}".format(trial))
            print("mean-of-exemplars generated size: {}".format(len(args_dict.Rest_output_label_exemplars)))
            args_dict.Rest_output_data_exemplars, args_dict.Rest_output_feas_exemplars, args_dict.Rest_output_logits_exemplars, args_dict.Rest_output_label_exemplars = Rest_output_data_exemplars, Rest_output_feas_exemplars, Rest_output_logits_exemplars, Rest_output_label_exemplars
            args_dict.MI1_output_data_exemplars, args_dict.MI1_output_feas_exemplars, args_dict.MI1_output_logits_exemplars, args_dict.MI1_output_label_exemplars = MI1_output_data_exemplars, MI1_output_feas_exemplars, MI1_output_logits_exemplars, MI1_output_label_exemplars
            args_dict.MI2_output_data_exemplars, args_dict.MI2_output_feas_exemplars, args_dict.MI2_output_logits_exemplars, args_dict.MI2_output_label_exemplars = MI2_output_data_exemplars, MI2_output_feas_exemplars, MI2_output_logits_exemplars, MI2_output_label_exemplars
                    
        # set the instance class for updating 
        train_label_now_ = np.unique(train_label[-1])
        train_label_exemplars = train_label_now_%2 + 1  # if label is 1, generate label of 2, else if label is 2, generate label of 1

        old_data_exmeplars = []
        old_feas_exemplars = []
        old_logits_exemplars = []
        old_labels_exemplars = []
        new_feas_exemplars = []
        new_labels_exemplars = []

        if train_label_now_ == 0.0:
            old_data_exmeplars.append(args_dict.MI1_output_data_exemplars)
            old_data_exmeplars.append(args_dict.MI2_output_data_exemplars)
            old_feas_exemplars.append(args_dict.MI1_output_feas_exemplars)
            old_feas_exemplars.append(args_dict.MI2_output_feas_exemplars)
            old_logits_exemplars.append(args_dict.MI1_output_logits_exemplars)
            old_logits_exemplars.append(args_dict.MI2_output_logits_exemplars)
            old_labels_exemplars.append(args_dict.MI1_output_label_exemplars)
            old_labels_exemplars.append(args_dict.MI2_output_label_exemplars)
            new_feas_exemplars.append(args_dict.Rest_output_feas_exemplars)
            new_labels_exemplars.append(args_dict.Rest_output_label_exemplars)
        
        if train_label_now_ == 1.0:
            old_data_exmeplars.append(args_dict.Rest_output_data_exemplars)
            old_data_exmeplars.append(args_dict.MI2_output_data_exemplars)
            old_feas_exemplars.append(args_dict.Rest_output_feas_exemplars)
            old_feas_exemplars.append(args_dict.MI2_output_feas_exemplars)
            old_logits_exemplars.append(args_dict.Rest_output_logits_exemplars)
            old_logits_exemplars.append(args_dict.MI2_output_logits_exemplars)
            old_labels_exemplars.append(args_dict.Rest_output_label_exemplars)
            old_labels_exemplars.append(args_dict.MI2_output_label_exemplars)
            new_feas_exemplars.append(args_dict.MI1_output_feas_exemplars)
            new_labels_exemplars.append(args_dict.MI1_output_label_exemplars)
        
        if train_label_now_ == 2.0:
            old_data_exmeplars.append(args_dict.Rest_output_data_exemplars)
            old_data_exmeplars.append(args_dict.MI1_output_data_exemplars)
            old_feas_exemplars.append(args_dict.Rest_output_feas_exemplars)
            old_feas_exemplars.append(args_dict.MI1_output_feas_exemplars)
            old_logits_exemplars.append(args_dict.Rest_output_logits_exemplars)
            old_logits_exemplars.append(args_dict.MI1_output_logits_exemplars)
            old_labels_exemplars.append(args_dict.Rest_output_label_exemplars)
            old_labels_exemplars.append(args_dict.MI1_output_label_exemplars)
            new_feas_exemplars.append(args_dict.MI2_output_feas_exemplars)
            new_labels_exemplars.append(args_dict.MI2_output_label_exemplars)
            
        # generate the old data loader
        sub_oldclass_data_distill = np.concatenate(old_data_exmeplars, axis=0)        
        sub_oldclass_fea_distill = np.concatenate(old_feas_exemplars, axis=0)
        sub_oldclass_logits_distill = np.concatenate(old_logits_exemplars, axis=0)
        sub_oldclass_labels = np.concatenate(old_labels_exemplars, axis=0)
        sub_oldclass_datafea_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_fea_distill)
        sub_oldclass_datalogits_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_logits_distill)
        sub_oldclass_datalabels_distill = brain_dataset(sub_oldclass_data_distill, sub_oldclass_labels)
        sub_oldclass_datafea_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datafea_distill, batch_size=batch_size_online, shuffle=True)
        sub_oldclass_datalogits_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datalogits_distill, batch_size=batch_size_online, shuffle=True)
        sub_oldclass_datalabels_distill_loader = torch.utils.data.DataLoader(sub_oldclass_datalabels_distill, batch_size=batch_size_online, shuffle=True)
        
        # generate the new data loader
        sub_newclass_fea_distill = np.concatenate(new_feas_exemplars, axis=0)
        sub_newclass_labels_distill = np.concatenate(new_labels_exemplars, axis=0)
        sub_newclass_fealabel_distill = brain_dataset(sub_newclass_fea_distill, sub_newclass_labels_distill)
        sub_newclass_fealabel_distill_loader = torch.utils.data.DataLoader(sub_newclass_fealabel_distill, batch_size=batch_size_online, shuffle=True)

        # update the online stored data
        args_dict.data_online_store = np.concatenate((args_dict.data_online_store, train_list), axis=0)
        args_dict.label_online_store = np.concatenate((args_dict.label_online_store, train_label), axis=0)

        # combine the datasets with the new data
        combined_feature_array = np.concatenate((combined_feature_array, train_list), axis=0)
        combined_label_array = np.concatenate((combined_label_array, train_label), axis=0).astype(np.int64)
        print("training dataset volume with new data: {}".format(combined_feature_array.shape[0]))

        # if time for updating, updating the whole model
        if (trial) % update_wholeModel == 0:
            unique_labels = np.unique(combined_label_array)
            # Split the updated training set into source and target sets
            sub_train_feature_update_source = []
            sub_train_label_update_source = []
            sub_train_feature_update_target = []
            sub_train_label_update_target = []
            focalloss_alpha = []  # preparing for the focalloss alpha
            if update_wholeModel!=8:
                _update_wholeModel = 8
            else:
                _update_wholeModel = update_wholeModel

            for label in unique_labels:
                indices = np.where(combined_label_array == label)[0]
                
                if label != 0:
                    #print("MI label: ", label)
                    target_indices = indices[-int(_update_wholeModel/2)*samples_online:]
                else:
                    #print("Rest label: ", label)
                    # choose int(_update_wholeModel/2)*batch_size_online samples from the newest data
                    target_indices = indices[-int(_update_wholeModel/2)*samples_online:]
                print("label: {}".format(label))
                print("val set size: {}".format(target_indices.shape))

                source_indices = list(set(indices) - set(target_indices))
                print("train set size: {}".format(len(source_indices)))

                sub_train_feature_update_target.append(combined_feature_array[target_indices])
                sub_train_label_update_target.append(combined_label_array[target_indices])
                sub_train_feature_update_source.append(combined_feature_array[source_indices])
                sub_train_label_update_source.append(combined_label_array[source_indices])
            sub_train_feature_update_source = np.concatenate(sub_train_feature_update_source, axis=0)
            sub_train_label_update_source = np.concatenate(sub_train_label_update_source, axis=0)
            sub_train_feature_update_target = np.concatenate(sub_train_feature_update_target, axis=0)
            sub_train_label_update_target = np.concatenate(sub_train_label_update_target, axis=0)
            print("training the whole model")
            print("train size: {}".format(sub_train_feature_update_source.shape))
            print("val size: {}".format(sub_train_feature_update_target.shape))

            # Form the new training sets
            source_train_set = brain_dataset(sub_train_feature_update_source, sub_train_label_update_source)
            target_train_set = brain_dataset(sub_train_feature_update_target, sub_train_label_update_target)
            source_train_loader = torch.utils.data.DataLoader(source_train_set, batch_size=batch_size_online, shuffle=True)
            target_train_loader = torch.utils.data.DataLoader(target_train_set, batch_size=batch_size_online, shuffle=True)

        # form the new data training set if updat_trial
        if (trial) % update_trial == 0:
            print("Updating with the new data trial: {}, label{}".format(trial, train_label_now_))
            sub_newdata_data_update = []
            sub_newdata_label_update = []
            indices = np.where(combined_label_array == train_label_now_)   
            #selected_indices = indices[0][-update_trial * samples_online:]    # this part should be changed as using the whole [-trial_pre:] data for training 
            selected_indices = indices[0][-trial_pre:]  # using the method of collecting [-trial_pre:] data for optimization
            sub_newdata_data_update.append(combined_feature_array[selected_indices])
            sub_newdata_label_update.append(combined_label_array[selected_indices])
            sub_newdata_data_update = np.concatenate(sub_newdata_data_update, axis=0)
            sub_newdata_label_update = np.concatenate(sub_newdata_label_update, axis=0)
            sub_newdata_datalabel = brain_dataset(sub_newdata_data_update, sub_newdata_label_update)
            sub_newdata_datalabel_loader = torch.utils.data.DataLoader(sub_newdata_datalabel, batch_size=batch_size_online, shuffle=True)
            print("New dataset size: {}".format(sub_newdata_label_update.shape))
        # the loss function
        criterion = nn.CrossEntropyLoss()

        # form the online test set 
        #sub_train_feature_batches = data_online_store.reshape(-1, data_online_store.shape[-2], data_online_store.shape[-1])
        #sub_train_label_batches = label_online_store.reshape(-1)
        #_sub_updating_predict = brain_dataset(sub_train_feature_batches, sub_train_label_batches)
        #sub_updating_predict = torch.utils.data.DataLoader(_sub_updating_predict, batch_size=sub_train_feature_batches.shape[0], shuffle=False)

        if (trial) % update_trial == 0:           
            print("******* Updating the model trial: {} ************".format(trial))
            start_time = time.time()
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting
            # print(experiment_name)
            #derived arg
            result_save_subjectdir = os.path.join(Online_result_save_rootdir, sub_name_online, experiment_name)
            result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')

            makedir_if_not_exist(result_save_subjectdir)
            makedir_if_not_exist(result_save_subject_checkpointdir)

            if (trial) % (update_trial) == 0: 
                accuracy_per_class_iter = compute_total_accuracy_per_class(args_dict.accuracies_per_class_iterations)
                args_dict.accuracy_per_class_iters.append(accuracy_per_class_iter)
                print(accuracy_per_class_iter)
                accuracy_per_class_iter_Rest = compute_total_accuracy_per_class(args_dict.accuracies_per_class_iterations_Rest)
            
                #training loop
                # set the best validation accuracy(val_2)
                if train_label_now_[0] != 0:
                    best_val_accuracy = 0.8 * 100 * accuracy_per_class_iter[int(train_label_now_[0])]
                    best_train_accuracy = 0.8 * 100 * (accuracy_per_class_iter[int(train_label_exemplars[0])] + accuracy_per_class_iter_Rest[0]\
                                                                    )/2
                else:
                    best_val_accuracy = 0.8 * 100 * accuracy_per_class_iter_Rest[0]
                    best_train_accuracy = 0.8 * 100 * (accuracy_per_class_iter[1] + accuracy_per_class_iter[2]\
                                                                    )/2

            is_best = False

            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_accuracy = []
            
            result_save_dict = dict()

            """
            if (trial_idx + 1) % 6 == 0:
                _n_epoch_online = n_epoch_online
            else:
                _n_epoch_online = int(2 * np.exp((trial_idx-5)/25))
            """
            
            _n_epoch_online = n_epoch_online
            
            """criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("feature updating")
            for epoch in trange(_n_epoch_online, desc='online feature distillation'):
                
                average_loss_this_epoch = train_one_epoch_fea_distillation(model, optimizer, criterion, sub_oldclass_datafea_distill_loader, sub_newclass_fealabel_distill_loader, sub_newdata_datalabel_loader, device, alpha=0.5)
                #val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, target_train_loader, device)
                #train_accuracy, _, _ , _ = eval_model_fea(model, target_train_loader, device)
            """

            criterion = nn.CrossEntropyLoss()
            #optimizer = torch.optim.Adam(model.encoder_output.encoder.Encoder_Cls.parameters(), lr=lr)  # only update the cls part
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("cls updating")
            
            for epoch in trange(_n_epoch_online, desc='online classification update'):
                
                #average_loss_this_epoch = train_one_epoch_label_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalabels_distill_loader, device, T=2.0, alpha=alpha_distill)
                average_loss_this_epoch = train_one_epoch_logitlabel_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalogits_distill_loader, sub_oldclass_datalabels_distill_loader, device, T=2.0, alpha=alpha_distill)
                #average_loss_this_epoch = train_one_epoch_logit_distillation(model, optimizer, criterion, sub_newdata_datalabel_loader, sub_oldclass_datalogits_distill_loader, device, T=2, alpha=alpha_distill)
                val_accuracy, _, _ , _ = eval_model_fea(model, sub_newdata_datalabel_loader, device)
                #val_accuracy, _, _, _, _, accuracy_per_class = eval_model_confusion_matrix_fea(model, sub_newdata_datalabel_loader, device)
                train_accuracy, _, _ , _ = eval_model_fea(model, sub_oldclass_datalabels_distill_loader, device)
                
                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag, only when the accuracies of two classes of motor imagery are larger than random choice
                is_best = (val_accuracy >= best_val_accuracy) and (train_accuracy >= best_train_accuracy)
                #is_best = (train_accuracy >= best_train_accuracy)
                #is_best = (train_accuracy>=50*(accuracy_per_class_iter[int(train_label_exemplars[0])] + accuracy_per_class_iter[0]\
                #                                                            )/2) and (val_accuracy>=50*accuracy_per_class_iter[int(train_label_now_[0])])
                
                if is_best:
                    print("best_val_accuracy: {}".format(val_accuracy))
                    print("best_train_accuracy: {}".format(train_accuracy))
                    best_val_accuracy = val_accuracy
                    best_train_accuracy = train_accuracy

                    torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session)))

                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy
                    """for cls_i in range(accuracy_per_class.shape[0]):
                        result_save_dict['class_accuracy_' + str(cls_i)] = accuracy_per_class[cls_i]"""
                
                #if (best_val_accuracy >= 80.0) and (best_train_accuracy >= 80.0):
                #    break

            # updating the whole model
            if (trial) % update_wholeModel == 0:
                print("******* Updating the whole model trial: {} ************".format(trial))
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                whole_model_is_best = False
                whole_model_best_val_accuracy = 0
                
                _n_epoch_online = n_epoch_online * 2
                
                for epoch in trange(_n_epoch_online, desc='online classification update whole model'):
                    average_loss_this_epoch = train_one_epoch_fea(model, optimizer, criterion, source_train_loader, device)
                    whole_model_val_accuracy, _, _, _, _, whole_model_accuracy_per_class = eval_model_confusion_matrix_fea(model, target_train_loader, device)
                    whole_model_train_accuracy, _, _ , _ = eval_model_fea(model, source_train_loader, device)
                    
                    epoch_train_loss.append(average_loss_this_epoch)
                    epoch_train_accuracy.append(whole_model_train_accuracy)
                    epoch_validation_accuracy.append(whole_model_val_accuracy)

                    whole_model_is_best = (whole_model_val_accuracy >= whole_model_best_val_accuracy)
                    if whole_model_is_best:
                        print("whole model best_val_accuracy: {}".format(whole_model_val_accuracy))
                        #print("whole model best_train_accuracy: {}".format(whole_model_train_accuracy))
                        whole_model_best_val_accuracy = whole_model_val_accuracy
                        #best_train_accuracy = train_accuracy

                        #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.statedict'))
                        torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model_{}.pt'.format(session)))
                        
                        result_save_dict['bestepoch_val_accuracy'] = whole_model_val_accuracy
            
            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #save the model at last epoch
            #torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))
            #encoder_to_use.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder.pt'))
            #encoder_to_use_output.save(os.path.join(result_save_subject_checkpointdir, 'last_model_encoder_output.pt'))
            
            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            Offline_write_performance_info_FixedTrainValSplit_ConfusionMatrix(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict)
            end_time = time.time()
            total_time = end_time - start_time
            write_program_time(os.path.join(Online_result_save_rootdir, sub_name_online), total_time)

            if trial == total_trials:
                accuracy_save2csv(args_dict.predict_accuracies, result_save_subjectdir)
                accuracy_iteration_plot(args_dict.predict_accuracies, result_save_subjectdir)
                accuracy_perclass_save2csv(args_dict.accuracy_per_class_iters, result_save_subjectdir)
                accuracy_perclass_iteration_plot(args_dict.accuracy_per_class_iters, result_save_subjectdir)


    

    

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
