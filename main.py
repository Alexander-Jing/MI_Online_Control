import socket
import scipy.io
import os
import numpy as np
import json
from progress.bar import Bar
import math
import csv
from helpers.ServerDataProcessing import NpEncoder
from tqdm import tqdm
from easydict import EasyDict as edict
import torch

import argparse
import time
import shutil
import re

from helpers import models
from helpers.brain_data import Offline_read_csv, preprocess_eeg_data_online
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model,\
     save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time, str2bool, save_best_validation_class_accuracy_offline,\
     load_best_validation_class_accuracy_offline, load_best_validation_path_offline
from Offline_models.Offline_train_EEGNet import Offline_train_classifierEEGNet
from Online_models.Online_train_EEGNet import Online_train_classifier
from Offline_models.Offline_train_EEGLM import Offline_train_classifierLM
from Online_models.Online_train_EEGLM import Online_train_classifierLM
from Online_models.Online_train_EEGLM_incremental import Online_train_classifierLM_incremental
from Online_models.Online_train_EEGLM_incremental_KD import Online_train_classifierLM_incremental_KD
from Online_models.Online_train_EEGNet_incremental_KD import Online_train_classifierEEGNet_incremental_KD
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch, synthesize_hypersearch_confusionMatrix

from Online_tests.SeverControlOnline import SeverControlOnlineTest

def SeverControlOnline(args_dict):
    """
    # 在线和matlab进行实时交互的服务端程序
    参数：
    sub_name: 被试名称
    config_length: 发送数据config_data的长度, 设置为(config_data数据的长度+2)
    ip: 服务器ip地址
    port: 监听端口号
    
    """
    # 确定相关参数
    ip = args_dict.ip
    port = args_dict.port
    sub_name_online = args_dict.sub_name_online
    config_length = 13
    
    addr = (ip, port) #设置服务端ip地址和端口号
    buff_size = int(65535)         #消息的最大长度
    tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    tcpSerSock.bind(addr)
    tcpSerSock.listen(1)
    tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buff_size)
    #tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buff_size)
    while True:
        print('等待连接...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('***********New Manipulation**********')
        print('连接到:', addr)

        while True:
            decode = []                                   #存放解码后的数据
            recv_data = []
            #只接受第一个数据包的数据，提取里面的数据量值
            while not recv_data:
                recv_data = tcpCliSock.recv(int(buff_size))  # 这里是按照字节读取的，len(recv_data)返回的是字节数
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            #如果没有接收完大小为data_bytes[0]的数据，则一直接收
            pbar = tqdm(total=data_bytes[0])
            while len(recv_data) < data_bytes[0]:
                data = []
                while not data:
                    data = tcpCliSock.recv(int(buff_size))
                recv_data += data              #数据拼接
                pbar.update(len(data))
            pbar.close()
            
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            
            decode = np.frombuffer(bytes(recv_data[0:(int(data_bytes)+8)]),dtype='>f4')     #数据解码,注意这里要多加4个字节，data_bytes[0]是传输数据部分的大小，还要加上第一个4字节的变量，但是这里莫名其妙多加了4个字节，暂时没有查明是什么原因                
            
            print('接收到数据字节数：', (data_bytes[0]) )  # 接收到的数据，其中每一个变量都是float32的形式，按照发送的格式，其中第0个float32是整个数据的大小（总共多少个float32的数据格式），第1个是命令，用于控制服务器，后面的第2和第3个float32数据分别是矩阵的宽和高，再后面第4个float32数据起是实际数据
            print('接收到数据量：',(data_bytes[0]/4-config_length))
            
            order = int(decode[1])  # 接收的命令
            if order == 1:  # 命令1，用于对于输入的数据进行实时分类
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                echo = decode[config_length:]
                mat = echo.reshape((channels, window_length), order='F')  # 收到数据，用于输入模型
                print("received data shape: ", mat.shape)
                if args_dict.data_preprocessing:
                    mat = preprocess_eeg_data_online(mat,channel_selection=True, \
                                                    channel_list=args_dict.channel_list, target_channel_list=args_dict.target_channel_list, \
                                                    max_scale=128)
                print("preprocessed data shape: ", mat.shape)
                scores = np.full((1,window_length), score)  
                data = np.vstack((mat,scores))  # 这部分用于存储数据以及后续的训练
                
                # 对数据进行存储,这个文件夹存储的是在被试的每一个session中的每一个trial中的所有数据和score
                save_folder = args_dict.Online_folder_path
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    print(save_folder + '  created')
                
                sub_save_path = '/' + 'class_' + str(motor_class) + '_session_' + str(session) + '_trial_' + str(trial) + '_window_' + str(window) + '_score_' + format(score, '.2f') + '.csv'
                with open(save_folder + sub_save_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                    print(save_folder + sub_save_path + '  saved')

                # 准备好相关参数
                args_dict.order = order
                args_dict.motor_class = motor_class
                args_dict.data = data
                args_dict.session = session
                args_dict.trial = trial
                args_dict.winodw = window

                # 模型计算出来的结果准备发送回matlab
                #predict = Online_train_classifier(args_dict)
                #predict, probability = Online_train_classifierLM(args_dict)
                #predict, probability = Online_train_classifierLM_incremental(args_dict)
                predict, probability = Online_train_classifierLM_incremental_KD(args_dict)
                # 在字典里增加一个L来保存返回数据包的大小
                result={'L':2e8,  # 以字典的形式发送数据
                        'R':[predict, probability],
                        }
                send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
                result['L'] = len(send_result) #把数据包真实大小写进字典
                #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
                matlab_buffer = int(8388608/256)
                fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
                send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
                print('需要发回%d字节的数据包'%len(send_result))
                tcpCliSock.sendto(send_result,addr) #数据发送
                print('发送完成\n')
            
            if order == 4:  # 命令4，用于对于离线的数据进行分类分析
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                echo = decode[config_length:]
                mat = echo.reshape((channels, window_length), order='F')  # 收到数据，用于输入模型
                print("received data shape: ", mat.shape)
                if args_dict.data_preprocessing:
                    mat = preprocess_eeg_data_online(mat,channel_selection=True, \
                                                    channel_list=args_dict.channel_list, target_channel_list=args_dict.target_channel_list, \
                                                    max_scale=128)
                print("preprocessed data shape: ", mat.shape)
                scores = np.full((1,window_length), score)  
                data = np.vstack((mat,scores))  # 这部分用于存储数据以及后续的训练
                
                # 对数据进行存储,这个文件夹存储的是在被试的每一个session中的每一个trial中的所有数据和score
                save_folder = args_dict.Online_folder_path
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    print(save_folder + '  created')
                
                sub_save_path = '/' + 'class_' + str(motor_class) + '_session_' + str(session) + '_trial_' + str(trial) + '_window_' + str(window) + '_score_' + format(score, '.2f') + '.csv'
                with open(save_folder + sub_save_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                    print(save_folder + sub_save_path + '  saved')

                # 准备好相关参数
                args_dict.order = order
                args_dict.motor_class = motor_class
                args_dict.data = data
                args_dict.session = session
                args_dict.trial = trial
                args_dict.winodw = window

                # 模型计算出来的结果准备发送回matlab
                #predict = Online_train_classifier(args_dict)
                #predict, probability = Online_train_classifierLM(args_dict)
                #predict, probability = Online_train_classifierLM_incremental(args_dict)
                predict, probability = Online_train_classifierLM_incremental_KD(args_dict)
                # 在字典里增加一个L来保存返回数据包的大小
                result={'L':2e8,  # 以字典的形式发送数据
                        'R':[predict, probability],
                        }
                send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
                result['L'] = len(send_result) #把数据包真实大小写进字典
                #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
                matlab_buffer = int(8388608/256)
                fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
                send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
                print('需要发回%d字节的数据包'%len(send_result))
                tcpCliSock.sendto(send_result,addr) #数据发送
                print('发送完成\n')

            if order == 2:  # 命令2，使用收集的数据来更新模型
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                # 准备模型训练
                args_dict.session = session
                args_dict.trial = trial
                #args_dict.n_epoch = 4
                args_dict.order = order
                args_dict.winodw = window

                makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint'))
                # 更新模型
                #Online_train_classifier(args_dict)
                #Online_train_classifierLM(args_dict)
                #Online_train_classifierLM_incremental(args_dict)
                Online_train_classifierLM_incremental_KD(args_dict)
                print('session: ' + str(session) + ', ' + 'trial: ' + str(trial) + ' model updated\n' )
            break
        tcpCliSock.close()
    tcpSerSock.close()

def SeverControlOnlineSelection(args_dict):
    """
    # 在线和matlab进行实时交互的服务端程序
    参数：
    sub_name: 被试名称
    config_length: 发送数据config_data的长度, 设置为(config_data数据的长度+2)
    ip: 服务器ip地址
    port: 监听端口号
    
    """
    # 确定相关参数
    ip = args_dict.ip
    port = args_dict.port
    sub_name_online = args_dict.sub_name_online
    mode = args_dict.mode
    config_length = 13
    
    addr = (ip, port) #设置服务端ip地址和端口号
    buff_size = int(65535)         #消息的最大长度
    tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    tcpSerSock.bind(addr)
    tcpSerSock.listen(1)
    tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buff_size)
    #tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buff_size)
    while True:
        print('等待连接...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('***********New Manipulation**********')
        print('连接到:', addr)

        while True:
            decode = []                                   #存放解码后的数据
            recv_data = []
            #只接受第一个数据包的数据，提取里面的数据量值
            while not recv_data:
                recv_data = tcpCliSock.recv(int(buff_size))  # 这里是按照字节读取的，len(recv_data)返回的是字节数
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            #如果没有接收完大小为data_bytes[0]的数据，则一直接收
            pbar = tqdm(total=data_bytes[0])
            while len(recv_data) < data_bytes[0]:
                data = []
                while not data:
                    data = tcpCliSock.recv(int(buff_size))
                recv_data += data              #数据拼接
                pbar.update(len(data))
            pbar.close()
            
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            
            decode = np.frombuffer(bytes(recv_data[0:(int(data_bytes)+8)]),dtype='>f4')     #数据解码,注意这里要多加4个字节，data_bytes[0]是传输数据部分的大小，还要加上第一个4字节的变量，但是这里莫名其妙多加了4个字节，暂时没有查明是什么原因                
            
            print('接收到数据字节数：', (data_bytes[0]) )  # 接收到的数据，其中每一个变量都是float32的形式，按照发送的格式，其中第0个float32是整个数据的大小（总共多少个float32的数据格式），第1个是命令，用于控制服务器，后面的第2和第3个float32数据分别是矩阵的宽和高，再后面第4个float32数据起是实际数据
            print('接收到数据量：',(data_bytes[0]/4-config_length))
            
            order = int(decode[1])  # 接收的命令
            if order == 1:  # 命令1，用于对于输入的数据进行实时分类
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                echo = decode[config_length:]
                mat = echo.reshape((channels, window_length), order='F')  # 收到数据，用于输入模型
                print("received data shape: ", mat.shape)
                if args_dict.data_preprocessing:
                    mat = np.expand_dims(mat,axis=0)
                    mat = preprocess_eeg_data_online(mat,channel_selection=True, \
                                                    channel_list=args_dict.channel_list, target_channel_list=args_dict.target_channel_list, \
                                                    max_scale=128)
                    mat = mat[0]
                print("preprocessed data shape: ", mat.shape)
                scores = np.full((1,window_length), score)  
                data = np.vstack((mat,scores))  # 这部分用于存储数据以及后续的训练，注意，这里把scores单独作为一行也存进去了，实际数据不包括最后一行
                
                # 准备好相关参数
                args_dict.order = order
                args_dict.motor_class = motor_class
                args_dict.data = data
                args_dict.session = session
                args_dict.trial = trial
                args_dict.winodw = window

                # 模型计算出来的结果准备发送回matlab
                #predict = Online_train_classifier(args_dict)
                #predict, probability = Online_train_classifierLM(args_dict)
                #predict, probability = Online_train_classifierLM_incremental(args_dict)
                if mode == 'Online EEGLM':
                    predict, probability = Online_train_classifierLM_incremental_KD(args_dict)
                elif mode == 'Online EEGNet':
                    predict, probability, probabilities_array = Online_train_classifierEEGNet_incremental_KD(args_dict)
                    print('predict prob: ', probabilities_array)
                
                # 对数据进行存储,这个文件夹存储的是在被试的每一个session中的每一个trial中的所有数据和score
                save_folder = args_dict.Online_folder_path
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    print(save_folder + '  created')
                
                sub_save_path = '/' + 'class_' + str(motor_class) + '_session_' + str(session) + '_trial_' + str(trial) + '_window_' + str(window) + '_score_' + format(probability, '.2f') + '.csv'
                with open(save_folder + sub_save_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                    print(save_folder + sub_save_path + '  saved')
                
                # 在字典里增加一个L来保存返回数据包的大小
                if mode == 'Online EEGLM':
                    result={'L':2e8,  # 以字典的形式发送数据
                            'R':[predict, probability],
                            }
                elif mode == 'Online EEGNet':
                    result={'L':2e8,  # 以字典的形式发送数据
                            'R':[predict, probabilities_array[0,0], probabilities_array[0,1], probabilities_array[0,2]],
                            }
                send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
                result['L'] = len(send_result) #把数据包真实大小写进字典
                #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
                matlab_buffer = int(8388608/256)
                fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
                send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
                print('需要发回%d字节的数据包'%len(send_result))
                tcpCliSock.sendto(send_result,addr) #数据发送
                print('发送完成\n')
            
            if order == 4:  # 命令4，用于对于离线的数据进行分类分析
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                echo = decode[config_length:]
                mat = echo.reshape((channels, window_length), order='F')  # 收到数据，用于输入模型
                print("received data shape: ", mat.shape)
                if args_dict.data_preprocessing:
                    mat = np.expand_dims(mat,axis=0)
                    mat = preprocess_eeg_data_online(mat,channel_selection=True, \
                                                    channel_list=args_dict.channel_list, target_channel_list=args_dict.target_channel_list, \
                                                    max_scale=128)
                    mat = mat[0]
                print("preprocessed data shape: ", mat.shape)
                scores = np.full((1,window_length), score)  
                data = np.vstack((mat,scores))  # 这部分用于存储数据以及后续的训练,注意，这里把scores单独作为一行也存进去了，实际数据不包括最后一行

                # 对数据进行存储,这个文件夹存储的是在被试的每一个session中的每一个trial中的所有数据和score
                save_folder = args_dict.Online_folder_path
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    print(save_folder + '  created')
                
                sub_save_path = '/' + 'class_' + str(motor_class) + '_session_' + str(session) + '_trial_' + str(trial) + '_window_' + str(window) + '_score_' + format(score, '.2f') + '.csv'
                with open(save_folder + sub_save_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                    print(save_folder + sub_save_path + '  saved')
                
                # 准备好相关参数
                args_dict.order = order
                args_dict.motor_class = motor_class
                args_dict.data = data
                args_dict.session = session
                args_dict.trial = trial
                args_dict.winodw = window

                # 模型计算出来的结果准备发送回matlab
                #predict = Online_train_classifier(args_dict)
                #predict, probability = Online_train_classifierLM(args_dict)
                #predict, probability = Online_train_classifierLM_incremental(args_dict)
                predict, probability = Online_train_classifierLM_incremental_KD(args_dict)
                # 在字典里增加一个L来保存返回数据包的大小
                result={'L':2e8,  # 以字典的形式发送数据
                        'R':[predict, probability],
                        }
                send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
                result['L'] = len(send_result) #把数据包真实大小写进字典
                #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
                matlab_buffer = int(8388608/256)
                fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
                send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
                print('需要发回%d字节的数据包'%len(send_result))
                tcpCliSock.sendto(send_result,addr) #数据发送
                print('发送完成\n')

            if order == 2:  # 命令2，使用收集的数据来更新模型
                # 对于数据的 config_data进行解码
                window_length = int(decode[2])
                channels = int(decode[3])
                motor_class = int(decode[4])
                session = int(decode[5])
                trial = int(decode[6])
                window = int(decode[7])
                score = float(decode[8])
                # 准备模型训练
                args_dict.session = session
                args_dict.trial = trial
                #args_dict.n_epoch = 4
                args_dict.order = order
                args_dict.winodw = window

                makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint'))
                # 更新模型
                #Online_train_classifier(args_dict)
                #Online_train_classifierLM(args_dict)
                #Online_train_classifierLM_incremental(args_dict)
                #Online_train_classifierLM_incremental_KD(args_dict)
                if mode == 'Online EEGLM':
                    Online_train_classifierLM_incremental_KD(args_dict)
                elif mode == 'Online EEGNet':
                    Online_train_classifierEEGNet_incremental_KD(args_dict)
                print('session: ' + str(session) + ', ' + 'trial: ' + str(trial) + ' model updated\n' )
            break
        tcpCliSock.close()
    tcpSerSock.close()


def SeverControlOffline(args_dict):
    
    # 确定相关参数
    ip = args_dict.ip
    port = args_dict.port
    sub_name_offline = args_dict.sub_name_offline
    mode = args_dict.mode
    
    addr = (ip, port) #设置服务端ip地址和端口号
    buff_size = 65535         #消息的最大长度
    tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    tcpSerSock.bind(addr)
    tcpSerSock.listen(1)
    tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
    while True:
        print('等待连接...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('连接到:', addr)
        while True:
            decode = []                                   #存放解码后的数据
            recv_data = []
            #只接受第一个数据包的数据，提取里面的数据量值
            while not recv_data:
                recv_data = tcpCliSock.recv(int(buff_size))  # 这里是按照字节读取的，len(recv_data)返回的是字节数
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            #如果没有接收完大小为data_bytes[0]的数据，则一直接收
            pbar = tqdm(total=data_bytes[0])
            while len(recv_data) < data_bytes[0]:
                data = []
                while not data:
                    data = tcpCliSock.recv(int(buff_size))
                recv_data += data              #数据拼接
                pbar.update(len(data))
            pbar.close()
            
            data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
            
            decode = np.frombuffer(bytes(recv_data[0:(int(data_bytes)+8)]),dtype='>f4')     #数据解码,注意这里要多加4个字节，data_bytes[0]是传输数据部分的大小，还要加上第一个4字节的变量，但是这里莫名其妙多加了4个字节，暂时没有查明是什么原因                
            
            # print(decode.shape)
            print('接收到数据字节数：', (data_bytes[0]) )  # 接收到的数据，其中每一个变量都是float32的形式，按照发送的格式，其中第0个float32是整个数据的大小（总共多少个float32的数据格式），第1个是命令，用于控制服务器，后面的第2和第3个float32数据分别是矩阵的宽和高，再后面第4个float32数据起是实际数据
            print('接收到数据量：',(data_bytes[0]/4-6))
            
            order = int(decode[1])  # 接收的命令
            
            if order == 3:
                window_length = int(decode[2])
                channels = int(decode[3])
                windows = int(decode[4])
                classes = int(decode[5])
                echo = decode[6:]
                mat = echo.reshape(((classes * windows * channels), window_length), order='F')
                print("received data shape: ", mat.shape)
                
                if not os.path.exists(args_dict.Offline_folder_path):
                    os.makedirs(args_dict.Offline_folder_path)
                    print(args_dict.Offline_folder_path + '  created')
                
                sub_mat = np.split(mat, classes, axis=0)  # 根据类进行分开数据
                for i, sub_class in enumerate(sub_mat):
                    sub_data = np.split(sub_class, windows, axis=0)
                    
                    for j, sub_window in enumerate(sub_data):
                        
                        if args_dict.data_preprocessing:
                            sub_window = np.expand_dims(sub_window,axis=0)
                            sub_window = preprocess_eeg_data_online(sub_window,channel_selection=True, \
                                                                    channel_list=args_dict.channel_list, target_channel_list=args_dict.target_channel_list, \
                                                                    max_scale=128)
                            sub_window = sub_window[0]
                        
                        sub_file_name = 'class_' + str(i) +'_window' + '_' + str(j) + '.csv'
                        
                        with open(os.path.join(args_dict.Offline_folder_path, sub_file_name), 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(sub_window)
                            print(os.path.join(args_dict.Offline_folder_path, sub_file_name) + '  saved')
                
                args_dict.windows_num = len(sub_data)  # 知道每一类的windows划分的数量，用于后续的确认数据集大小，注意这里假定每一类都是相同长度的
                # 离线训练模型
                #Offline_train_classifier(args_dict)
                if mode == 'Offline EEGLM':
                    Offline_train_classifierLM(args_dict)
                if mode == 'Offline EEGNet':
                    Offline_train_classifierEEGNet(args_dict)
                # 搜索离线训练模型超参数
                experiment_dir = os.path.join(args_dict.Offline_result_save_rootdir, args_dict.sub_name_offline)
                summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
                if not os.path.exists(summary_save_dir):
                    os.makedirs(summary_save_dir)    
                best_validation_class_accuracy, best_validation_path = synthesize_hypersearch_confusionMatrix(experiment_dir, summary_save_dir)
                # 存储best_validation_class_accuracy到summary_save_dir文件夹下面best_validation_class_accuracy.csv
                save_best_validation_class_accuracy_offline(best_validation_class_accuracy, summary_save_dir)
                # 在字典里增加一个L来保存返回数据包的大小
                result={'L':2e8,  # 以字典的形式发送数据
                        'R':best_validation_class_accuracy,
                        }
                send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
                result['L'] = len(send_result) #把数据包真实大小写进字典
                #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
                matlab_buffer = 8388608
                fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
                send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
                print('需要发回%d字节的数据包'%len(send_result))
                tcpCliSock.sendto(send_result,addr) #数据发送
                print('发送完成\n')
            break
        tcpCliSock.close()
    tcpSerSock.close() 
    
if __name__ == "__main__":
    
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
    parser.add_argument('--sub_name_offline', default='Jyt_offline', type=str, help='name of the subject in the offline experiment')
    parser.add_argument('--sub_name_online', default='Jyt_online', type=str, help='name of the subject in the online experiment')
    parser.add_argument('--windows_num', default=149, type=int, help='number of windows')
    
    parser.add_argument('--Offline_folder_path', default='./Offline_DataCollected', help="Offline folder to the dataset")
    parser.add_argument('--Offline_result_save_rootdir', default='./Offline_experiments', help="Directory containing the experiment models")
    parser.add_argument('--restore_file', default='None', help="xxx.statedict")
    parser.add_argument('--proportion', default=0.8, type=float, help='proportion of the training set of the whole dataset')
    parser.add_argument('--n_epoch', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_offline', default=100, type=int, help="number of epoch")
    parser.add_argument('--n_epoch_online', default=100, type=int, help="number of epoch")
    parser.add_argument('--batch_size', default=64, type=int, help="number of batch size")
    parser.add_argument('--Online_folder_path', default='./Online_DataCollected', help="Online folder to the dataset")
    parser.add_argument('--Online_result_save_rootdir', default='./Online_experiments', help="Directory containing the experiment models")
    parser.add_argument('--batch_size_online', default=16, type=int, help="number of batch size for online updating")
    parser.add_argument('--samples_online', default=9, type=int, help="number of samples chosen for online updating")
    parser.add_argument('--validation_path_manual', default=False, type=str2bool, help="whether to set the path of the best validation performance model")
    parser.add_argument('--best_validation_path', default='lr0.001_dropout0.5', type=str, help="if validation_path_manual, set path of the best validation performance model")
    parser.add_argument('--unfreeze_encoder_offline', default=True, type=str2bool, help="whether to unfreeze the encoder params during offline training process")
    parser.add_argument('--unfreeze_encoder_online', default=True, type=str2bool, help="whether to unfreeze the encoder params during online training process")
    parser.add_argument('--use_pretrain', default=False, type=str2bool, help="whether to use the pretrain models")
    parser.add_argument('--trial_pre', default=120, type=int, help="the data used to train the model online each class")
    parser.add_argument('--trial_pre_target', default=20, type=int, help="whether to use the pretrain models")
    parser.add_argument('--update_trial', default=1, type=int, help="number of trails for instant updating")
    parser.add_argument('--update_wholeModel', default=15, type=int, help="number of trails for longer updating")
    parser.add_argument('--total_trials', default=64, type=int, help="number of total trails for training")
    parser.add_argument('--alpha_distill', default=0.5, type=float, help="alpha of the distillation and cls loss func")
    parser.add_argument('--para_m', default=0.99, type=float, help="hyper parameter for momentum updating")
    parser.add_argument('--cons_rate', default=0.01, type=float, help="hyper parameter for constractive loss")
    parser.add_argument('--data_preprocessing', default=False, type=str2bool, help="whether to use the data preprocessing method to normalized the data")
    parser.add_argument('--session_manual', default=False, type=str2bool, help="if interrupted by the communication errors, use the session_manual mode")
    parser.add_argument('--session_manual_id', default=0, type=int, help="if interrupted by the communication errors, use the session_manual mode, input the session id,\
                         the whole model online updating will start at the beginning of the id")

    parser.add_argument('--ip', default='172.18.22.21', type=str, help='the IP address')
    parser.add_argument('--port', default=8880, type=int, help='the port')
    parser.add_argument('--mode', default='Online', type=str, help='choice of working mode: Offline or Online')
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    sub_name_offline = args.sub_name_offline
    sub_name_online = args.sub_name_online
    Offline_folder_path  = args.Offline_folder_path
    Online_folder_path = args.Online_folder_path
    windows_num = args.windows_num
    proportion = args.proportion
    Offline_result_save_rootdir = args.Offline_result_save_rootdir
    Online_result_save_rootdir = args.Online_result_save_rootdir
    restore_file = args.restore_file
    n_epoch_offline = args.n_epoch_offline
    n_epoch_online = args.n_epoch_online
    batch_size = args.batch_size
    ip = args.ip
    port = args.port
    mode = args.mode
    batch_size_online = args.batch_size_online
    validation_path_manual = args.validation_path_manual
    best_validation_path = args.best_validation_path
    unfreeze_encoder_offline = args.unfreeze_encoder_offline
    unfreeze_encoder_online = args.unfreeze_encoder_online
    use_pretrain = args.use_pretrain
    trial_pre = args.trial_pre
    trial_pre_target = args.trial_pre_target
    update_trial = args.update_trial
    alpha_distill = args.alpha_distill
    update_wholeModel = args.update_wholeModel
    samples_online = args.samples_online
    total_trials = args.total_trials
    data_preprocessing = args.data_preprocessing
    session_manual = args.session_manual
    session_manual_id = args.session_manual_id
    para_m = args.para_m
    cons_rate = args.cons_rate
    
    #save_folder = './Online_DataCollected' + str(sub_name)
    #sanity check:
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))
    print('sub_name: {}, type: {}'.format(sub_name_online, type(sub_name_online)))
    print('Offline_folder_path: {}, type: {}'.format(os.path.join(Offline_folder_path, sub_name_offline), type(Offline_folder_path)))
    print('Online_folder_path: {}, type: {}'.format(os.path.join(Online_folder_path, sub_name_online), type(Offline_folder_path)))
    print('windows_num: {}, type: {}'.format(windows_num, type(windows_num)))
    print('proportion: {}, type: {}'.format(proportion, type(proportion)))
    print('Offline_result_save_rootdir: {}, type: {}'.format(Offline_result_save_rootdir, type(Offline_result_save_rootdir)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch: {} type: {}'.format(n_epoch_offline, type(n_epoch_offline)))
    print('batch size: {} type: {}'.format(batch_size, type(batch_size)))
   
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.sub_name_offline = sub_name_offline
    args_dict.sub_name_online = sub_name_online
    args_dict.Offline_folder_path = os.path.join(Offline_folder_path, sub_name_offline)
    args_dict.Online_folder_path = os.path.join(Online_folder_path, sub_name_online)   
    args_dict.windows_num = windows_num
    args_dict.proportion = proportion
    args_dict.Offline_result_save_rootdir = Offline_result_save_rootdir
    args_dict.Online_result_save_rootdir = Online_result_save_rootdir
    args_dict.restore_file = restore_file
    args_dict.n_epoch_offline = n_epoch_offline
    args_dict.n_epoch_online = n_epoch_online
    args_dict.batch_size = batch_size
    args_dict.ip = ip
    args_dict.port = port
    args_dict.mode = mode
    args_dict.batch_size_online = batch_size_online
    args_dict.best_validation_path = best_validation_path
    args_dict.unfreeze_encoder_offline = unfreeze_encoder_offline
    args_dict.unfreeze_encoder_online = unfreeze_encoder_online
    args_dict.use_pretrain = use_pretrain
    args_dict.trial_pre = trial_pre
    args_dict.trial_pre_target = trial_pre_target
    args_dict.update_trial = update_trial
    args_dict.alpha_distill = alpha_distill
    args_dict.update_wholeModel = update_wholeModel
    args_dict.samples_online = samples_online
    args_dict.total_trials = total_trials
    args_dict.data_preprocessing = data_preprocessing
    args_dict.session_manual = session_manual
    args_dict.session_manual_id = session_manual_id
    args_dict.para_m = para_m
    args_dict.cons_rate = cons_rate

    # data of our device
    args_dict.channel_list = ['FCZ','FC4','CPZ','FT7','CP3','FT8','FC3','CP4','OZ','TP7','TP8',
                    'O2','O1','M2','P8','P4','PZ','P3','P7','M1','T8','C4','CZ','C3',
                    'T7','F8','F4','FZ','F3','F7','FP2','FP1'
    ]
    # selected data channels
    args_dict.target_channel_list = [
            'F7', 'F3', 'FZ', 'F4', 'F8',
            'FT7','FC3','FCZ','FC4','FT8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'TP7','CP3','CPZ','CP4','TP8',
            'P7', 'P3', 'PZ', 'P4', 'P8',
                    'O1', 'O2'
    ]
    if args_dict.data_preprocessing:
        args_dict.data_online_store = np.empty((0,len(args_dict.target_channel_list)+1,512))  # data processing will add a new line
        args_dict.label_online_store = np.empty((0,))  # this is for the online update data, so that data will not be loaded again from the 1st trial when updating after trial n
    else:
        args_dict.data_online_store = np.empty((0,len(args_dict.channel_list),512))  
        args_dict.label_online_store = np.empty((0,))  # this is for the online update data, so that data will not be loaded again from the 1st trial when updating after trial n
    
    seed_everything(seed)
    
    if mode == 'Offline EEGLM':
        SeverControlOffline(args_dict)
    if mode == 'Offline EEGNet':
        SeverControlOffline(args_dict)
    elif mode == 'Online EEGLM':    
        
        #GPU setting
        cuda = torch.cuda.is_available()
        if cuda:
            print('Detected GPUs', flush = True)
            #device = torch.device('cuda')
            device = torch.device('cuda:{}'.format(gpu_idx))
        else:
            print('DID NOT detect GPUs', flush = True)
            device = torch.device('cpu')
        args_dict.device = device

        # get the best_validation_class_accuracy and best_validation_path
        experiment_dir_offline = os.path.join(Offline_result_save_rootdir, sub_name_offline)
        summary_save_dir_offline = os.path.join(experiment_dir_offline, 'hypersearch_summary')
        best_validation_class_accuracy = load_best_validation_class_accuracy_offline(summary_save_dir_offline)
        best_validation_path = load_best_validation_path_offline(summary_save_dir_offline)
        if validation_path_manual:
            best_validation_path = args.best_validation_path
            print("set best_validation_path manually: {}".format(best_validation_path))

        args_dict.best_validation_path = best_validation_path
        args_dict.best_validation_class_accuracy = best_validation_class_accuracy
        
        # get the lr and dropout value of the restore_file
        filename = best_validation_path
        match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", filename)
        if match:
            lr = float(match.group(1))
            dropout = float(match.group(2))
            # print(f"lr={lr}, dropout={dropout}")
        else:
            print("No match found.")
        args_dict.lr = lr
        args_dict.dropout = dropout

        restore_file = best_validation_path
        # move the best model from the offline experiments results
        Offline_path_encoder = os.path.join(Offline_result_save_rootdir, sub_name_offline, restore_file, 'checkpoint', 'best_model_encoder.pt')  # using the name online_model.statedict for all the online manipulations
        Offline_path_encoder_output = os.path.join(Offline_result_save_rootdir, sub_name_offline, restore_file, 'checkpoint', 'best_model_encoder_output.pt')  # using the name online_model.statedict for all the online manipulations
        
        makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint'))
        restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'online_model_encoder.pt')  # using the name online_model.statedict for all the online manipulations
        restore_path_encoder_output = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'online_model_encoder_output.pt')  # using the name online_model.statedict for all the online manipulations
        args_dict.restore_path_encoder = restore_path_encoder
        args_dict.restore_path_encoder_output = restore_path_encoder_output
        
        # Copy Offline path to restore path
        if use_pretrain:
            shutil.copy(Offline_path_encoder, restore_path_encoder)
            print('Successfully copied {} to {}'.format(Offline_path_encoder, restore_path_encoder))
            shutil.copy(Offline_path_encoder_output, restore_path_encoder_output)
            print('Successfully copied {} to {}'.format(Offline_path_encoder_output, restore_path_encoder_output))
        if not use_pretrain:
            shutil.copy(Offline_path_encoder_output, restore_path_encoder_output)
            print('Successfully copied {} to {}'.format(Offline_path_encoder_output, restore_path_encoder_output))
        
        sub_train_feature_array_offline, sub_train_label_array_offline, \
            sub_val_feature_array_offline, sub_val_label_array_offline = Offline_read_csv(os.path.join(Offline_folder_path, sub_name_offline), windows_num, proportion)
        args_dict.sub_train_feature_array_offline = sub_train_feature_array_offline
        args_dict.sub_train_label_array_offline = sub_train_label_array_offline
        args_dict.sub_val_feature_array_offline = sub_val_feature_array_offline
        args_dict.sub_val_label_array_offline = sub_val_label_array_offline
        
        accuracy_per_class_init = best_validation_class_accuracy
        predict_accuracies = []
        accuracies_per_class = []
        accuracy_per_class_iters = []
        accuracies_per_class_iterations = []
        accuracies_per_class_iterations.append([0, 0])
        accuracies_per_class_iterations.append([1, 0])
        accuracies_per_class_iterations.append([2, 0])
        accuracies_per_class_iterations_Rest = []
        accuracies_per_class_iterations_Rest.append([0, accuracy_per_class_init[0]])
        args_dict.predict_accuracies = predict_accuracies
        args_dict.accuracies_per_class = accuracies_per_class
        args_dict.accuracies_per_class_iterations = accuracies_per_class_iterations
        args_dict.accuracies_per_class_iterations_Rest = accuracies_per_class_iterations_Rest
        args_dict.accuracy_per_class_iters = accuracy_per_class_iters

        args_dict.Rest_output_data_exemplars, args_dict.Rest_output_feas_exemplars, args_dict.Rest_output_logits_exemplars, args_dict.Rest_output_label_exemplars = [],[],[],[]
        args_dict.MI1_output_data_exemplars, args_dict.MI1_output_feas_exemplars, args_dict.MI1_output_logits_exemplars, args_dict.MI1_output_label_exemplars = [],[],[],[]
        args_dict.MI2_output_data_exemplars, args_dict.MI2_output_feas_exemplars, args_dict.MI2_output_logits_exemplars, args_dict.MI2_output_label_exemplars = [],[],[],[]
          
        #SeverControlOnline(args_dict)
        SeverControlOnlineSelection(args_dict)
    
    elif mode == 'Online EEGNet':    
        
        #GPU setting
        cuda = torch.cuda.is_available()
        if cuda:
            print('Detected GPUs', flush = True)
            #device = torch.device('cuda')
            device = torch.device('cuda:{}'.format(gpu_idx))
        else:
            print('DID NOT detect GPUs', flush = True)
            device = torch.device('cpu')
        args_dict.device = device

        # get the best_validation_class_accuracy and best_validation_path
        experiment_dir_offline = os.path.join(Offline_result_save_rootdir, sub_name_offline)
        summary_save_dir_offline = os.path.join(experiment_dir_offline, 'hypersearch_summary')
        best_validation_class_accuracy = load_best_validation_class_accuracy_offline(summary_save_dir_offline)
        best_validation_path = load_best_validation_path_offline(summary_save_dir_offline)
        # set the best validation path manually if need
        if validation_path_manual:
            best_validation_path = args.best_validation_path
            print("set best_validation_path manually: {}".format(best_validation_path))
        
        args_dict.best_validation_path = best_validation_path
        args_dict.best_validation_class_accuracy = best_validation_class_accuracy
        
        # get the lr and dropout value of the restore_file
        filename = best_validation_path
        match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", filename)
        if match:
            lr = float(match.group(1))
            dropout = float(match.group(2))
            # print(f"lr={lr}, dropout={dropout}")
        else:
            print("No match found.")
        args_dict.lr = lr
        args_dict.dropout = dropout

        if not session_manual:
            restore_file = best_validation_path
            # move the best model from the offline experiments results
            Offline_path_encoder = os.path.join(Offline_result_save_rootdir, sub_name_offline, restore_file, 'checkpoint', 'best_model.pt')  
        
            makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint'))
            restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'best_model_{}.pt'.format(0))  # using the name online_model.pt for all the online manipulations
            args_dict.restore_path_encoder = restore_path_encoder
            
            # Copy Offline path to restore path
            shutil.copy(Offline_path_encoder, restore_path_encoder)
            print('Successfully copied {} to {}'.format(Offline_path_encoder, restore_path_encoder))
        else:
            restore_file = best_validation_path
            restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'best_model_{}.pt'.format(session_manual_id-1))  
            args_dict.restore_path_encoder = restore_path_encoder
            print('manual mode, using the best model last session: {}'.format(restore_path_encoder))
        
        sub_train_feature_array_offline, sub_train_label_array_offline, \
            sub_val_feature_array_offline, sub_val_label_array_offline = Offline_read_csv(os.path.join(Offline_folder_path, sub_name_offline), windows_num, proportion)
        args_dict.sub_train_feature_array_offline = sub_train_feature_array_offline
        args_dict.sub_train_label_array_offline = sub_train_label_array_offline
        args_dict.sub_val_feature_array_offline = sub_val_feature_array_offline
        args_dict.sub_val_label_array_offline = sub_val_label_array_offline
        
        accuracy_per_class_init = best_validation_class_accuracy
        predict_accuracies = []
        accuracies_per_class = []
        accuracy_per_class_iters = []
        accuracies_per_class_iterations = []
        accuracies_per_class_iterations.append([0, 0])
        accuracies_per_class_iterations.append([1, 0])
        accuracies_per_class_iterations.append([2, 0])
        accuracies_per_class_iterations_Rest = []
        accuracies_per_class_iterations_Rest.append([0, 0])
        # accuracies_per_class_iterations_Rest.append([0, accuracy_per_class_init[0]])
        args_dict.predict_accuracies = predict_accuracies
        args_dict.accuracies_per_class = accuracies_per_class
        args_dict.accuracies_per_class_iterations = accuracies_per_class_iterations
        args_dict.accuracies_per_class_iterations_Rest = accuracies_per_class_iterations_Rest
        args_dict.accuracy_per_class_iters = accuracy_per_class_iters

        args_dict.Rest_output_data_exemplars, args_dict.Rest_output_feas_exemplars, args_dict.Rest_output_logits_exemplars, args_dict.Rest_output_label_exemplars = [],[],[],[]
        args_dict.MI1_output_data_exemplars, args_dict.MI1_output_feas_exemplars, args_dict.MI1_output_logits_exemplars, args_dict.MI1_output_label_exemplars = [],[],[],[]
        args_dict.MI2_output_data_exemplars, args_dict.MI2_output_feas_exemplars, args_dict.MI2_output_logits_exemplars, args_dict.MI2_output_label_exemplars = [],[],[],[]

        #SeverControlOnline(args_dict)
        SeverControlOnlineSelection(args_dict)

    elif mode == 'Online_post':    
        
        #GPU setting
        cuda = torch.cuda.is_available()
        if cuda:
            print('Detected GPUs', flush = True)
            #device = torch.device('cuda')
            device = torch.device('cuda:{}'.format(gpu_idx))
        else:
            print('DID NOT detect GPUs', flush = True)
            device = torch.device('cpu')
        args_dict.device = device
        
        best_validation_path = args_dict.best_validation_path
        # get the lr and dropout value of the restore_file
        filename = best_validation_path
        match = re.search(r"lr(\d+\.\d+)_dropout(\d+\.\d+)", filename)
        if match:
            lr = float(match.group(1))
            dropout = float(match.group(2))
            # print(f"lr={lr}, dropout={dropout}")
        else:
            print("No match found.")
        args_dict.lr = lr
        args_dict.dropout = dropout

        restore_file = best_validation_path
        # move the best model from the offline experiments results
        Offline_path_encoder = os.path.join(Online_result_save_rootdir, sub_name_offline, restore_file, 'checkpoint', 'best_model_encoder.pt')  # using the name online_model.statedict for all the online manipulations
        Offline_path_encoder_output = os.path.join(Online_result_save_rootdir, sub_name_offline, restore_file, 'checkpoint', 'best_model_encoder_output.pt')  # using the name online_model.statedict for all the online manipulations
        
        makedir_if_not_exist(os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint'))
        restore_path_encoder = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'online_model_encoder.pt')  # using the name online_model.statedict for all the online manipulations
        restore_path_encoder_output = os.path.join(Online_result_save_rootdir, sub_name_online, restore_file, 'checkpoint', 'online_model_encoder_output.pt')  # using the name online_model.statedict for all the online manipulations
        args_dict.restore_path_encoder = restore_path_encoder
        args_dict.restore_path_encoder_output = restore_path_encoder_output
        
        # Copy Offline path to restore path
        if use_pretrain:
            shutil.copy(Offline_path_encoder, restore_path_encoder)
            print('Successfully copied {} to {}'.format(Offline_path_encoder, restore_path_encoder))
            shutil.copy(Offline_path_encoder_output, restore_path_encoder_output)
            print('Successfully copied {} to {}'.format(Offline_path_encoder_output, restore_path_encoder_output))
        if not use_pretrain:
            shutil.copy(Offline_path_encoder_output, restore_path_encoder_output)
            print('Successfully copied {} to {}'.format(Offline_path_encoder_output, restore_path_encoder_output))
        """
        sub_train_feature_array_offline, sub_train_label_array_offline, \
            sub_val_feature_array_offline, sub_val_label_array_offline = Offline_read_csv(os.path.join(Offline_folder_path, sub_name_offline), windows_num, proportion)
        """
        args_dict.best_validation_class_accuracy = 0.0
        args_dict.sub_train_feature_array_offline = []
        args_dict.sub_train_label_array_offline = []
        args_dict.sub_val_feature_array_offline = []
        args_dict.sub_val_label_array_offline = []
        
        #SeverControlOnline(args_dict)
        SeverControlOnlineSelection(args_dict)
    elif mode == 'OnlineTest':
        SeverControlOnlineTest(args_dict)
        #SeverControlOnline(args_dict)
        