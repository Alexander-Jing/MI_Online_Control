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

import argparse
import time

from helpers import models
from helpers import brain_data
from helpers.utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time
from Offline_models.Offline_train_EEGNet import Offline_train_classifierEEGNet
from Online_models.Online_train_EEGNet import Online_train_classifier
from Offline_synthesizing_results.synthesize_hypersearch_for_a_subject import synthesize_hypersearch

def SeverControlOnlineTest(args_dict):
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
    sub_name = args_dict.sub_name
    config_length = 13
    
    addr = (ip, port) #设置服务端ip地址和端口号
    buff_size = 65535         #消息的最大长度
    tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    tcpSerSock.bind(addr)
    tcpSerSock.listen(1)
    tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
    while True:
        print('等待连接...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('***********Online Test New Manipulation**********')
        print("Caution: it is the TESTING MODE! Models will NOT predict!")
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
                
                # 模型计算出来的结果准备发送回matlab
                predict = motor_class  # 这里是模拟的返回数据
                
                # 在字典里增加一个L来保存返回数据包的大小
                result={'L':2e8,  # 以字典的形式发送数据
                        'R':predict,
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
                args_dict.n_epoch = 4
                args_dict.order = order
                # 更新模型
                Online_train_classifier(args_dict)
                print('session: ' + str(session) + ', ' + 'trial: ' + str(trial) + ' model updated\n' )
            break
        tcpCliSock.close()
    tcpSerSock.close()