
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-
'''
LeNet：卷积-池化-卷积-池化-全连接-全连接-全连接
本例做适当调整
part1: mnist_inference.py
主要用来构建三层神经网络的全连接结构
'''
import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置图像数据参数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32
# 第二层卷积层尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 64
# 全连接层的节点个数
FC_SIZE = 512

'''
生成weights变量[get_varibale]
支持正则化损失函数
'''
def get_weight_variable(shape, stddev=0.1, regularizer=None):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))  
    return weights

'''
生成biases变量[get_varibale]
'''
def get_biase_variable(shape, initv=0.0):
    biases = tf.get_variable('biases', shape, initializer=tf.constant_initializer(initv))
    return biases

'''
定义卷积神经网络前向传播过程
支持滑动平均模型
支持正则化损失函数
添加train，用于区分训练过程和测试过程
'''
def inference(input_tensor, train, avg_class, regularizer, reuse=False):
    # 定义第一层卷积层：输入28×28×1 输出28×28×32
    with tf.variable_scope('layer1-conv1', reuse=reuse):
        weights = get_weight_variable([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
        biases = get_biase_variable([CONV1_DEEP])
        # 使用5×5×32的过滤器，步长为1,全0填充
        conv1 = tf.nn.conv2d(input_tensor, weights, strides=[1,1,1,1], padding='SAME')
        # 使用relu激活函数
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
        
    # 定义第二层池化层：输入28×28×32 输出14×14×32
    with tf.name_scope('layer2-pool1'):
        # ksize和strides首尾必须为1，ksize过滤器尺寸，strides步长
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
    # 定义第三层卷积层：输入14×14×32 输出14×14×64
    with tf.variable_scope('layer3-conv2', reuse=reuse):
        weights = get_weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
        biases = get_biase_variable([CONV2_DEEP])
        # 使用5×5×64的过滤器，步长为1,全0填充
        conv2 = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1], padding='SAME')
        # 使用relu激活函数
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
    
    # 定义第四层池化层：输入14×14×64 输出7×7×64
    with tf.name_scope('layer4-pool2'):
        # ksize和strides首尾必须为1，ksize过滤器尺寸，strides步长
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
    # 将池化结果转化为全连接层的输入：输入7×7×64 输出3136×1
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    
    # 定义第5层全连接层：输入3136×1 输出512×1
    with tf.variable_scope('layer5-fc1', reuse=reuse):
        # 只有全连接层的权重需要加入正则化
        weights = get_weight_variable([nodes, FC_SIZE], regularizer=regularizer)
        biases = get_biase_variable([FC_SIZE], initv=0.1)
        # 使用relu激活函数
        fc1 = tf.nn.relu(tf.matmul(reshaped, weights)+biases)
        # dropout避免过拟合：在训练过程中会随机将部分节点输出为0
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
    
    # 定义第6层softmax层：输入512×1 输出10×1
    with tf.variable_scope('layer6-fc2', reuse=reuse):
        # 只有全连接层的权重需要加入正则化
        weights = get_weight_variable([FC_SIZE, NUM_LABELS], regularizer=regularizer)
        biases = get_biase_variable([NUM_LABELS], initv=0.1)
        # 使用relu激活函数
        logit = tf.matmul(fc1, weights)+biases
        
    return logit


# In[ ]:




