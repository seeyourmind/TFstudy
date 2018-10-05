
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-
'''
本例是对5MNIST_ShuZiShiBieWenTi的优化处理
part1: mnist_inference.py
主要用来构建三层神经网络的全连接结构
'''
import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

'''
生成weights变量[get_varibale]
支持正则化损失函数
'''
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))  
    return weights

'''
生成biases变量[get_varibale]
'''
def get_biase_variable(shape):
    biases = tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))
    return biases

'''
定义前向传播过程
支持滑动平均模型
支持正则化损失函数
'''
def inference(input_tensor, avg_class, regularizer, reuse=False):
    # 定义第一层神经网络的变量
    with tf.variable_scope('layer1', reuse=reuse):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = get_biase_variable([LAYER1_NODE])
    # 定义第二层神经网络的变量
    with tf.variable_scope('layer2', reuse=reuse):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = get_biase_variable([OUTPUT_NODE]) 
    # 定义前向传播过程
    with tf.variable_scope('', reuse=True):
        # 获取权重与偏移量
        weights1 = tf.get_variable('layer1/weights', [INPUT_NODE, LAYER1_NODE])
        biases1 = tf.get_variable('layer1/biases', [LAYER1_NODE])
        weights2 = tf.get_variable('layer2/weights', [LAYER1_NODE, OUTPUT_NODE])
        biases2 = tf.get_variable('layer2/biases', [OUTPUT_NODE])
        # 判断是否使用滑动平均模型
        if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
            return tf.matmul(layer1, weights2)+biases2
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+avg_class.average(biases1))
            return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)


# In[ ]:




