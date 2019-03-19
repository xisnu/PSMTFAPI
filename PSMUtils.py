from __future__ import print_function
import numpy as np
import tensorflow as tf


# gpu_memory = tf.GPUOptions(per_process_memory_fraction=0.25)
# config=tf.ConfigProto(gpu_options=gpu_memory)

def get_layer_shape(layer):
    thisshape = tf.Tensor.get_shape(layer)
    ts = [thisshape[i].value for i in range(len(thisshape))]
    return ts


def BasicLSTM(varscope, nodes, rnn_input, input_states=None):
    with tf.variable_scope(varscope):
        cell = tf.nn.rnn_cell.BasicLSTMCell(nodes, state_is_tuple=True)
        if (input_states is None):
            lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32)
        else:
            lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=input_states)
    return lstm_out, lstm_state


def LSTM(varscope, nodes, rnn_input, input_states=None):
    with tf.variable_scope(varscope):
        cell = tf.nn.rnn_cell.LSTMCell(nodes, state_is_tuple=True)
        if (input_states is None):
            lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32)
        else:
            lstm_out, lstm_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=input_states)
    return lstm_out, lstm_state


def BLSTM(nb_nodes, input, name, seq_len=None, return_state=False, time_major=True):
    cell_name = name + "_def"
    nb_nodes = int(nb_nodes / 2)
    with tf.variable_scope(cell_name):
        f_cell = tf.nn.rnn_cell.LSTMCell(nb_nodes, state_is_tuple=True)
        b_cell = tf.nn.rnn_cell.LSTMCell(nb_nodes, state_is_tuple=True)
    op_name = name + "_op"
    with tf.variable_scope(op_name):
        if (seq_len is not None):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input, sequence_length=seq_len,
                                                                     dtype=tf.float32, time_major=time_major)
        else:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, input, dtype=tf.float32,
                                                                     time_major=time_major)

    merge = tf.concat(outputs, 2)
    # if(time_major):
    #     merge=tf.transpose(merge,[1,0,2])

    if (return_state):
        return merge, output_states
    else:
        return merge


def Convolution2D(input, filter_w, filter_h, nbfilters, stride, layername, return_filter=False, activation=''):
    # filter_dim [w,h] stride [w,h]
    shape = get_layer_shape(input)
    filter_in = shape[-1]
    # print(filter_w,filter_h,filter_in,nbfilters)
    filter = tf.Variable(tf.truncated_normal([filter_h, filter_w, filter_in, nbfilters], name=layername + "_Filter"))
    bias = tf.Variable(tf.truncated_normal([nbfilters]))
    shift = [1, stride[1], stride[0], 1]
    convolved = tf.nn.conv2d(input, filter, shift, padding='SAME', name=layername + "_convolution2d")
    if (activation == 'relu'):
        convolved = tf.nn.relu(convolved)
    else:
        convolved = tf.nn.tanh(convolved)
    convolved = tf.add(convolved, bias)
    if (return_filter):
        return convolved, filter
    print("%s: output "%layername, get_layer_shape(convolved))
    return convolved


def Convolution1D(input, filter_width, nbfilters, stride, layername, activation=True):
    shape = get_layer_shape(input)
    filter_in = shape[-1]
    filter = tf.Variable(tf.truncated_normal([filter_width, filter_in, nbfilters], name=layername + "_Filter"))
    # shift = [1, stride, 1]
    convolved = tf.nn.conv1d(input, filter, stride, padding='SAME', name=layername + "_convolution1d")
    if (activation):
        convolved = tf.nn.relu(convolved)
    return convolved


def Pooling1D(input, poolsize, stride, layername):
    input_4d = tf.expand_dims(input, axis=1)
    ksize = [1, 1, poolsize, 1]
    shift = [1, 1, stride, 1]
    pooled = tf.nn.max_pool(input_4d, ksize, shift, padding='SAME', name=layername + "_maxpool")
    input_3d = tf.squeeze(pooled, axis=1)
    return input_3d


def Pooling2D(input, poolsize, stride, layername):
    ksize = [1, poolsize[1], poolsize[0], 1]
    shift = [1, stride[1], stride[0], 1]
    pooled = tf.nn.max_pool(input, ksize, shift, padding='SAME', name=layername + "_maxpool")
    return pooled


def FullyConnected(input, nbnodes, layername, give_prob=False):
    shape = get_layer_shape(input)
    in_dim = shape[-1]
    # print("In dimesion ",in_dim)
    dense_prob = None
    W = tf.Variable(tf.truncated_normal([in_dim, nbnodes]), name=layername + "_W")
    B = tf.constant(0.1, shape=[nbnodes], name=layername + "_B")
    dense_out = tf.matmul(input, W) + B
    if (give_prob):
        dense_prob = tf.nn.softmax(dense_out)
    return dense_out, dense_prob


def TimeDistributedDense(input, dim, format='TNF', layername='td', logit=True, time_major=True):
    # format TNF means Time Major
    input_shape = get_layer_shape(input)
    if (format == 'TNF'):
        time = input_shape[0]
    elif (format == 'NTF'):
        time = input_shape[1]
    feat = input_shape[-1]
    time_squeezed = tf.reshape(input, [-1, feat])
    dense_out, dense_prob = FullyConnected(time_squeezed, dim, layername + 'td_dense', give_prob=True)
    shape = get_layer_shape(dense_out)
    new_feat = shape[-1]
    td_logit = tf.reshape(dense_out, [-1, time, new_feat])
    td_prob = tf.reshape(dense_prob, [-1, time, new_feat])
    if (time_major):
        td_logit = tf.transpose(td_logit, [1, 0, 2])
        td_prob = tf.transpose(td_prob, [1, 0, 2])
    return td_logit, td_prob

# class_labels=np.random.randint(0,10,[150])
# convert_to_onehot(class_labels,10)
