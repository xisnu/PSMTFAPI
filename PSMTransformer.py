import tensorflow as tf
import keras.backend as k
from keras.layers import Layer,Recurrent

class Bumblebee(Layer):
    def __init__(self,nodes,**kwargs):
        self.nodes=nodes
        super(Bumblebee,self).__init__(**kwargs)
        print("Transformer Initiated")

    def build(self, input_shape):
        self.timesteps=input_shape[1]
        self.input_dim=input_shape[2]
        self.Wq=self.add_weight('Wq',[self.input_dim,self.nodes],initializer='uniform')#I,H
        self.Wk=self.add_weight('Wk',[self.input_dim,self.nodes],initializer='uniform')#I,H
        self.Wv=self.add_weight('Wv',[self.input_dim,self.nodes],initializer='uniform')#I,H
        super(Bumblebee,self).build(input_shape)
        print("Transformer Built")

    def call(self, inputs, **kwargs):
        self.Q=k.dot(inputs,self.Wq)#N,T,I x I,H = N,T,H
        self.K=k.dot(inputs,self.Wk)#N,T,I x I,H = N,T,H
        self.V=k.dot(inputs,self.Wv)#N,T,I x I,H = N,T,H
        K_t=tf.transpose(self.K,[0,2,1])#N,H,T
        dot_qk=k.batch_dot(self.Q,K_t,[2,1])#N,T,H x N,H,T = N,T,T
        qt_softmax=k.softmax(dot_qk,axis=2)#N,T,T
        qtv=k.batch_dot(qt_softmax,self.V,[2,1])#N,T,H
        print("Transformer called: return shape",k.int_shape(qtv))
        return qtv

    def compute_output_shape(self, input_shape):
        return (None,self.timesteps,self.nodes)

