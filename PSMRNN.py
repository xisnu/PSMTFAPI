from __future__ import print_function
import keras,math
from keras.layers import Layer,RNN,LSTM
from keras.layers.recurrent import Recurrent
import keras.backend as K
import tensorflow as tf
from keras.layers.wrappers import Bidirectional
from keras.initializers import RandomNormal
from keras.regularizers import l2
import numpy as np


def P_time_distributed_dense(x,w,b):
    """
    :param x: a 3D input sequence (N,T,F)
    :param w: a weight matrix for every timestep
    :param b: a bias for every timestep
    :return: x dot w
    """
    #input_dim=K.shape(x)[-1]
    out=K.dot(x,w) #[N,e_ts,E] X [E,A] = [N,e_ts,A]
    out=K.bias_add(out,b) #[N,e_ts,A]+[A] = [N,e_ts,A]
    return out

class PLSTMCell(Layer):
    def __init__(self,nb_units,**kwargs):
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        super(PLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim=input_shape[-1]
        #self.timesteps=input_shape[-2]
        print("nodes = %d ,input_dim %d"%(self.nodes,self.input_dim))
        #inititate weights for Input gate
        self.Wki=self.add_weight(shape=[self.input_dim,self.nodes],initializer='orthogonal',name='W_ki')
        self.Wri=self.add_weight('W_ri',shape=[self.nodes,self.nodes],initializer='orthogonal')
        # inititate weights for Forget/Reset gate
        self.Wkf = self.add_weight('W_kf', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wrf = self.add_weight('W_rf', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # inititate weights for Update gate
        self.Wkc = self.add_weight('W_kc', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wrc = self.add_weight('W_rc', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # inititate weights for Output gate
        self.Wko = self.add_weight('W_ko', shape=[self.input_dim, self.nodes], initializer='orthogonal')
        self.Wro = self.add_weight('W_ro', shape=[self.nodes, self.nodes], initializer='orthogonal')
        # All gate weights initialized
        print("Weights Initiated")

    def call(self, inputs, states, training=None):
        print(states)
        self.h_tm1=states[0]
        self.c_tm1=states[1]
        self.inputs=inputs
        self.call_input_gate()
        self.call_forget_gate()
        self.call_output_gate()
        self.call_update_gate()
        #Now compute Updated C state
        C_reset=self.c_tm1*self.f
        C_new=self.i*self.c
        self.C_out=C_reset+C_new
        #Now compute Updated H state
        self.h_out=K.tanh(self.C_out)*self.o

        return self.h_out,[self.h_out,self.C_out]

    def call_input_gate(self):
        Iiw=K.dot(self.inputs,self.Wki)
        Irw=K.dot(self.h_tm1,self.Wri)
        i=Iiw+Irw
        self.i=K.hard_sigmoid(i)

    def call_forget_gate(self):
        Fiw=K.dot(self.inputs,self.Wkf)
        Frw=K.dot(self.h_tm1,self.Wrf)
        f=Fiw+Frw
        self.f=K.hard_sigmoid(f)

    def call_update_gate(self):
        Ciw=K.dot(self.inputs,self.Wkc)
        Crw=K.dot(self.h_tm1,self.Wrc)
        c=Ciw+Crw
        self.c=K.tanh(c)

    def call_output_gate(self):
        Oiw=K.dot(self.inputs,self.Wko)
        Orw=K.dot(self.h_tm1,self.Wro)
        o=Oiw+Orw
        self.o=K.hard_sigmoid(o)

class PAttentionCell(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,output_dim,return_alphas=False,name="Attention",**kwargs):
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim
        super(PAttentionCell, self).__init__(**kwargs)
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name

    def build(self, input_shape):
        self.input_dim=input_shape[-1]
        self.timesteps=input_shape[-2]
        print("nodes = %d ,input_dim %d"%(self.nodes,self.input_dim))
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Wr=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer)
        #inititate weights for Context vector Ct calculation
        self.Va=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='Va') #[A] to be multiplied with tanh(alignment output)
        self.Wa=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Wa')#[D,A] to be multiplied by Si-1 (Decoder hidden state)
        self.Ua=self.add_weight(shape=[self.input_dim,self.nodes],initializer=self.weight_initializer,name='Ua')#[E,A] to be multiplied with hj (Encoder output/annotation)
        self.ba=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='ba')#[A] Attention bias
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr')#[O,D]
        self.Urr=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr')#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br')#[D]
        self.Cr=self.add_weight(shape=[self.input_dim,self.nodes],initializer=self.weight_initializer,name='Cr')#[E,D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz')#[O,D]
        self.Urz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz')#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz')#[D]
        self.Cz = self.add_weight(shape=[self.input_dim, self.nodes], initializer=self.weight_initializer, name='Cz')#[E,D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp')#[O,D]
        self.Urp=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp')#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp')#[D]
        self.Cp = self.add_weight(shape=[self.input_dim, self.nodes], initializer=self.weight_initializer, name='Cp')#[E,D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.output_dim],initializer=self.weight_initializer,name='Wko')#[O,O]
        self.Uro=self.add_weight(shape=[self.nodes,self.output_dim],initializer=self.weight_initializer,name='Uro')#[D,O]
        self.bo = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bo')#[D]
        self.Co = self.add_weight(shape=[self.input_dim, self.output_dim], initializer=self.weight_initializer, name='Co')#[E,O]
        print("Weights Initiated")


    def call(self, input_sequence,mask=None, training=None, initial_state=None):
        #print(states)
        self.input_sequence=input_sequence#[N,e_ts,E]
        self.U_h=P_time_distributed_dense(self.input_sequence,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        return super(PAttentionCell,self).call(input_sequence)

    def step(self, inputs, states):
        ytm,stm=states #states=[N,O],[N,D]
        #enc_hidden=dec_hidden
        #repeat the hidden state over all timesteps of encoder
        self.repeated_state=K.repeat(stm,self.timesteps) #N,e_ts,D
        #multiply the hidden state with Wa
        self._WaStm = K.dot(self.repeated_state,self.Wa) #[N,e_ts,D] X [D,A] = [N,e_ts,A]
        self._WaStm = K.tanh(self._WaStm+self.U_h)#[N,e_ts,A]+[N,e_ts,A] = [N,e_ts,A]
        Va_expanded=K.expand_dims(self.Va) #[A,1]
        self.e_ij = K.dot(self._WaStm,Va_expanded) #[N,e_ts,1]
        self.alpha_ij=K.softmax(self.e_ij,axis=1) #still [N,e_ts,1] one alpha_ij for every combinantion of input pos i and output position j
        #Now calculate context vector
        self.c_t=K.batch_dot(self.alpha_ij,self.input_sequence,axes=1)#[N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        self.c_t=K.squeeze(self.c_t,1)#[N,E]
        self.call_reset_gate(ytm,stm)# get self.r
        self.call_update_gate(ytm,stm)# get self.Z_t
        self.call_input_gate(ytm,stm)#get self.S_t :this is new state from this step
        self.call_output_gate(ytm,stm)#get self.Y_t :this is new output token from this step
        if(self.return_alphas):
            return self.alpha_ij,[self.Y_t,self.S_t]
        else:
            return self.Y_t,[self.Y_t,self.S_t]#[N,O],states=[N,O],[N,D]

    #return self.S_t=[N,D]
    def call_input_gate(self,ytm,stm):#proposal
        r_stm=self.r*stm#[N,D]*[N,D]
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(r_stm,self.Urp)#[N,D] X [D,D] = [N,D]
        Icw=K.dot(self.c_t,self.Cp)#[N,E] X [E,D] = [N,D]
        I=K.tanh(Iiw+Irw+Icw+self.bp)#[N,D]+[N,D]+[N,D]+[D] = [N,D]
        #new updated state
        self.S_t=((1-self.Z_t)*stm)+(self.Z_t*I)#[N,D]
        print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_reset_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Urr)#[N,D] X [D,D] = [N,D]
        Rcw=K.dot(self.c_t,self.Cr)#[N,E] X [E,D] = [N,D]
        R=Riw+Rrw+Rcw+self.br#[N,D]+[N,D]+[D] = [N,D]
        self.r=K.sigmoid(R)#[N,D]
        print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Urz)#[N,D] X [D,D] = [N,D]
        Zcw=K.dot(self.c_t,self.Cz)#[N,E] X [E,D] =[N,D]
        z=Ziw+Zcw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        self.Z_t=K.sigmoid(z)#[N,D]
        print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,O]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,O] = [N,O]
        Orw=K.dot(stm,self.Uro)#[N,D] X [D,O] = [N,O]
        Ocw=K.dot(self.c_t,self.Co)#[N,E] X [E,O] = [N,O]
        self.Y_t=K.sigmoid(Oiw+Orw+Ocw)#[N,O]
        print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(PAttentionCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

class PLSTM(RNN):
    def __init__(self,nodes,return_sequences=True,return_state=True,go_backwards=False,stateful=False,unroll=False,input_dim=None):
        if(input_dim is not None):
            cell=PLSTMCell(input_dim)
        else:
            cell=PLSTMCell(nodes)
        super(PLSTM,self).__init__(cell,return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(PLSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)

class PSMLSTM(Recurrent):
    def __init__(self,nodes,name,return_sequence=False,initializer='uniform',**kwargs):
        self.nodes=nodes
        self.name=name
        self.return_sequences=return_sequence
        self.initializer=keras.initializers.get(initializer)
        super(PSMLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building:..Input shape ",input_shape)
        self.input_dim=input_shape[-1]
        self.input_time=input_shape[-2]
        self.states=[None,None]
        #Initial state weights
        self.Wsm1=self.add_weight('Wir',(self.input_dim,self.nodes),initializer=self.initializer)
        #Reset / Forget gate weights
        self.Wir=self.add_weight('Wir',(self.input_dim,self.nodes),initializer=self.initializer)
        self.Wrr=self.add_weight('Wrr',(self.nodes,self.nodes),initializer=self.initializer)
        self.br=self.add_weight('br',shape=[self.nodes],initializer=self.initializer)
        #Update gate weights
        self.Wiu=self.add_weight('Wiu',(self.input_dim,self.nodes),initializer=self.initializer)
        self.Wru = self.add_weight('Wru', (self.nodes, self.nodes), initializer=self.initializer)
        self.bu=self.add_weight('bu',[self.nodes],initializer=self.initializer)
        #Output gate weights
        self.Wio = self.add_weight('Wio', (self.input_dim, self.nodes), initializer=self.initializer)
        self.Wro = self.add_weight('Wro', (self.nodes, self.nodes), initializer=self.initializer)
        self.bo = self.add_weight('bo', [self.nodes], initializer=self.initializer)
        # Input gate weights
        self.Wii = self.add_weight('Wii', (self.input_dim, self.nodes), initializer=self.initializer)
        self.Wri = self.add_weight('Wri', (self.nodes, self.nodes), initializer=self.initializer)
        self.bi = self.add_weight('bi', [self.nodes], initializer=self.initializer)
        print("Build:All weights initialized for layer ",self.name)

    def reset_gate(self,xt,htm):
        Ri=K.dot(xt,self.Wir)#[N,I] x [I,H] = [N,H]
        Rr=K.dot(htm,self.Wrr)#[N,H] x [H,H] = [N,H]
        #print("reset: Inputs", K.int_shape(Ri),K.int_shape(Rr),K.int_shape(self.br))
        R=Ri+Rr+self.br#[N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(R)

    def input_gate(self,xt,htm):
        Ii=K.dot(xt,self.Wii)#[N,I] x [I,H] = [N,H]
        Ir = K.dot(htm, self.Wri)  #[N,H] x [H,H] = [N,H]
        I=Ii+Ir+self.bi #[N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(I)

    def update_gate(self,xt,htm):
        Ui = K.dot(xt, self.Wiu)  # [N,I] x [I,H] = [N,H]
        Ur = K.dot(htm, self.Wru)  # [N,H] x [H,H] = [N,H]
        U = Ui + Ur + self.bu  # [N,H]+[N,H]+[H] = [N,H]
        return K.tanh(U)

    def output_gate(self,xt,htm):
        Oi = K.dot(xt, self.Wio)  # [N,I] x [I,H] = [N,H]
        Or = K.dot(htm, self.Wro)  # [N,H] x [H,H] = [N,H]
        O = Oi + Or + self.bo  # [N,H]+[N,H]+[H] = [N,H]
        return K.sigmoid(O)


    def step(self, inputs, states):
        print("state shape, inputs shape", K.int_shape(states[0]),K.int_shape(inputs))
        Ctm,htm=states
        xt=inputs
        ig=self.input_gate(xt,htm)#[N,H]
        rg=self.reset_gate(xt,htm)#[N,H]
        ug=self.update_gate(xt,htm)#[N,H]
        og=self.output_gate(xt,htm)#[N,H]
        C_reset=Ctm*rg
        C_update=ig*ug
        C_t=C_reset+C_update
        h_t=K.tanh(C_t)*og
        return h_t,[C_t,h_t]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        print("Calling: Inputs shape ",K.int_shape(inputs[:,0]))
        return super(PSMLSTM, self).call(inputs)

    def get_initial_state(self, inputs):
        Sm1=K.tanh(K.dot(inputs[:,0],self.Wsm1))
        #Sm1=K.zeros(shape=[self.nodes])
        return [Sm1,Sm1]

    def compute_output_shape(self, input_shape):
        if(self.return_sequences):
            shape=(None,self.input_time,self.nodes)
        else:
            shape=(None,self.nodes)
        return shape

class BadhanauAttentionDecoder(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,annotation_dim,output_dim,return_alphas=False,name="Attention",**kwargs):
        super(BadhanauAttentionDecoder, self).__init__(**kwargs)
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim
        self.annotation_dim=annotation_dim
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name
        self.predict=False

    def build(self, input_shape):
        #print("build: input shape",input_shape)
        self.input_dim=input_shape[0][-1]
        #self.timesteps=input_shape[-2]
        self.d_ts=input_shape[0][1]
        #print("nodes = ,input_dim , annotation dim ",self.nodes,self.input_dim,self.annotation_dim)
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Wr=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer)
        #inititate weights for Context vector Ct calculation
        self.Va=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='Va') #[A] to be multiplied with tanh(alignment output)
        self.Wa=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Wa')#[D,A] to be multiplied by Si-1 (Decoder hidden state)
        self.Ua=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Ua')#[E,A] to be multiplied with hj (Encoder output/annotation)
        self.ba=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='ba')#[A] Attention bias
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr')#[O,D]
        self.Urr=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr')#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br')#[D]
        self.Cr=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Cr')#[E,D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz')#[O,D]
        self.Urz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz')#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz')#[D]
        self.Cz = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cz')#[E,D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp')#[O,D]
        self.Urp=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp')#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp')#[D]
        self.Cp = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cp')#[E,D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.output_dim],initializer=self.weight_initializer,name='Wko')#[O,O]
        self.Uro=self.add_weight(shape=[self.nodes,self.output_dim],initializer=self.weight_initializer,name='Uro')#[D,O]
        self.bo = self.add_weight(shape=[self.output_dim], initializer=self.weight_initializer, name='bo')#[D]
        self.Co = self.add_weight(shape=[self.annotation_dim, self.output_dim], initializer=self.weight_initializer, name='Co')#[E,O]
        print("Built %s"%(self.name))


    def call(self, inputs,mask=None, training=None, initial_state=None,from_prev=True):
        #print(states)
        self.input_sequence = inputs[0]  # [N,d_ts,E]
        self.annotation = inputs[1]  # [N,e_ts,E]
        if(from_prev):
            self.input_from_prev=True
        else:
            self.input_from_prev = False
        self.e_ts=K.int_shape(self.annotation)[1]
        #print("Call is OK. encoder time %d decoder time %d"%(self.e_ts,self.d_ts))
        #self.U_h=P_time_distributed_dense(self.annotation,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        self.U_h = K.dot(self.annotation, self.Ua)  # [N,e_ts,E] X [E,A] = [N,e_ts,A]
        return super(BadhanauAttentionDecoder,self).call(self.input_sequence)

    def step(self, inputs, states):
        #print("Step:inputs shape ",K.int_shape(inputs))
        if(self.input_from_prev):
            ytm, stm = states  # states=[N,O],[N,D] take input from previous output
        else:#take input from given data
            ytm = inputs
            _, stm = states
        #enc_hidden=dec_hidden
        '''_____Computation of a() starts ____'''
        #repeat the hidden state over all timesteps of encoder
        self.repeated_state=K.repeat(stm,self.e_ts) #N,e_ts,D
        #multiply the hidden state with Wa
        self._WaStm = K.dot(self.repeated_state,self.Wa) #[N,e_ts,D] X [D,A] = [N,e_ts,A]
        #self.U_h=K.dot(self.annotation,self.Ua)#[N,e_ts,E] X [E,A] = [N,e_ts,A]
        self._WaStm = K.tanh(self._WaStm+self.U_h+self.ba)#[N,e_ts,A]+[N,e_ts,A] = [N,e_ts,A]
        Va_expanded=K.expand_dims(self.Va) #[A,1]
        self.e_ij = K.dot(self._WaStm,Va_expanded) #[N,e_ts,1]
        '''_____Computation of a() ends, Now find alpha_ij ____'''
        self.alpha_ij=K.softmax(self.e_ij,axis=1) #still [N,e_ts,1] one alpha_ij for every combinantion of input pos i and output position j
        #print("Step:Alpha shape ",K.int_shape(self.alpha_ij))
        '''_____alpha_ij computed, Now compute Context vector Ct ____'''
        self.c_t=K.batch_dot(self.alpha_ij,self.annotation,axes=1)#[N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        self.c_t=K.squeeze(self.c_t,1)#[N,E]
        '''_____Context vector found, Now compute gates____'''
        self.call_reset_gate(ytm,stm)# get self.r
        self.call_update_gate(ytm,stm)# get self.Z_t
        self.call_input_gate(ytm,stm)#get self.S_t :this is new state from this step
        self.call_output_gate(ytm,stm)#get self.Y_t :this is new output token from this step
        if(self.return_alphas):
            return self.alpha_ij,[self.Y_t,self.S_t]
        else:
            return self.Y_t,[self.Y_t,self.S_t]#[N,O],states=[N,O],[N,D]

    #return self.S_t=[N,D]
    def call_input_gate(self,ytm,stm):#proposal
        r_stm=self.r*stm#[N,D]*[N,D]
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(r_stm,self.Urp)#[N,D] X [D,D] = [N,D]
        Icw=K.dot(self.c_t,self.Cp)#[N,E] X [E,D] = [N,D]
        I=K.tanh(Iiw+Irw+Icw+self.bp)#[N,D]+[N,D]+[N,D]+[D] = [N,D]
        #new updated state
        self.S_t=((1-self.Z_t)*stm)+(self.Z_t*I)#[N,D]
        #print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_reset_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Urr)#[N,D] X [D,D] = [N,D]
        Rcw=K.dot(self.c_t,self.Cr)#[N,E] X [E,D] = [N,D]
        R=Riw+Rrw+Rcw+self.br#[N,D]+[N,D]+[D] = [N,D]
        self.r=K.sigmoid(R)#[N,D]
        #print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Urz)#[N,D] X [D,D] = [N,D]
        Zcw=K.dot(self.c_t,self.Cz)#[N,E] X [E,D] =[N,D]
        z=Ziw+Zcw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        self.Z_t=K.sigmoid(z)#[N,D]
        #print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,O]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,O] = [N,O]
        Orw=K.dot(stm,self.Uro)#[N,D] X [D,O] = [N,O]
        Ocw=K.dot(self.c_t,self.Co)#[N,E] X [E,O] = [N,O]
        self.Y_t=K.sigmoid(Oiw+Orw+Ocw+self.bo)#[N,O]
        #print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        #print('get_initial_state:inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(BadhanauAttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.d_ts, self.e_ts)
        else:
            return (None, self.d_ts, self.output_dim)

class PSMAttentionDecoder(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,output_dim,return_alphas=False,name="PSMAttention",output_context=False,train_on_both=False,**kwargs):
        super(PSMAttentionDecoder, self).__init__(**kwargs)
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim        
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name
        self.output_context=output_context
        self.predict=False
        print("Training on both Prev_out and current input ",train_on_both)
        self.train_on_both=train_on_both

    def build(self, input_shape):
        print("build: input shape",input_shape)
        self.annotation_dim = input_shape[1][-1]
        self.input_dim=input_shape[0][-1]
        #self.timesteps=input_shape[-2]
        self.d_ts=input_shape[0][1]
        #print("nodes = ,input_dim , annotation dim ",self.nodes,self.input_dim,self.annotation_dim)
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Wr=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer )
        #inititate weights for Context vector Ct calculation
        self.Va=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='Va' ) #[A] to be multiplied with tanh(alignment output)
        self.Wa=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Wa' )#[D,A] to be multiplied by Si-1 (Decoder hidden state)
        self.Ua=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Ua' )#[E,A] to be multiplied with hj (Encoder output/annotation)
        self.ba=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='ba' )#[A] Attention bias
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr' )#[O,D]
        self.Ur=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr' )#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br' )#[D]
        self.Cr=self.add_weight(shape=[self.annotation_dim,self.nodes],initializer=self.weight_initializer,name='Cr' )#[E,D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz' )#[O,D]
        self.Uz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz' )#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz' )#[D]
        self.Cz = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cz' )#[E,D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp' )#[O,D]
        self.Up=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp' )#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp' )#[D]
        self.Cp = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Cp' )#[E,D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wko' )#[O,D]
        self.Uo=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Uro' )#[D,D]
        self.bo = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bo' )#[D]
        self.Co = self.add_weight(shape=[self.annotation_dim, self.nodes], initializer=self.weight_initializer, name='Co' )#[E,D]
        self.Wy = self.add_weight(shape=[self.nodes, self.output_dim], initializer=self.weight_initializer, name='Wy' )#[D,O]
        if(self.output_context):
            self.Wd = self.add_weight(shape=[(self.annotation_dim+self.output_dim), self.output_dim], initializer=self.weight_initializer, name='Wd' )  # [E+O,O]
            self.bd = self.add_weight(shape=[self.output_dim], initializer=self.weight_initializer, name='bd' )  # [O]
        print("Built %s"%(self.name))


    def call(self, inputs,mask=None, training=None, initial_state=None,from_prev=True):
        #print(states)
        self.input_sequence = inputs[0]  # [N,d_ts,E]
        self.annotation = inputs[1]  # [N,e_ts,E]
        if(from_prev):
            self.input_from_prev=True
        else:
            self.input_from_prev = False
        self.e_ts=K.int_shape(self.annotation)[1]
        one = K.constant(1.0)
        step = one / self.e_ts
        self.gaussian_step=step
        #print("Call is OK. encoder time %d decoder time %d"%(self.e_ts,self.d_ts))
        #self.U_h=P_time_distributed_dense(self.annotation,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        self.U_h = K.dot(self.annotation, self.Ua)  # [N,e_ts,E] X [E,A] = [N,e_ts,A]
        return super(PSMAttentionDecoder,self).call(self.input_sequence)


    def step(self, inputs, states):
        #print("Step:inputs shape ",K.int_shape(inputs))
        if(self.input_from_prev):
            ytm, stm = states  # states=[N,O],[N,D] take input from previous output
        else:#take input from given data
            ytm = inputs
            _, stm = states
        #enc_hidden=dec_hidden
        if(self.train_on_both):
            ytm=K.softmax(states[0]+inputs)
        '''_____Computation of a() starts ____'''
        #repeat the hidden state over all timesteps of encoder
        self.repeated_state=K.repeat(stm,self.e_ts) #N,e_ts,D
        #multiply the hidden state with Wa
        self._WaStm = K.dot(self.repeated_state,self.Wa) #[N,e_ts,D] X [D,A] = [N,e_ts,A]
        self._WaStm = K.tanh(self._WaStm+self.U_h+self.ba)#[N,e_ts,A]+[N,e_ts,A] = [N,e_ts,A]
        Va_expanded=K.expand_dims(self.Va) #[A,1]
        self.e_ij = K.dot(self._WaStm,Va_expanded) #[N,e_ts,1]
        # mu=(self.d_ts/K.cast(self.e_ts,tf.float32))*this_step
        # soft_attention=self.gen_gaussian(mu,0.1)#[e_ts,1]
        # self.e_ij = self.e_ij * soft_attention
        print("Step:e_ij after soft attention ", K.int_shape(self.e_ij))
        '''_____Computation of a() ends, Now find alpha_ij ____'''
        self.alpha_ij=K.softmax(self.e_ij,axis=1) #still [N,e_ts,1] one alpha_ij for every combinantion of input pos i and output position j
        #print("Step:Alpha shape ",K.int_shape(self.alpha_ij))
        '''_____alpha_ij computed, Now compute Context vector Ct ____'''
        print("Alpha ",K.int_shape(self.alpha_ij)," annotation ",K.int_shape(self.annotation))
        self.c_t=tf.reduce_sum(self.alpha_ij*self.annotation,axis=1)#,axes=1)#[N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        print("C_t shape ",K.int_shape(self.c_t))
        #self.c_t=K.squeeze(self.c_t,1)#[N,E]

        if(self.output_context):
            ytm=self.context_gate(ytm)

        '''_____Context vector found, Now compute gates____'''
        self.call_reset_gate(ytm,stm)# get self.r
        self.call_update_gate(ytm,stm)# get self.Z_t
        self.call_proposal_gate(ytm,stm)#get self.S_t :this is new state from this step
        self.call_output_gate(ytm,stm)#get self.Y_t :this is new output token from this step
        if(self.return_alphas):
            return self.alpha_ij,[self.Y_t,self.S_t]
        else:
            return self.Y_t,[self.Y_t,self.S_t]#[N,O],states=[N,O],[N,D]

    def context_gate(self,ytm):
        d = K.concatenate([ytm, self.c_t], axis=1)  # [N,O+E]
        d = K.dot(d, self.Wd)  # [N,O]
        d = K.tanh(d + self.bd)  # [N,O]
        return d

    #return self.S_t=[N,D]
    def call_proposal_gate(self,ytm,stm):#proposal
        r_stm=self.r*stm#[N,D]*[N,D]
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(r_stm,self.Up)#[N,D] X [D,D] = [N,D]
        Icw=K.dot(self.c_t,self.Cp)#[N,E] X [E,D] = [N,D]
        I=K.tanh(Iiw+Irw+self.bp+Icw)#[N,D]+[N,D]+[N,D]+[D] = [N,D]
        #new updated state
        self.S_t=tf.add(((1-self.Z_t)*stm),(self.Z_t*I))#[N,D]
        #print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_reset_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Ur)#[N,D] X [D,D] = [N,D]
        Rcw=K.dot(self.c_t,self.Cr)#[N,E] X [E,D] = [N,D]
        R=Riw+Rrw+Rcw+self.br#[N,D]+[N,D]+[D] = [N,D]
        self.r=K.sigmoid(R)#[N,D]
        #print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Uz)#[N,D] X [D,D] = [N,D]
        Zcw=K.dot(self.c_t,self.Cz)#[N,E] X [E,D] =[N,D]
        z=Ziw+Zcw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        self.Z_t=K.sigmoid(z)#[N,D]
        #print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,O]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,D] = [N,D]
        Orw=K.dot(stm,self.Uo)#[N,D] X [D,D] = [N,D]
        Ocw=K.dot(self.c_t,self.Co)#[N,E] X [E,D] = [N,D]
        self.h_t=K.sigmoid(Orw+Oiw+self.bo+Ocw)#[N,D]
        self.Y_t=self.h_t*K.tanh(self.S_t)#[N,D]
        self.Y_t=K.dot(self.Y_t,self.Wy)#[N,O]
        self.Y_t=tf.nn.softmax(self.Y_t)
        #print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        #print('get_initial_state:inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])
        self.states=[y0, s0]

        return self.states

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(PSMAttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.d_ts, self.e_ts)
        else:
            return (None, self.d_ts, self.output_dim)

def pyramid(x):
    #x is a 3D tensor of shape N,TS,F
    #returns a 3D tensor of shape N,TS/2,F
    shape=K.int_shape(x)
    f=shape[-1]
    actual_ts=shape[-2]
    if(actual_ts%2!=0):
        x=x[:,:-1,:]
    ts=(shape[-2])/2
    t1=K.reshape(x[:,::2,:],[-1,ts,f])
    t2 = K.reshape(x[:, 1::2, :], [-1, ts, f])
    t3=K.concatenate([t1,t2],axis=2)
    # t3=tf.add(t1,t2)
    return t3

def pyramid_output(input_shape):
    shape=list(input_shape)
    shape[1]=shape[1]/2
    shape[2]=shape[2]*2
    return tuple(shape)
#How to use Pyramid

class LuongAttentionDecoder(Recurrent):
    '''
    A=Attention size (nb_units)
    O=Output size (output_dim) Lexicon size
    E=Number of Encoder nodes (input_dim)
    D=Number of Decoder nodes
    e_ts=Encoder timestep
    '''

    def __init__(self,nb_units,annotation_dim,output_dim,return_alphas=False,name="LuongAttention",mode='dot',**kwargs):
        super(LuongAttentionDecoder, self).__init__(**kwargs)
        self.nodes=nb_units
        self.state_size=(nb_units,nb_units)
        self.output_dim=output_dim
        self.annotation_dim=annotation_dim
        self.return_sequences=True
        self.return_alphas=return_alphas
        self.weight_constraint=keras.constraints.get(None)
        self.weight_initializer=keras.initializers.get('uniform')
        self.name=name
        self.mode=mode
        self.predict=False
        print("Implementing Luong Attention: Score computation by %s"%self.mode)

    def build(self, input_shape):
        #print("build: input shape",input_shape)
        self.input_dim=input_shape[0][-1]
        #self.timesteps=input_shape[-2]
        self.d_ts=input_shape[0][1]
        #print("nodes = ,input_dim , annotation dim ",self.nodes,self.input_dim,self.annotation_dim)
        #Weights for all the gates are initialized here
        '''
        every gate has four weights
            1)Wk=weight for input to gate (kernel)
            2)Ur=weight for recursive connection
            3)b=bias for this gate
            4)C=weight for context vector
        '''
        self.states = [None, None]
        #Weight matrix for initial state calculation
        self.Ws = self.add_weight('Ws',shape=(self.input_dim, self.nodes),initializer=self.weight_initializer)
        #initiate weights for Reset gate(r)
        self.Wkr=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkr')#[O,D]
        self.Urr=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urr')#[D,D]
        self.br=self.add_weight(shape=[self.nodes],initializer=self.weight_initializer,name='br')#[D]
        # initiate weights for Update gate(z)
        self.Wkz=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkz')#[O,D]
        self.Urz=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urz')#[D,D]
        self.bz = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bz')#[D]
        #initiate weights for Input gate(p)/proposal
        self.Wkp=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wkp')#[O,D]
        self.Urp=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Urp')#[D,D]
        self.bp = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bp')#[D]
        #Initiate weights for Output gate(o)
        self.Wko=self.add_weight(shape=[self.output_dim,self.nodes],initializer=self.weight_initializer,name='Wko')#[O,O]
        self.Uro=self.add_weight(shape=[self.nodes,self.nodes],initializer=self.weight_initializer,name='Uro')#[D,O]
        self.bo = self.add_weight(shape=[self.nodes], initializer=self.weight_initializer, name='bo')#[D]
        # inititate weights for Context vector Ct calculation
        if(self.mode=='dot'):
            self.Wa=self.add_weight(shape=[self.nodes,1],initializer=self.weight_initializer,name='Wa_dot')
        elif(self.mode=='concat'):
            self.Wa = self.add_weight(shape=[self.nodes*2, 1], initializer=self.weight_initializer, name='Wa_concat')
        self.Wc=self.add_weight(shape=[(self.annotation_dim+self.nodes),self.output_dim],initializer=self.weight_initializer, name='Wc')
        self.built=True
        print("Built %s"%(self.name))


    def call(self, inputs,mask=None, training=None, initial_state=None,from_prev=True):
        #print(states)
        self.input_sequence = inputs[0]  # [N,d_ts,E]
        self.annotation = inputs[1]  # [N,e_ts,E]
        print("Annotation shape ",K.int_shape(self.annotation))
        if(from_prev):
            self.input_from_prev=True
        else:
            self.input_from_prev = False
        self.e_ts=K.int_shape(self.annotation)[1]
        #print("Call is OK. encoder time %d decoder time %d"%(self.e_ts,self.d_ts))
        #self.U_h=P_time_distributed_dense(self.annotation,self.Ua,self.ba)#[N,e_ts,A] also E=A (is that required ?)
        #self.U_h = K.dot(self.annotation, self.Ua)  # [N,e_ts,E] X [E,A] = [N,e_ts,A]
        return super(LuongAttentionDecoder,self).call(self.input_sequence)

    def score(self,h_t,hbar_s):
        #h_t is target hidden state N,D, must be repeated e_ts times
        #hbar_s is all source annotations N,e_ts,E
        #note that D=E
        repeated_state = K.repeat(h_t, self.e_ts)  # N,e_ts,D
        if(self.mode=='dot'):
            score=repeated_state*hbar_s #[N,e_ts,D]*[N,e_ts,E]=N,e_ts,D
        elif(self.mode=='concat'):
            score=K.concatenate([repeated_state,hbar_s]) #N,e_ts,(E+D)
        print("score shape ", K.int_shape(score))
        a_t=K.dot(score,self.Wa) #N,e_ts,1
        a_t=K.softmax(a_t)#N,e_ts,1
        return a_t

    def step(self, inputs, states):
        #print("Step:inputs shape ",K.int_shape(inputs))
        ytm, stm = states  # states=[N,D],[N,D] take input from previous output
        #enc_hidden=dec_hidden
        i_t=self.call_input_gate(ytm,stm) #N,D
        f_t=self.call_forget_gate(ytm,stm) #N,D
        cbar_t=self.call_update_gate(ytm,stm) #N,D
        O_t=self.call_output_gate(ytm,stm) #N,D
        c_t=f_t*stm+i_t*cbar_t #[N,D]*[N,D]+[N,D]*[N,D]
        h_t=K.tanh(c_t)*O_t #N,D
        '''_____Computation of context vector starts ____'''
        #repeat the hidden state over all timesteps of encoder
        self.a_t=self.score(h_t,self.annotation)
        print("a_t shape ",K.int_shape(self.a_t))
        self.c_t = K.batch_dot(self.a_t, self.annotation, axes=1)  # [N,e_ts,1] X [N,e_ts,E] = [N,1,E]
        self.c_t = K.squeeze(self.c_t, 1)  # [N,E]
        print("c_t shape ", K.int_shape(self.c_t))
        '''_____Computation of context vector ends ____'''

        '''_____Computation of hbar_t starts ____'''
        self.hbar_t=K.concatenate([self.c_t,h_t])#concat N,E and N,D gives N,(E+D)
        self.hbar_t=K.tanh(K.dot(self.hbar_t,self.Wc))#[N,E+D]*[E+D,O]=[N,O]
        self.hbar_t=K.softmax(self.hbar_t)
        if(self.return_alphas):
            return self.a_t,[self.hbar_t,self.c_t]
        else:
            return self.hbar_t,[self.hbar_t,self.c_t]#[N,O],states=[N,O],[N,D]

    #return self.S_t=[N,D]
    def call_input_gate(self,ytm,stm):
        Iiw=K.dot(ytm,self.Wkp)#[N,O] X [O,D] = [N,D]
        Irw=K.dot(stm,self.Urp)#[N,D] X [D,D] = [N,D]
        I=K.sigmoid(Iiw+Irw+self.bp)#[N,D]+[N,D]+[D] = [N,D]
        return I
        #print("Proposal gate output ", K.int_shape(self.S_t))

    # self.r=returns [N,D]
    def call_forget_gate(self,ytm,stm):
        Riw=K.dot(ytm,self.Wkr)#[N,O] X [O,D] = [N,D] #previous predicted token
        Rrw=K.dot(stm,self.Urr)#[N,D] X [D,D] = [N,D]
        R=Riw+Rrw+self.br#[N,D]+[N,D]+[D] = [N,D]
        F=K.sigmoid(R)#[N,D]
        return F
        #print("Reset/Forget gate output ",K.int_shape(self.r))

    #returns self.Z_t=[N,D]
    def call_update_gate(self,ytm,stm):
        Ziw=K.dot(ytm,self.Wkz)#[N,O] X [O,D] = [N,D] #previous predicted token
        Zrw=K.dot(stm,self.Urz)#[N,D] X [D,D] = [N,D]
        z=Ziw+Zrw+self.bz#[N,D]+[N,D]+[D] = [N,D]
        C_=K.tanh(z)#[N,D]
        return C_
        #print("Update gate output ", K.int_shape(self.Z_t))

    #return self.Y_t = [N,D]
    def call_output_gate(self,ytm,stm):
        Oiw=K.dot(ytm,self.Wko)#[N,O] X [O,D] = [N,D]
        Orw=K.dot(stm,self.Uro)#[N,D] X [D,D] = [N,D]
        O=K.sigmoid(Oiw+Orw+self.bo)#[N,D]
        return O
        #print("Output gate output ", K.int_shape(self.Y_t))#[N,O] ?

    def get_initial_state(self, inputs):
        #print('get_initial_state:inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = K.tanh(K.dot(inputs[:, 0], self.Ws))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.nodes,
            'return_probabilities': self.return_alphas
        }
        base_config = super(LuongAttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_alphas:
            return (None, self.d_ts, self.e_ts)
        else:
            return (None, self.d_ts, self.output_dim)

class WindowTimeDistributed(Layer):
    def __init__(self,nbfilters,window_size,stride,**kwargs):
        self.window_size=window_size
        self.stride=stride
        self.output_dim=nbfilters
        super(WindowTimeDistributed,self).__init__(**kwargs)

    def build(self, input_shape):
        #input is a 3D tensor [N,TS,H]
        self.input_dim=input_shape[-1]
        self.input_ts=input_shape[-2]
        self.kernel=K.constant(1,shape=(self.window_size,self.input_dim,self.output_dim),name="kernel")
        #self.kernel=self.add_weight("kernel",shape=(self.window_size,self.input_dim,self.output_dim),initializer='uniform')
        print("Input timesteps %d, Dimension %d"%(self.input_ts,self.input_dim))
        self.built=True

    def call(self,inputs,**kwargs):
        print("Calling Windowed Time Distributed")
        zeros = tf.constant(0, dtype=tf.float32, shape=[1])
        self.mask=tf.cast(tf.not_equal(inputs,zeros),tf.float32)
        self.conv_output=tf.tanh(tf.nn.convolution(inputs,self.kernel,strides=[self.stride],padding='SAME'))
        self.conv_output=self.conv_output*self.mask
        # exp=tf.exp(self.conv_output)
        # self.conv_output=(exp-1)/(exp+1)
        # self.conv_output = K.conv1d(inputs,self.kernel,strides=self.stride,padding='valid')
        return self.conv_output

    def compute_output_shape(self, input_shape):
        print("Output shape is")
        #output_ts=((self.input_ts-self.window_size)/self.stride)+1
        return (None,self.input_ts,self.output_dim)

class M2NRNN(Recurrent):
    def __init__(self,nodes,window_size,stride=1,**kwargs):
        self.units = nodes
        super(M2NRNN,self).__init__(**kwargs)
        self.nodes=nodes
        self.window_size=window_size+1
        self.stride=stride
        print("Initiated")

    def build(self, input_shape):
        print("Built: Input shape ",input_shape)
        self.input_dim=input_shape[-1]
        self.input_ts=input_shape[-2]
        self.states = [None,0]
        self.dim_in_block = self.input_dim * self.window_size
        self.W=self.add_weight('W',(self.input_dim,1),initializer='uniform')
        #self.U=self.add_weight('U',(self.nodes,self.nodes),initializer='uniform')
        self.Wk=self.add_weight('Wk',(self.window_size,self.nodes),initializer='uniform')
        self.Ws=self.add_weight('Ws',(self.dim_in_block,self.nodes),trainable=False,initializer='uniform')
        self.built=True
        print("Built complete")

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.blocks=self.input_ts/self.window_size
        self.dummy_sequence=[]
        i=0
        end=0
        while (end < self.input_ts):
            end=i+self.window_size
            if(end<self.input_ts):
                sub_sequence=inputs[:,i:end,:]#N,ws,I
                flat=K.reshape(sub_sequence,[-1,self.dim_in_block])#N,ws.I
                self.dummy_sequence.append(flat)
                i=i+self.stride
        print("Step: sliding done. shape ",len(self.dummy_sequence))
        self.dummy_sequence=tf.transpose(self.dummy_sequence,[1,0,2])
        self.out_ts=K.int_shape(self.dummy_sequence)[1]
        print("step:Dummy sequence shape ",K.int_shape(self.dummy_sequence))
        return super(M2NRNN,self).call(self.dummy_sequence)

    def step(self,inputs,states):
        print("Step: Inputs shape",K.int_shape(inputs))
        stm=states[0] #N,H
        time=states[1]
        input_t=K.reshape(inputs,[-1,self.window_size,self.input_dim])
        focus_sequence=input_t#N,ws,I
        focus_embedding=K.dot(focus_sequence,self.W)#N,ws,1
        focus_embedding=K.squeeze(focus_embedding,-1)#N,ws
        #prev_out=K.tanh(K.dot(stm,self.U))
        focus_ff=K.sigmoid(K.dot(focus_embedding,self.Wk))#N,H
        out_= focus_ff
        time=time+2
        return out_,[out_,time]

    def get_initial_state(self, inputs):
        print("Initial state : Input shape ",K.int_shape(inputs))
        input_0=tf.transpose(inputs,[1,0,2])[0]
        s0=K.dot(input_0,self.Ws)
        print("Initial state : State shape ",K.int_shape(s0))
        return [s0,1]

    def compute_output_shape(self, input_shape):
        return (None,self.out_ts,self.nodes)

class PSMAttentionEncoder(Recurrent):

    def __init__(self,blocksize,block_dim,stride=1,time_squeezed=True,**kwargs):
        super(PSMAttentionEncoder,self).__init__(**kwargs)
        self.block_size=blocksize
        self.block_nodes=block_dim
        self.time_squeezed=False
        self.stride=stride

    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.input_dim = input_shape[-1]
        print("Building: Input shape ",input_shape)
        self.w_init=self.add_weight('init',shape=[self.input_dim,self.block_nodes],initializer='uniform',trainable=False)
        self.block_lstm=Bidirectional(LSTM(self.block_nodes,return_sequences=False,return_state=True),merge_mode='sum')
        #self.W=self.add_weight('kernel',shape=[self.input_dim,self.block_nodes],initializer='uniform')
        #self.U=self.add_weight('recurrent',shape=[self.block_nodes,self.block_nodes],initializer='uniform')
        self.states=[None,None]
        self.built=True

    def get_initial_state(self, inputs):
        #get first step
        input_0=inputs[:,0:1,:]
        input_0=K.reshape(input_0,[-1,self.block_size,self.input_dim])
        input_0=tf.transpose(input_0,[1,0,2])[0]
        state=K.dot(input_0,self.w_init)
        return [state,state]


    def call(self, inputs, mask=None, training=None, initial_state=None):
        #print("Calling: Input shape",K.int_shape(inputs))
        self.nb_blocks=int(math.ceil((self.timesteps-self.block_size)/self.stride))
        print("Number of blocks %d"%self.nb_blocks)
        self.new_inputs=[]
        start=0
        for b in range(self.nb_blocks):
            end=min(start+self.block_size,self.timesteps)
            #print('Block range %d to %d'%(start,end))
            block_array=tf.slice(inputs,[0,start,0],[-1,end-start,-1])#inputs[:,start:end,:]
            block_shape = K.int_shape(block_array)
            #print("Sliced block shape ",block_shape)
            self.block_timesteps=block_shape[1] #4
            #block_dim=block_shape[2] #1 ==self.input_dim
            #print("Inside Block: timesteps %d dim %d"%(self.block_timesteps,self.input_dim))
            block_array=K.reshape(block_array,[-1,self.block_timesteps*self.input_dim])
            #print("Block from %d to %d, timesteps %d"%(start,end,self.block_timesteps)," shape ",K.int_shape(block_array))
            self.new_inputs.append(block_array)
            start=start+self.stride
        self.new_inputs=tf.convert_to_tensor(self.new_inputs)
        self.new_inputs=tf.transpose(self.new_inputs,[1,0,2])
        print("New Inputs shape",K.int_shape(self.new_inputs))
        return super(PSMAttentionEncoder,self).call(self.new_inputs)


    def step(self, inputs, states):
        #inputs N,block_ts*input_dim
        print("Step: inputs ",K.int_shape(inputs))
        step_input=K.reshape(inputs,[-1,self.block_timesteps,self.input_dim])#N,timesteps,input_dim
        step_output=self.block_lstm(step_input)#N,block_dim
        #print("Step: output shape ",K.int_shape(step_output[0]))
        return step_output[0],[step_output[1],step_output[2]]

    def compute_output_shape(self, input_shape):
        return (None,self.nb_blocks,self.block_nodes)

class PSMEncoder(Recurrent):

    def __init__(self,blocksize,block_dim,stride=-1,time_squeezed=True,**kwargs):
        super(PSMEncoder,self).__init__(**kwargs)
        self.block_size=blocksize
        self.block_nodes=block_dim
        self.time_squeezed=False
        if(stride<0):
            stride=blocksize
        self.stride=stride

    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.input_dim = input_shape[-1]
        print("Building: Input shape ",input_shape)
        self.w_init=self.add_weight('init',shape=[self.input_dim,self.block_nodes],initializer='uniform',trainable=False)
        self.block_lstm=Bidirectional(LSTM(self.block_nodes,return_sequences=False,return_state=True),merge_mode='sum')
        self.Wk=self.add_weight('kernel',shape=[self.input_dim,self.block_nodes],initializer='uniform')#I,H
        self.U=self.add_weight('recurrent',shape=[self.block_nodes,self.block_nodes],initializer='uniform')
        self.states=[None,None]
        self.built=True

    def get_initial_state(self, inputs):
        #get first step
        input_0=inputs[:,0:1,:]
        input_0=K.reshape(input_0,[-1,self.block_size,self.input_dim])
        input_0=tf.transpose(input_0,[1,0,2])[0]
        state=K.dot(input_0,self.w_init)
        return [state,state]


    def call(self, inputs, mask=None, training=None, initial_state=None):
        #print("Calling: Input shape",K.int_shape(inputs))
        self.nb_blocks=int(math.ceil((self.timesteps-self.block_size)/self.stride))+1
        print("Number of blocks %d"%self.nb_blocks)
        self.new_inputs=[]
        start=0
        for b in range(self.nb_blocks):
            end=min(start+self.block_size,self.timesteps)
            #print('Block range %d to %d'%(start,end))
            block_array=tf.slice(inputs,[0,start,0],[-1,end-start,-1])#inputs[:,start:end,:]
            block_shape = K.int_shape(block_array)
            #print("Sliced block shape ",block_shape)
            self.block_timesteps=block_shape[1] #4
            #block_dim=block_shape[2] #1 ==self.input_dim
            #print("Inside Block: timesteps %d dim %d"%(self.block_timesteps,self.input_dim))
            block_array=K.reshape(block_array,[-1,self.block_timesteps*self.input_dim])
            #print("Block from %d to %d, timesteps %d"%(start,end,self.block_timesteps)," shape ",K.int_shape(block_array))
            self.new_inputs.append(block_array)
            start=start+self.stride
        self.new_inputs=tf.convert_to_tensor(self.new_inputs)
        self.new_inputs=tf.transpose(self.new_inputs,[1,0,2])
        print("New Inputs shape",K.int_shape(self.new_inputs))
        return super(PSMEncoder,self).call(self.new_inputs)

    def innerRNN(self,input_sequence):
        #input_sequence=N,timesteps,input_dim
        steps=K.int_shape(input_sequence)[1]
        sequence_time_major=tf.transpose(input_sequence,[1,0,2])
        ytm=K.dot(sequence_time_major[0],self.Wk)#N,H
        output=[]
        for s in range(steps):
            out_kernel=K.dot(sequence_time_major[s],self.Wk)#N,I x I,H=N,H
            out_recurrent=K.dot(ytm,self.U)#N,H
            out=K.tanh(out_kernel+out_recurrent)#N,H
            output.append(out)
            ytm=out
        return output



    def step(self, inputs, states):
        #inputs N,block_ts*input_dim
        print("Step: inputs ",K.int_shape(inputs))
        step_input=K.reshape(inputs,[-1,self.block_timesteps,self.input_dim])#N,timesteps,input_dim
        step_output=self.innerRNN(step_input)#block_timesteps,N,block_dim
        last_step=step_output[-1]
        #print("Step: output shape ",K.int_shape(step_output[0]))
        return last_step,[last_step,last_step]

    def compute_output_shape(self, input_shape):
        return (None,self.nb_blocks,self.block_nodes)

class RNNSoftWindow(Recurrent):
    def __init__(self,nodes,nbmixtures,**kwargs):
        self.K=nbmixtures
        self.nodes=5*self.K
        super(RNNSoftWindow,self).__init__(**kwargs)

    def build(self, input_shape):
        print('Building: Input shape ',input_shape)
        self.timesteps=input_shape[0][1]
        self.input_dim=input_shape[0][2]
        self.Nc=input_shape[1][2]
        self.states=[None,None,-1]
        self.Wh=self.add_weight('Wh',[self.input_dim,3*self.K],initializer='uniform')
        self.bh=self.add_weight('bh',[3*self.K],initializer='uniform')
        self.Wk=self.add_weight('Wk',[self.input_dim,self.nodes],initializer='uniform')
        self.Wu=self.add_weight('Wu',[self.nodes,self.nodes],initializer='uniform')
        self.Ww=self.add_weight('Ww',[self.Nc,self.nodes],initializer='uniform')
        self.b=self.add_weight('b',[self.nodes],initializer='uniform')
        self.init_w=self.add_weight('init_w',[self.input_dim,self.K],initializer='uniform',trainable=False)
        print("Building Complete")
        super(RNNSoftWindow,self).build(input_shape)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.input_sequence=inputs[0]
        self.character_sequence=inputs[1]#N,U,Nc One Hot
        self.U=K.int_shape(self.character_sequence)[1]
        self.character_sequence=tf.transpose(self.character_sequence,[1,0,2])#U,N,Nc
        self.parameters=K.dot(self.input_sequence,self.Wh)+self.bh #N,T,D x D,3K = N,T,3K
        self.alpha=tf.transpose(K.exp(self.parameters[:,:,:self.K]),[1,0,2])#T,N,K
        self.beta=tf.transpose(K.exp(self.parameters[:,:,self.K:self.K*2]),[1,0,2])#T,N,K
        self.L=tf.transpose(self.parameters[:,:,self.K*2:self.K*3],[1,0,2])#T,N,K
        print("L:",K.int_shape(self.L))
        print("alpha ",K.int_shape(self.alpha))
        print("beta ",K.int_shape(self.beta))
        return super(RNNSoftWindow,self).call(self.input_sequence)

    def compute_soft_window(self,states):
        t = states[-1]+1
        #K.print_tensor(t,"internal")
        Ltm = states[-2]  # N,K
        Lt = Ltm + K.exp(self.L[t])  # N,K
        Wt = 0
        for u in range(self.U):
            phi_t_u = tf.reduce_sum(self.alpha[t] * K.exp(-self.beta[t] * (Lt - u) ** 2), axis=-1,keepdims=True)  # N,1
            Cu = self.character_sequence[u]  # N,Nc
            Wt = Wt + tf.multiply(phi_t_u, Cu)
        return Wt, [Wt, Lt, t ]

    def step(self,inputs,states):
        stm=states[0]
        Wt,[Wt,Lt,t]=self.compute_soft_window(states)
        yt=K.tanh(K.dot(inputs,self.Wk)+K.dot(stm,self.Wu)+K.dot(Wt,self.Ww)+self.b)
        #t=t+1
        return yt,[yt,Lt,t]

    def get_initial_state(self, inputs):#Must be [N,D N,K, int]
        print("Computing Initial State: inputs shape",K.int_shape(inputs))#N,T,D
        x=tf.transpose(inputs,[1,0,2])[0]#N,I
        yt0=K.dot(x,self.Wk)#N,D
        lt0=K.dot(x,self.init_w)#N,K
        return [yt0,lt0,-1]

    def compute_output_shape(self, input_shape):
        return (None,self.timesteps,self.nodes)








'''
ts=10
f=3
input_layer=Input((ts,f))
lstm1=LSTM(8,return_sequences=True,return_state=False)(input_layer)
pyrm=Lambda(pyramid,output_shape=pyramid_output)(lstm1)
m=Model(input_layer,pyrm)
m.summary()
'''
# def find_top_n_from_ts(prob,n,inds):
#     
# def beam_search_decoding(out_prob):
#     #out_prob is an array [ts,Nc], we pick top choice
#     ts=out_prob.shape[0]
#     for t in ts