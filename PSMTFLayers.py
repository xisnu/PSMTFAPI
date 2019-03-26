from __future__ import print_function

from PSMUtils import *


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

    def compute_output_shape(self, input_shape):
        return [None,self.num_outputs]


# layer = MyDenseLayer(10)
# print(layer(tf.zeros([10, 5])))
# print(layer.trainable_variables)
# print(K.int_shape(layer))



class SimpleLoop:
    def __init__(self,ts,nbfeat):
        self.model=tf.Graph()
        self.ts=ts
        self.nbfeat=nbfeat

    def build(self):
        with self.model.as_default():
            self.model_input=tf.placeholder(tf.int32,[None,self.ts,self.nbfeat])
            self.output=self.loop(self.model_input)

    def for_each_timestep(self,po, ci):
        po=tf.split(po,[1,1],1)
        print('PO shape ',get_layer_shape(po[0]))
        y0=po[0]-po[1]
        y1=ci+y0
        print("y1 shape ",get_layer_shape(y1))
        y2=ci-y0
        print("y2 shape ", get_layer_shape(y2))
        ret=tf.concat([y1,y2],1)
        print("ret shape ",get_layer_shape(ret))
        return ret

    def loop(self,X):
        X_tm=tf.transpose(X,[1,0,2])
        init=tf.concat([X_tm[0],X_tm[-1]],1,name='init_x')
        print("Init shape ", get_layer_shape(init))
        res=tf.scan(self.for_each_timestep,X_tm,initializer=init)
        res=tf.transpose(res,[1,0,2])
        return res

    def run(self,data):
        with tf.Session(graph=self.model) as sess:
            feed={self.model_input:data}
            out=sess.run(self.output,feed_dict=feed)
            print(out)

class SimpleNestedLoop:
    def __init__(self,ts,nbfeat):
        self.model=tf.Graph()
        self.ts=ts
        self.nbfeat=nbfeat
        self.Ws=4
        self.t=0

    def build(self):
        with self.model.as_default():
            self.model_input=tf.placeholder(tf.float32,[None,self.ts,self.nbfeat])
            self.step_output=[]
            self.output=self.loop_over_timesteps(self.model_input)

    def for_each_step(self,t,ts):
        #data[index]=data[index]+1
        index=t+1
        return [index,ts]

    def condition(self,t,ts):
        return tf.less(t,ts)

    def loop_over_timesteps(self,X):#X=ts,N,F
        input_sequence=tf.transpose(X,[1,0,2])
        t=0
        self.ts=tf.constant(get_layer_shape(input_sequence)[0],dtype=tf.int32)
        r=tf.while_loop(self.condition,self.for_each_step,[t,self.ts],return_same_structure=True)
        return r

    def run(self,data):
        with tf.Session(graph=self.model) as sess:
            feed={self.model_input:data}
            out=sess.run([self.step_output],feed_dict=feed)
            print(out)


class psmRNNCell(object):
    def __init__(self,nodes,name):
        self.nodes=nodes
        self.name=name

    def build(self,input_shape):
        self.input_shape=input_shape
        self.input_dim = self.input_shape[-1]
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            self.Wk = tf.get_variable('Wk', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wk',self.Wk)
            self.Wr = tf.get_variable('Wr', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Wr', self.Wr)
            self.Ir = tf.get_variable('Initial_state',shape=[self.input_dim,self.nodes],trainable=False)
        print("RNN built: input shape ", self.input_shape)

    def rnn_step(self,previous_output,step_input):
        ci_out=tf.matmul(step_input,self.Wk,name='kernel_mult')
        po_out=tf.matmul(previous_output,self.Wr,name='recurrent_mult')
        step_output=tf.tanh(tf.add(ci_out,po_out))
        return step_output

    def loop_over_timestep(self,input):
        input_tm=tf.transpose(input,[1,0,2])
        initial_state=tf.matmul(input_tm[0],self.Ir)
        output_tm=tf.scan(self.rnn_step,input_tm,initializer=initial_state)
        output=tf.transpose(output_tm,[1,0,2])
        return output


def psmRNN(nodes,input,name,return_time_major=False,return_sequence=True):
    rnncell=psmRNNCell(nodes,name)
    input_shape=get_layer_shape(input)
    rnncell.build(input_shape)
    output=rnncell.loop_over_timestep(input)
    if(not return_sequence):
        output = tf.transpose(output, [1, 0, 2])
        output = output[-1]
        return output
    if(return_time_major):
        output=tf.transpose(output,[1,0,2])
    return output

class psmLSTMCell(object):

    def __init__(self,nodes,name):
        self.nodes=nodes
        self.name=name

    def build(self,input_shape):
        self.input_shape = input_shape
        self.input_dim = self.input_shape[-1]
        self.nb_params=1
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            #input gate
            self.Wi = tf.get_variable('Wi', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wi',self.Wi)
            self.Ui = tf.get_variable('Ui', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Ui', self.Ui)
            self.bi = tf.get_variable('bi',shape=[self.nodes])
            tf.summary.histogram('bi',self.bi)
            self.nb_params+=(self.input_dim*self.nodes)+(self.nodes*self.nodes)+self.nodes
            #forget gate
            self.Wf = tf.get_variable('Wf', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wf', self.Wf)
            self.Uf = tf.get_variable('Uf', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Uf', self.Uf)
            self.bf = tf.get_variable('bf', shape=[self.nodes])
            tf.summary.histogram('bf', self.bf)
            self.nb_params += (self.input_dim * self.nodes) + (self.nodes * self.nodes) + self.nodes
            # update/proposal gate
            self.Wp = tf.get_variable('Wp', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wp', self.Wp)
            self.Up = tf.get_variable('Up', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Up', self.Up)
            self.bp = tf.get_variable('bp', shape=[self.nodes])
            tf.summary.histogram('bp', self.bp)
            self.nb_params += (self.input_dim * self.nodes) + (self.nodes * self.nodes) + self.nodes
            #output gate
            self.Wo = tf.get_variable('Wo', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wo', self.Wo)
            self.Uo = tf.get_variable('Uo', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Uo', self.Uo)
            self.bo = tf.get_variable('bo', shape=[self.nodes])
            tf.summary.histogram('bo', self.bo)
            self.nb_params += (self.input_dim * self.nodes) + (self.nodes * self.nodes) + self.nodes
            self.Ir = tf.get_variable('Initial_state',shape=[self.input_dim,self.nodes],trainable=False)
        print("LSTM built: input shape ", self.input_shape," total parameters %d"%self.nb_params)

    def rnn_step(self,previous_output,step_input):
        previous_output=tf.split(previous_output,[self.nodes,self.nodes],1)
        Stm1=previous_output[1]
        Ytm1=previous_output[0]
        #input gate
        i_k=tf.matmul(step_input,self.Wi,name='input_gate_kernel')
        i_r=tf.matmul(Stm1,self.Ui,name='input_gate_recurrent')
        self.i_output=tf.sigmoid(tf.add(i_k,i_r)+self.bi)
        #forget gate
        f_k = tf.matmul(step_input, self.Wf, name='forget_gate_kernel')
        f_r = tf.matmul(Stm1, self.Uf, name='forget_gate_recurrent')
        self.f_output = tf.sigmoid(tf.add(f_k, f_r)+self.bf)
        #update/proposal gate
        p_k = tf.matmul(step_input, self.Wp, name='update_gate_kernel')
        p_r = tf.matmul(Stm1, self.Up, name='update_gate_recurrent')
        self.p_output = tf.tanh(tf.add(p_k, p_r)+self.bp)#St_
        # output gate
        o_k = tf.matmul(step_input, self.Wo, name='output_gate_kernel')
        o_r = tf.matmul(Stm1, self.Uo, name='output_gate_recurrent')
        self.o_output = tf.sigmoid(tf.add(o_k, o_r)+self.bo)
        #Computing new state
        self.St=(Stm1*self.f_output)+(self.i_output*self.p_output)
        #compute output
        self.Yt=tf.tanh(self.St)*self.o_output
        output_pack=tf.concat([self.Yt,self.St],1)
        print("LSTM: All computed, Returning ",get_layer_shape(output_pack))
        return output_pack

    def loop_over_timestep(self,input):
        print("LSTM: initiate loop: ",get_layer_shape(input))
        input_tm=tf.transpose(input,[1,0,2])
        initial_state=tf.matmul(input_tm[0],self.Ir,name='Mult_Initial_State')
        init=tf.concat([initial_state,initial_state],1)
        output_tm=tf.scan(self.rnn_step,input_tm,initializer=init)
        output=tf.transpose(output_tm,[1,0,2])
        return output

def psmLSTM(nodes,input,name,return_time_major=False,return_sequence=True):
    rnncell=psmLSTMCell(nodes,name)
    input_shape=get_layer_shape(input)
    rnncell.build(input_shape)
    output=rnncell.loop_over_timestep(input)
    if(not return_sequence):
        output = tf.transpose(output, [1, 0, 2])
        output = output[-1]
        return output
    if(return_time_major):
        output=tf.transpose(output,[1,0,2])
    return output

class psmBadhanauAttention(psmLSTMCell):
    #encoder gives encoding of shape N,ets,E
    #decoder is given input of shape N,dts,Nc where Nc is number of character classes
    def __init__(self,nodes,name,use_previous_output=True):
        self.attention_dim=nodes
        self.use_previous_output=use_previous_output
        self.name=name
        super(psmBadhanauAttention,self).__init__(nodes,name)

    def build(self,input_shape):
        #input shape is a list of shapes [shape_of_encoding,shape_of_decoderinput]
        encoding_shape=input_shape[0]
        if isinstance(encoding_shape[1],int):
            self.ets=encoding_shape[1]
        else:
            self.ets=encoding_shape[1].value
        self.annotation_dim=encoding_shape[-1]
        decoderinput_shape=input_shape[1]
        if isinstance(decoderinput_shape[-1],int):
            self.output_dim=decoderinput_shape[-1]
        else:
            self.output_dim = decoderinput_shape[-1].value
        self.dts=decoderinput_shape[1]
        print("______________________________")
        print("Annotation dim:",self.annotation_dim," Output dim:",self.output_dim," Attention dim:",self.attention_dim)
        print("Encoder T:",self.ets," Decoder T:",self.dts)
        super(psmBadhanauAttention,self).build(decoderinput_shape)
        self.encoder_dim=encoding_shape[-1]
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            #weights for alignment model a()---------#
            self.Ua=tf.get_variable('Ua',shape=[self.encoder_dim,self.attention_dim],dtype=tf.float32)#E,A
            tf.summary.histogram('Ua',self.Ua)
            self.Wa=tf.get_variable('Wa',shape=[self.nodes,self.attention_dim],dtype=tf.float32)#D,A
            tf.summary.histogram('Wa',self.Wa)
            self.Ba=tf.get_variable('Ba',shape=[self.attention_dim],dtype=tf.float32)
            tf.summary.histogram('Ba',self.Ba)
            self.Va=tf.get_variable('Va',shape=[self.attention_dim,1],dtype=tf.float32)
            tf.summary.histogram('Va', self.Va)
            #----------------------------------------#
            self.W = tf.get_variable('W',shape=[self.output_dim,self.attention_dim],dtype=tf.float32)
            tf.summary.histogram('W',self.W)
            #weights for context vector
            self.Cf = tf.get_variable('Cf',shape=[self.annotation_dim,self.attention_dim],dtype=tf.float32)
            tf.summary.histogram('Cf',self.Cf)
            self.Ci = tf.get_variable('Cp', shape=[self.annotation_dim, self.attention_dim], dtype=tf.float32)
            tf.summary.histogram('Ci', self.Ci)
            self.C = tf.get_variable('C', shape=[self.annotation_dim,self.attention_dim],dtype=tf.float32)
            tf.summary.histogram('C',self.C)
            self.Co = tf.get_variable('Co', shape=[self.annotation_dim, self.attention_dim], dtype=tf.float32)
            tf.summary.histogram('Co', self.Co)
            #for output gate
            self.Wo_Nc=tf.get_variable('Wo_Nc', shape=[self.attention_dim, self.output_dim], dtype=tf.float32)
            tf.summary.histogram('Wo_Nc', self.Wo_Nc)
            print("Attention built")
            print("______________________________")

    def compute_alignment_annotation(self,annotation):
        #computing Ua.hj with annotation N,ets,E
        self.h=annotation #N,ets,E
        self.a_hj=tf.tensordot(annotation,self.Ua,axes=[2,0],name='mul_a_hj')#N,ets,E x E,A = N,ets,A
        self.a_hj=tf.transpose(self.a_hj,[1,0,2]) #making this time major (ets,N,A) for using in loop of timesteps

    def compute_alignment_state(self,Stm):
        a_stm=tf.matmul(Stm,self.Wa,name='mul_a_stm')#N,D x D,A = N,A
        return a_stm

    def rnn_step(self,previous_output,step_input):
        split_dim=[self.output_dim,self.attention_dim,self.ets]
        print("Split by: ",split_dim)
        previous_output=tf.split(previous_output,split_dim,1)
        Ytm,Stm=previous_output[0],previous_output[1] #Ytm=N,D Ytm=N,D
        if(not self.use_previous_output):
            Ytm=step_input
            print("\tUsing Decoder Input")
        #print("Ytm:",get_layer_shape(Ytm)," Stm",get_layer_shape(Stm))
        #now compute Wa.Stm
        self.a_stm=self.compute_alignment_state(Stm)
        #now compute e and alpha
        e=tf.tensordot(tf.tanh(self.a_stm+self.a_hj+self.Ba),self.Va,[-1,0],'dot_eij')
        self.alpha=tf.transpose(e,[1,0,2])#N,ets,1
        self.alpha = tf.nn.softmax(self.alpha,axis=-2)
        print("alpha ", get_layer_shape(self.alpha))
        #now compute context vector
        self.c=tf.reduce_sum(self.h*self.alpha,axis=1)#N,E
        self.alpha=tf.squeeze(self.alpha,axis=-1)
        #print("context ",get_layer_shape(self.c))
        #now compute LSTM gates
        #forget / reset gate
        f_k = tf.matmul(Ytm, self.Wf, name='forget_gate_kernel')#N,Nc x Nc,A = N,A
        f_r = tf.matmul(Stm, self.Uf, name='forget_gate_recurrent')#N,A x A,A = N,A
        f_c = tf.matmul(self.c, self.Cf, name='forget_gate_context')#N,E x E,A = N,A
        self.r = tf.sigmoid(f_k+f_r+f_c + self.bf) #N,A
        #print("Forget: ",get_layer_shape(self.r))
        #input gate
        i_k = tf.matmul(Ytm, self.Wi, name='update_gate_kernel')#N,Nc x Nc,A = N,A
        i_r = tf.matmul(Stm, self.Ui, name='update_gate_recurrent')#N,A x A,A = N,A
        i_c = tf.matmul(self.c, self.Ci, name='update_gate_context')# N,E x E,A = N,A
        self.z = tf.tanh(i_k + i_r + i_c + self.bi) #N,A
        #print("Input: ", get_layer_shape(self.z))
        # new state proposal
        ns_op = tf.matmul(Ytm, self.W,name='new_state_output_proposal')
        ns_sp = tf.matmul(Stm*self.r,self.Up, name='new_state_state_proposal')
        ns_cp = tf.matmul(self.c,self.C, name='new_state_context_proposal')
        St_= tf.tanh(ns_op+ns_sp+ns_cp) #N,A x A,A = N,A
        # computing new state
        self.St = (1-self.z)*Stm + self.z * St_
        #print("New State: ", get_layer_shape(self.St))
        #output gate
        o_k= tf.matmul(Ytm, self.Wo, name='output_gate_kernel')#N,Nc x Nc,A = N,A
        o_r = tf.matmul(Stm,self.Uo, name='output_gate_recurrent')#N,A x A,A = N,A
        o_c = tf. matmul(self.c, self.Co, name='output_gate_context')#N,E x E,A =N,A
        self.o = tf.tanh(o_k + o_r + o_c + self.bo)*self.St #N,A * N,A =N,A
        #print("Output: ", get_layer_shape(self.o))
        #output gate should produce Nc
        self.Yt = tf.nn.softmax(tf. matmul(self.o,self.Wo_Nc,name='output_to_Nc'))#N,A x A,Nc = N,Nc
        #print("Output Nc: ", get_layer_shape(self.Yt))
        packed_output=tf.concat([self.Yt,self.St,self.alpha],axis=1)
        #print("Pack: ",get_layer_shape(packed_output))
        return packed_output

    def loop_over_timestep(self,input):
        #there are two inputs encoder_out N,ets,E and decoder_in N,dts,Nc
        annotation=input[0]
        decoder_input=input[1]
        decoder_input_tm=tf.transpose(decoder_input,[1,0,2])#dts,N,Nc
        self.compute_alignment_annotation(annotation)
        #now create initial state
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            W_iy=tf.get_variable('W_init_y',shape=[self.output_dim,self.attention_dim],dtype=tf.float32,trainable=False)
            W_ia = tf.get_variable('W_init_alpah', shape=[self.output_dim, self.ets], dtype=tf.float32, trainable=False)
        ym1 = decoder_input_tm[0]#N,Nc
        sm1 =  tf.matmul(ym1,W_iy,name='initial_state_compute_y')#N,Nc x Nc,D = N,D
        alpham1= tf.matmul(ym1,W_ia,name='initial_state_compute_alpha')#N,Nc x Nc,E =N,E
        initial_state=tf.concat([ym1,sm1,alpham1],axis=1)
        #print("Initial State:",get_layer_shape(initial_state))
        comb_output_tm=tf.scan(self.rnn_step,decoder_input_tm,initializer=initial_state)# [[N,dts,Nc][N,dts,D]]
        comb_output = tf.transpose(comb_output_tm,[1,0,2])#reset time major order
        split_output = tf.split(comb_output,[self.output_dim,self.attention_dim,self.ets],axis=2)#split output and state
        output,state,alpha = split_output[0],split_output[1],split_output[2]
        return output,state,alpha

def psmAttentionDecoder(nodes,encoder_output,decoder_input,use_previous_output=True):
    attentionlayer = psmBadhanauAttention(nodes, 'badhanau',use_previous_output=use_previous_output)
    encoder_shape = encoder_output.shape
    decoder_input_shape = decoder_input.shape
    input_shape = [encoder_shape, decoder_input_shape]
    attentionlayer.build(input_shape)
    attentionlayer.compute_alignment_annotation(encoder_output)
    output,state,alpha=attentionlayer.loop_over_timestep([encoder_output,decoder_input])
    print("______________________________")
    print("Output: ",get_layer_shape(output))
    print("State: ", get_layer_shape(state))
    print("Alpha:",get_layer_shape(alpha))
    return output,state,alpha

def psmCrossEntropyLoss(predict,labels):
    loss=tf.reduce_mean(tf.reduce_sum(labels*tf.log(predict),axis=-1))
    return loss

class psmTransducer(object):
    def __init__(self,nodes,name,windowsize=5):
        self.nodes=nodes
        self.name=name
        self.window_size=windowsize

    def build(self,input_shape):
        self.input_shape=input_shape
        self.input_dim = self.input_shape[-1]
        self.T=self.input_shape[-2]
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            self.Wk = tf.get_variable('Wk', shape=[self.input_dim, self.nodes])
            tf.summary.histogram('Wk',self.Wk)
            self.Wr = tf.get_variable('Wr', shape=[self.nodes, self.nodes])
            tf.summary.histogram('Wr', self.Wr)
            self.Ir = tf.get_variable('Initial_state',shape=[self.input_dim,self.nodes],trainable=False)
        print("RNN built: input shape ", self.input_shape)

    def window_step(self,previous_output,step_input):
        ci_out = tf.matmul(step_input, self.Wk, name='kernel_mult')
        po_out = tf.matmul(previous_output, self.Wr, name='recurrent_mult')
        step_output = tf.tanh(tf.add(ci_out, po_out))
        return step_output

    def rnn_step(self,previous_output,step_input):
        self.t=previous_output
        window=self.input_tm[self.t:self.t+self.window_size-1]
        tf.scan(self.window_step,window)
        return self.t+1

    def loop_over_timestep(self,input):
        self.input_tm=tf.transpose(input,[1,0,2])
        initial_state=tf.matmul(self.input_tm[0],self.Ir)
        output_tm=tf.scan(self.rnn_step,self.input_tm,initializer=initial_state)
        output=tf.transpose(output_tm,[1,0,2])
        return output

class psmRNNCell_v2(object):
    def __init__(self,nodes,input_sequence,name):
        self.nodes=nodes
        self.name=name
        self.input_sequence=tf.transpose(input_sequence,[1,0,2])#convert to time major

    def build(self):
        input_shape=get_layer_shape(self.input_sequence)
        self.input_dim=input_shape[-1]
        self.T=input_shape[0]#NOT time major
        self.W=tf.get_variable('W_'+self.name,shape=[self.input_dim,self.nodes],dtype=tf.float32)#F,H ,initializer='random_uniform'
        tf.summary.histogram('W_'+self.name,self.W)
        self.U=tf.get_variable('U_'+self.name,shape=[self.nodes,self.nodes],dtype=tf.float32)#H,H,initializer='random_uniform'
        tf.summary.histogram('U_'+self.name,self.U)
        self.bias=tf.get_variable('Bias_'+self.name,shape=[self.nodes],dtype=tf.float32)#H,initializer='random_uniform'
        self.Winit=tf.zeros([self.input_dim,self.nodes],dtype=tf.float32,name='Winit')

    def stop_loop_if(self,t,prev_out,output_sequence):
        return tf.less(t,self.T)

    def rnn_op(self,t,prev_out,output_sequence):
        current_input=self.input_sequence[t]#N,F
        current_output=tf.tanh(tf.matmul(current_input,self.W,name='input_nodes')+tf.matmul(prev_out,self.U,name='nodes_nodes')+self.bias)#N,H
        output_sequence=output_sequence.write(t,current_output)
        t=t+1
        return [t,current_output,output_sequence]

    def rnn(self):
        initial_state=tf.matmul(self.input_sequence[0],self.Winit,name='initial_state')#N,H
        output_sequence = tf.TensorArray(tf.float32, size=self.T)
        T,_,Y=tf.while_loop(self.stop_loop_if,self.rnn_op,[0,initial_state,output_sequence])
        Y=Y.concat()
        Y=tf.reshape(Y,[self.T,-1,self.nodes])
        return Y


def psmRNN_v2(nodes,input,name,return_time_major=False,return_sequence=True):
    rnncell=psmRNNCell_v2(nodes,input,name)
    rnncell.build()
    output=rnncell.rnn()
    if (not return_sequence):
        return output[-1]
    if(not return_time_major):
        output=tf.transpose(output,[1,0,2])
    return output

