import tensorflow as tf
import numpy as np

ts=5
feat=2
H=3

input_sequence=tf.placeholder(tf.float32,shape=[ts,None,feat])
Wk=tf.constant(1.0,shape=[feat,H])
U=tf.constant(1.0,shape=[H,H])

def condition(ci,max,inp_seq,out_seq,prev_out):
    return tf.less(ci,max)

def body(ci,max,inp_seq,out_seq,prev_out):
    current_input=inp_seq[ci]
    current_output=tf.matmul(current_input,Wk)+tf.matmul(prev_out,U)
    out_seq = out_seq.write(ci, current_output)
    ci=ci+1
    return [ci,max,inp_seq,out_seq,current_output]


output_sequence=tf.TensorArray(tf.float32,size=ts)
start=0
init_state=tf.matmul(input_sequence[0],Wk)

_,_,_,output,_=tf.while_loop(condition,body,[start,ts,input_sequence,output_sequence,init_state])
output=output.stack()


with tf.Session() as sess:
    data=np.random.random([ts,3,feat])
    print(data)
    result=sess.run([output],feed_dict={input_sequence:data})
    print(result)
