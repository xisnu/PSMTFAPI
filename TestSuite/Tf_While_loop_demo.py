import tensorflow as tf

def condition(ci,max,res,po):
    return tf.less(ci,max)

def body(ci,max,res,po):
    new_sum=ci+po
    res = res.write(ci, new_sum)
    ci=ci+1
    return [ci,max,res,new_sum]

t1=tf.constant(0)
t2=tf.constant(10)
t3=tf.TensorArray(tf.int32,size=10)
t4=tf.constant(0)

res1,res2,res3,res4=tf.while_loop(condition,body,[t1,t2,t3,t4])
res3=res3.stack()

with tf.Session() as sess:
    r3,r4=sess.run([res3,res4])
    print(r3,r4)