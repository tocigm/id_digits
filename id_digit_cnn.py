import tensorflow as tf
import numpy as np
class ID_DIGIT_CNN(object):
    def __init__(self,conf,dataset):
        self.conf = conf
        self.dataset = dataset
        self.NUM_CLASSES = 10

        
    def build_model(self):
        self.X = tf.placeholder(tf.float32,[None,self.dataset.max_row,self.dataset.max_col,3])
        self.y = tf.placeholder(tf.int64,[None,])
        
        with tf.variable_scope("s1_conv1") as scope:
            kernel = tf.get_variable("weights",[5,5,3,64])
            conv   = tf.nn.conv2d(self.X,kernel,[1,1,1,1],padding="SAME")
            biases = tf.get_variable("biases",[64],initializer=tf.constant_initializer(0.0))
            convb  = tf.nn.bias_add(conv,biases)
            out1   = tf.nn.relu(convb,name=scope.name)
            
            pool_1 = tf.nn.max_pool(out1,ksize=[1,3,3,1],strides=[1,1,1,1],
                                    padding="SAME",name="pool1")

        with tf.variable_scope("fc1") as scope:
            lower_node_num = int(np.prod(pool_1.get_shape()[1:]))
            lower_vectorized = tf.reshape(pool_1,[-1,lower_node_num])
            weights = tf.get_variable('weights',[lower_node_num,1000])
            biases = tf.get_variable('biases',[1000],initializer=tf.constant_initializer(0.0))
            out_fc1 = tf.tanh(tf.nn.bias_add(tf.matmul(lower_vectorized,weights),biases))
            
            
        with tf.variable_scope('softmax') as scope:
            weights = tf.get_variable('weights',[1000,self.NUM_CLASSES],initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases  = tf.get_variable('biases',[self.NUM_CLASSES], initializer=tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(tf.matmul(out_fc1,weights),biases)

        #ne_hots = tf.one_hot(self.y,self.NUM_CLASSES)
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        loss = tf.reduce_mean(cross_ent,name='xent_mean')
        # Prediction
        return loss, logits
    
    def run(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(0,trainable=False)
            loss,logits = self.build_model()
            learning_rate = self.conf.learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op  = optimizer.minimize(loss,global_step=global_step)

            pred = tf.argmax(logits,axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.y),tf.float32))
            
            init = tf.global_variables_initializer()
            session = tf.Session()
            session.run(init)
            epoch = total_err = final_eval_acc = final_train_acc = es_count = lr_dec_count = 0
    
            while True:
                batch_x,batch_y =self.dataset.next(self.conf.batch_size)
                if batch_x is not None:
                    _,_err = session.run([train_op,loss],{self.X:batch_x,self.y:batch_y})
                    total_err += _err
                else:
                    x,y = self.dataset.get_train()
                    [train_acc] = session.run([acc],{self.X:x,self.y:y})
                    x,y = self.dataset.get_valid()
                    [eval_acc] = session.run([acc],{self.X:x,self.y:y})

                    epoch +=1
                    print("[Epoch %d] Err = %.3f train_acc = %.3f valid_acc=%.3f" %(epoch,total_err,train_acc,eval_acc))
                    total_err = 0
                    #EARLY STOPPING
                    if eval_acc>final_eval_acc:
                        final_eval_acc = eval_acc
                        final_train_acc = train_acc
                    else:
                        lr_dec_count+=1
                        if lr_dec_count > self.conf.LR_DEC_LIMIT:
                            lr_dec_count = 0
                            learning_rate = learning_rate/2
                            es_count +=1
                    if es_count> self.conf.EARLY_STOPPING or epoch>= self.conf.MAX_ITER:
                        break                                    
            return final_train_acc,final_eval_acc
        
