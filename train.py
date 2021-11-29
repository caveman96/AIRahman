# -*- coding: utf-8 -*-
# Author: Caveman96
Session_Name= "Nottingham"
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import time
import math
import numpy as np
import preprocess as pre
tf.set_random_seed(0)

#config=tf.ConfigProto(device_count= {'GPU':0})


SEQLEN = 100
BATCHSIZE = 200
ALPHASIZE = 98 #characters
INTERNALSIZE = 512 #internal hidden layers
NLAYERS = 3 #number of lstm/gru cells
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout

tf.reset_default_graph()

print("opening text")
ct= open("Nottingham.txt",encoding='ANSI')
codetext= pre.encode_text(ct.read())

lr = tf.placeholder(tf.float32,name='lrate')
pkeep = tf.placeholder(tf.float32, name='pkeep')
batchsize = tf.placeholder(tf.int32, name='batchsize')

X= tf.placeholder(tf.uint8,[None, None],name='X') #input
Xo= tf.one_hot(X, ALPHASIZE,1.0,0.0) #onehot input

Y_= tf.placeholder(tf.uint8,[None,None],name='Y_') #output (input+1)
Yo_= tf.one_hot(Y_, ALPHASIZE,1.0,0.0) #onehot output

Hin = tf.placeholder(tf.float32,[None, INTERNALSIZE*NLAYERS], name= 'Hin')

cells= [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]

dropcells = [rnn.DropoutWrapper(cell, input_keep_prob = pkeep) for cell in cells]
multicell= rnn.MultiRNNCell(dropcells, state_is_tuple= False)
multicell= rnn.DropoutWrapper(multicell,output_keep_prob = pkeep)

Yr, H= tf.nn.dynamic_rnn(multicell,Xo, dtype=tf.float32, initial_state= Hin)
#Yr is batch output, H is the batch of hidden states

H= tf.identity(H, name= 'H') #naming H

Yflat= tf.reshape(Yr,[-1,INTERNALSIZE]) #making 2d matrix for linear input to external layer
Ylogits= layers.linear(Yflat,ALPHASIZE) #linear layer
Yflat_= tf.reshape(Yo_,[-1, ALPHASIZE]) #2d matrix output

loss= tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels= Yflat_)
loss= tf.reshape(loss, [batchsize,-1]) #separate losses for each batch

Yo= tf.nn.softmax(Ylogits,name='Yo') 
Y= tf.argmax(Yo,1)
Y= tf.reshape(Y,[batchsize,-1],name='Y') #final output
train_step= tf.train.AdamOptimizer(lr).minimize(loss)

#==========================================================================#
# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))




saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar

count = 10 * BATCHSIZE * SEQLEN
#progress = pre.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

#===========================================================================#
#init
istate= np.zeros([BATCHSIZE,INTERNALSIZE*NLAYERS]) #zero initial state
init= tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)
step=0

summary_writer = tf.summary.FileWriter("log/" +Session_Name+ timestamp,sess.graph )
#main training loop
for x, y_, epoch in pre.rnn_minibatch_sequencer(codetext,BATCHSIZE,SEQLEN,nb_epochs=200):
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate,pkeep: dropout_pkeep, batchsize:BATCHSIZE}
    _, y, ostate= sess.run([train_step,Y,H], feed_dict=feed_dict)
	
	# log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
    if step % count == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        #pre.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
        summary_writer.add_summary(smm, step)
        print("epoch {} step {}".format(epoch,step))
		
	# display a short text generated with the current weights and biases (every 150 batches)
    '''if step // 10 % count == 0:
        print("\n Generating sample \n\n")
        ry = np.array([[pre.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, INTERNALSIZE * NLAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = pre.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(pre.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
       ''' 
        # save a checkpoint (every 500 batches)
    if step // 10 % count == 0:
        saved_file = saver.save(sess, 'checkpoints/'+Session_Name+ timestamp, global_step=step)
        print("Saved file: " + saved_file)
	
    

    # display progress bar
    #progress.step(reset=step % _50_BATCHES == 0)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN




