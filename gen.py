
import tensorflow as tf
import numpy as np
import preprocess as pre
import midi_manipulation as mm

# these must match what was saved !
ALPHASIZE = pre.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512



# use topn=10 for all but the last one which works with topn=2 for Shakespeare and topn=3 for Python
author = "checkpoints/mario"
w=open(author+".txt",'w+')
ncnt = 0
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(author+ ".meta")
    new_saver.restore(sess, author)
    x = pre.convert_from_alphabet(ord("`"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    # initial values
    y = x
    h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    
    for i in range(10000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        c = pre.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(pre.convert_to_alphabet(c))
        w.write(c)


w.close()

notein=mm.dec_int(author + ".txt",775)
mm.noteStateMatrixToMidi(notein, "res", 95)
