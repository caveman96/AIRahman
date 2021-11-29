import numpy as np

ALPHASIZE= 98

def convert_from_alphabet(a):
    #0-31 are control characters
	#9 is tab
    if a == 9:
        return 1
	#10 is \n
    if a == 10:
        return 127 - 30 
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0  # unknown
    return 0

def convert_to_alphabet(c):

    if c == 1:
        return  9  
    if c == 127 - 30:
        return 10  
    if 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0  # unknown
		

def encode_text(s):
    #encode each character based on above functions
    return list(map(lambda a: convert_from_alphabet(ord(a)), s))


def decode_to_text(c):
    #decode number back to characer and return string
    return "".join(map(lambda a: chr(convert_to_alphabet(a)), c))
	
	
def sample_from_probabilities(probabilities, topn=ALPHASIZE):
    #return a random value from the top topn probabilities
    p = np.squeeze(probabilities) #make single dimensional array
    p[np.argsort(p)[:-topn]] = 0 #get sorted indices and set everything except the topn to 0
    p = p / np.sum(p)  #softmax
    return np.random.choice(ALPHASIZE, 1, p=p)[0] #return a random value based on the probabilities
	

def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):

#iteratable function that returns a batch of inputs and outputs every epoch
   
    data = np.array(raw_data) #string to array
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    
	#basically convert data into batch_size rows
	#nb batches is the number of batches
    nb_batches = (data_len - 1) // (batch_size * sequence_size) #floor division so all batches have equal number of characters
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size 
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size]) #all x data
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size]) #xdata+1

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!) ir, sent second row as first and so on
            y = np.roll(y, -epoch, axis=0) #doing the same for y
            yield x, y, epoch


