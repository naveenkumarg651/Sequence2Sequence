from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import Adam

encoder_inputs1=[]
decoder_inputs=[]
decoder_outputs=[]
t=0
with open('G:/present/NLP/hin-eng/hin.txt',encoding='utf8') as f:
    for line in f:
        t+=1
        if t>10000:
            break
        
        eng,hin=line.rstrip().split('\t')
        encoder_inputs1.append(eng)
        a='<sos> '+hin
        b=hin+' <eos>'
        decoder_inputs.append(a)
        decoder_outputs.append(b)
all_lines=decoder_outputs+decoder_inputs
print("num samples:", len(encoder_inputs1))

tokenizer1=Tokenizer(num_words=20000)
tokenizer1.fit_on_texts(encoder_inputs1)
encoder_inputs=tokenizer1.texts_to_sequences(encoder_inputs1)
encoder_length=max(len(s) for s in encoder_inputs)
encoder_word2idx=tokenizer1.word_index
encoder_vocab_size=len(encoder_word2idx)+1
print('Found %s unique input tokens.' % len(encoder_word2idx))
encoder_data=pad_sequences(encoder_inputs,maxlen=encoder_length)
print("encoder_inputs.shape:", encoder_data.shape)
print("encoder_inputs[0]:", encoder_data[0])
tokenizer2=Tokenizer(num_words=20000,filters='')
tokenizer2.fit_on_texts(all_lines)
decoder_inputs=tokenizer2.texts_to_sequences(decoder_inputs)
decoder_outputs=tokenizer2.texts_to_sequences(decoder_outputs)
decoder1_length=max(len(s) for s in decoder_inputs)
decoder_word2idx=tokenizer2.word_index
decoder_vocab_size=len(decoder_word2idx)+1
print('Found %s unique output tokens.' % len(decoder_word2idx))

decoder_input_data=pad_sequences(decoder_inputs,maxlen=decoder1_length,padding='post')
print("decoder_inputs[0]:", decoder_input_data[0])
print("decoder_inputs.shape:", decoder_input_data.shape)
decoder_output_data=pad_sequences(decoder_outputs,maxlen=decoder1_length,padding='post')

decoder_targets=np.zeros((len(decoder_outputs),decoder1_length,decoder_vocab_size),dtype="float32")

for i,output in enumerate(decoder_output_data):
    for j,out in enumerate(output):
        decoder_targets[i,j,out]=1
#configurations

epochs=150
batch_size=64
validation_split=0.2
latent_dim=256
embedding_dim=100
MAX_NUM_WORDS = 20000

#loading pre trained glove vectors

embedding_matrix=np.random.randn(encoder_vocab_size,embedding_dim)

#wordvec={}
#
#with open('G:/present/NLP/glove.6B.100d.txt',encoding="utf8") as f:
#    for line in f:
#        
#        line=line.split()
#        word=line[0]
#        vec=np.asarray(line[1:],dtype="float32")
#        wordvec[word]=vec
#print('Found %s word vectors.' % len(wordvec))
#
#for word,i in encoder_word2idx.items():
#    if i < MAX_NUM_WORDS:    
#        vec=wordvec.get(word)
#        if vec is not None:
#            embedding_matrix[i]=vec


# main model
embedding_layer=Embedding(encoder_vocab_size,embedding_dim,weights=[embedding_matrix],input_length=encoder_length,trainable=True)

encoder_placeholder=Input(shape=(encoder_length,))
x=embedding_layer(encoder_placeholder)
lstm0=LSTM(100,return_sequences=True)
x=lstm0(x)
#lstm_1=LSTM(200,return_sequences=True)
#x=lstm_1(x)
#lstm_2=LSTM(150,return_sequences=True)
#x=lstm_2(x)
#lstm_3=LSTM(120,return_sequences=True)
#x=lstm_3(x)
#lstm_4=LSTM(90,return_sequences=True)
#x=lstm_4(x)
#lstm_5=LSTM(120,return_sequences=True)
#x=lstm_5(x)
#lstm_6=LSTM(180,return_sequences=True)
#x=lstm_6(x)
lstm1=LSTM(latent_dim,return_state=True)
encoder_output,encoder_h,encoder_c=lstm1(x)
encoder_states=[encoder_h,encoder_c]
decoder_placeholder=Input(shape=(decoder1_length,))
decoder_embedding=Embedding(decoder_vocab_size,latent_dim)
x=decoder_embedding(decoder_placeholder)
decoder_lstm1=LSTM(latent_dim,return_sequences=True)
x=decoder_lstm1(x,initial_state=encoder_states)
#decoder_lstm2=LSTM(180,return_sequences=True)
#x=decoder_lstm2(x)
#decoder_lstm3=LSTM(120,return_sequences=True)
#x=decoder_lstm3(x)
#decoder_lstm4=LSTM(90,return_sequences=True)
#x=decoder_lstm4(x)
#decoder_lstm5=LSTM(120,return_sequences=True)
#x=decoder_lstm5(x)
#decoder_lstm6=LSTM(180,return_sequences=True)
#x=decoder_lstm6(x)
lstm=LSTM(256,return_sequences=True,return_state=True)
decoder_output,_,_=lstm(x)
dense2=Dense(decoder_vocab_size,activation="softmax")
output=dense2(decoder_output)

model=Model([encoder_placeholder,decoder_placeholder],output)
model.compile(
  optimizer=Adam(lr=0.001),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r=model.fit([encoder_data,decoder_input_data],decoder_targets,validation_split=0.2,batch_size=32,epochs=epochs)

# sampling model

encoder_model=Model(encoder_placeholder,[encoder_h,encoder_c])

sample_decoder_placeholder=Input(shape=(1,))
sample_hidden_state=Input(shape=(latent_dim,))
sample_cell_state=Input(shape=(latent_dim,))
x=decoder_embedding(sample_decoder_placeholder)
x=decoder_lstm1(x,initial_state=[sample_hidden_state,sample_cell_state])
#x=decoder_lstm2(x)
#x=decoder_lstm3(x)
#x=decoder_lstm4(x)
#x=decoder_lstm5(x)
#x=decoder_lstm6(x)
sample_decoder_output,hidden,cell=lstm(x)
output=dense2(sample_decoder_output)

sample_decoder_model=Model([sample_decoder_placeholder,sample_hidden_state,sample_cell_state],[output,hidden,cell])

decoder_idx2word={v:k for k,v in decoder_word2idx.items()}


#sampling

while(True):
    sentence=input('\n\nenter the sentence to translate\n\n')
    tokenizer=Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(encoder_inputs1)
    sequence=tokenizer.texts_to_sequences([sentence])
    data=pad_sequences(sequence,maxlen=encoder_length)
    
    hidden,cell=encoder_model.predict(data)
    target=np.array([[decoder_word2idx['<sos>']]])
    k=''
    for i in range(decoder1_length):
        probs,hidden,cell=sample_decoder_model.predict([target,hidden,cell])
        probs=probs[0,0]
        
        idx=np.argmax(probs)
        word=decoder_idx2word.get(idx)
        print("word   {}".format(word))
        if word=='<eos>':
            break
        k+=' '+word
        target[0,0]=idx
    print('\n',k)
    o=input('would you like to translate further more sentences [y/n]\n')
    if o=='n':
        break
    
        






