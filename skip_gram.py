#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import spacy
import re
from nltk.stem import WordNetLemmatizer 
import nltk
from nltk.corpus import words
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
warnings.filterwarnings("ignore")
from sklearn import metrics


# In[199]:


from keras.preprocessing import text


# In[202]:


'''f=open("apple-computers.txt", "r", encoding="utf8")
computer =f.read()'''
with open("apple-computers.txt",encoding="utf8") as f:
    computer = f.readlines() 
with open("apple-fruit.txt",encoding="utf8") as f:
    fruit = f.readlines()   
df_comp= pd.DataFrame(computer)
df_fruit=pd.DataFrame(fruit)  
df_comp['label']='computer'
df_fruit['label']='fruit'


# In[433]:


df_fruit.rename(columns = {0: "Text"},inplace = True) 


# In[434]:


df_comp.rename(columns = {0: "Text"},inplace = True) 


# In[435]:


df=pd.concat([df_fruit,df_comp])


# In[436]:


df.reset_index(inplace=True,drop=True)


# In[437]:


df


# In[438]:


df=df[~(df['Text']=='\n')]


# In[439]:


df.reset_index(inplace=True,drop=True)


# In[440]:


def cleaning_data(data):
    proc_data=re.sub(r'\[[^\]]*\]'," ",data)#removing square brackets and everything in between
    proc_data=re.sub(r"\d+", " ",proc_data) #removing numbers
    proc_data=re.sub(r'\n'," ", proc_data)#removing newline
    proc_data=re.sub(r'\t'," ", proc_data)#removing tab
    proc_data=re.sub(r'\r\n'," ", proc_data)#removing newline
    proc_data=re.sub(r'[.|, # : _ ;! ? = // \- \* <>+\\ \'  \) \( \" \} \{ \%"]'," ",proc_data)#removing punctuation
    
    return proc_data


# In[441]:


df['clean_data']=df['Text'].astype('str').str.lower().apply(cleaning_data)


# In[442]:


df=df[~(df["clean_data"]=='see also ')] 


# In[222]:


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(df['clean_data'])


# In[223]:


word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}


# In[369]:


len(id2word) #counting starts from 1


# In[226]:


word2id.items()#dictionary of words with the index


# In[232]:


# In[371]:


vocab_size = len(word2id) + 1 #length of unique words we did plus one because encoding is starting from 1 so to accomodate 0
vector_dim = 100 #embedding vector size
epochs = 1000000
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in df["clean_data"]]



# In[261]:


from keras.preprocessing.sequence import skipgrams

# generate skip-grams
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]

# view sample skip-grams
couples, labels = skip_grams[0][0], skip_grams[0][1]


# In[303]:


word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")


# In[285]:


#Using keras function APIs
from keras.layers import Input, Embedding, LSTM, Dense,Dot,Reshape
from keras.layers import dot
from keras.models import Model
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')


# In[286]:


target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)


# In[323]:


similarity= dot([target, context], axes=1, normalize=True)#for cosine normalize is true


# In[325]:


# now perform the dot product operation to get a similarity measure
dot_product = dot([target, context], axes=1, normalize=False)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer


# In[326]:


output = Dense(1, activation='sigmoid')(dot_product)


# In[327]:


# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')


# In[328]:


# create a secondary validation model to run our similarity checks during training
validation_model = Model(input=[input_target, input_context], output=similarity)


# In[317]:





# In[331]:


'''class SimilarityCallback:
   def run_sim(self):
       for i in range(valid_size):
           valid_word = id2word[valid_examples[i]]
           top_k = 8  # number of nearest neighbor
           sim = self._get_sim(valid_examples[i])
           nearest = (-sim).argsort()[1:top_k + 1]
           log_str = 'Nearest to %s:' % valid_word
           for k in range(top_k):
               close_word = id2word[nearest[k]]
               log_str = '%s %s,' % (log_str, close_word)
           print(log_str)

   @staticmethod
   def _get_sim(valid_word_idx):
       sim = np.zeros((vocab_size,))
       in_arr1 = np.zeros((1,))
       in_arr2 = np.zeros((1,))
       for i in range(vocab_size):
           in_arr1[0,] = valid_word_idx
           in_arr2[0,] = i
           out = validation_model.predict_on_batch([in_arr1, in_arr2])
           sim[i] = out
       return sim
sim_cb = SimilarityCallback()'''


# In[332]:


arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))



# In[372]:


len(labels)


# In[340]:


loss


# In[341]:


model.summary()


# In[342]:


weights= embedding.get_weights()


# In[374]:


weights[0].shape #because embedding matrix it starts from 0 so 3600


# In[365]:


weights[0][1:].shape


# In[358]:


id2word


# In[425]:


embedding_df=pd.DataFrame(weights[0][1:], index=id2word.values())


# In[426]:


word_to_vector_map={}


# In[427]:


x1=df.to_dict('split')
key=x1['index']
value=x1['data']


# In[428]:


for i in range(len(key)):
    word_to_vector_map[key[i]]=value[i]


# In[643]:


# serialize model to JSON
model_json = model.to_json()
with open("lstm_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("lstm_model.h5")
print("Saved model to disk")
 


# In[412]:


#loading the model
loaded_model = pickle.load(open(filename, 'rb'))


# In[413]:


#saving word_to_vector_map dictionary using pickle
pickle_out = open("word_to_vector_map.pickle","wb")
pickle.dump(word_to_vector_map, pickle_out)
pickle_out.close()


# In[414]:


#loading the dictionary
pickle_in = open("word_to_vector_map.pickle","rb")
loaded_dict = pickle.load(pickle_in)


# In[416]:


#saved word to id
pickle_out = open("word2id.pickle","wb")
pickle.dump(word2id, pickle_out)
pickle_out.close()


# In[417]:


#saved if to word
pickle_out = open("id2word.pickle","wb")
pickle.dump(id2word, pickle_out)
pickle_out.close()


# # LSTM model to predict label

# In[459]:


import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


# In[567]:


def sentences_to_indices(X, word2id, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = text.text_to_word_sequence(X[i], lower=True, split=" ")
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word2id[w]
            # Increment j to j + 1
            j += 1
            
    ### END CODE HERE ###
    
    return X_indices


# In[578]:


'''temp=np.array([df['clean_data'].iloc[0],df['clean_data'].iloc[89],df['clean_data'].iloc[100]])'''


# In[471]:


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = vocab_size                 # adding 1 to fit Keras embedding (requirement)
    emb_dim = vector_dim     # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word2id.items():
        emb_matrix[index, :] = word_to_vector_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[472]:


embedding_layer = pretrained_embedding_layer(word_to_vector_map, word2id)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# In[593]:


def Apple_apple(input_shape, word_to_vector_map, word2id):
    """
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape
    #and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with Skipgram vectors
    embedding_layer = pretrained_embedding_layer(word_to_vector_map, word2id)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 2-dimensional vectors.
    X = Dense(2)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    ### END CODE HERE ###
    
    return model


# In[594]:


model = Apple_apple((207,), word_to_vector_map, word2id)
model.summary()


# In[595]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


from sklearn.model_selection import train_test_split
X=df['clean_data']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)
X_arr_train= X_train.as_matrix()
X_arr_test=X_test.as_matrix()
Y_oh_train = pd.get_dummies(y_train)
Y_oh_test =  pd.get_dummies(y_test)


# In[622]:


y_train=y_train.as_matrix()
y_test=y_test.as_matrix()


# In[627]:


#training on full data
X_arr= X.as_matrix()
Y_oh=pd.get_dummies(y)


# In[633]:


Y_oh=Y_oh.as_matrix()


# In[631]:


X_indices = sentences_to_indices(X_arr, word2id, 207)


# In[634]:


model.fit(X_indices, Y_oh, epochs = 50, batch_size = 32, shuffle=True)


# In[642]:


#saving the lstm model using pickel
filename = 'lstm_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[638]:


#testing on unseen files
with open("input00.txt",encoding="utf8") as f:
    test_data = f.readlines() 
 


# In[644]:


with open("output00.txt",encoding="utf8") as f:
    test_label = f.readlines() 


# In[639]:


test_data=pd.DataFrame(test_data)


# In[645]:


test_label=pd.DataFrame(test_label)


# In[646]:


test_label


# In[609]:


X_test_indices = sentences_to_indices(X_arr_test, word2id, max_len = 207)
loss, acc = model.evaluate(X_test_indices, Y_oh_test)
print("Test accuracy = ", acc)


# In[617]:


X_test_indices 


# In[625]:


X_test_indices = sentences_to_indices(X_arr_test, word2id, 207)
pred = model.predict(X_test_indices)
for i in range(len(X_arr_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(y_test[i]=='computer'):
        out=0
    else:
        out=1
    if(num != out):
        print('Expected label:'+str(y_test[i])+ ' prediction: '+ str(X_arr_test[i]) +' '+num )


# In[ ]:





# In[ ]:




