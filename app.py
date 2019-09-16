#!/usr/bin/env python
# coding: utf-8

# In[105]:


from flask import Flask,jsonify,request
from flask_restful import reqparse, abort, Api, Resource
from flask_jwt import JWT
import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing import sequence,text
import tensorflow as tf
import sqlite3
import pickle
from flask_cors import CORS
import numpy as np

    


# In[106]:


app = Flask(__name__)

api = Api(app)
model=None


# In[107]:


CORS(app)


# In[108]:


global graph
graph = tf.get_default_graph()


# In[109]:


with open("word2id.pickle","rb") as f2:
    word2id = pickle.load(f2)    


# In[110]:


def sentences_to_indices(X, word2id, max_len): 
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


# In[111]:


def load_model():
    global model
    # load json and create model
    json_file = open('lstm_model.json', 'r')
    loaded_model_json = json_file.read()

    #print(loaded_model.summary)
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("lstm_model.h5")
    print("Loaded model from disk")
    json_file.close()


# In[112]:


parser = reqparse.RequestParser()


# In[113]:


class PredictLabel(Resource):
    
    TABLE_NAME='responses'

    @classmethod
    def find_by_name(cls, query):
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        sql = "SELECT * FROM {table} WHERE query=?".format(table=cls.TABLE_NAME)
        result = cursor.execute(sql, (query,))
        row = result.fetchone()
        print(row)
        connection.close()
        if row:
            return {'query': row[0], 'label': row[1]}
        
    @classmethod
    def insert(cls,query, out):
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        sql = "INSERT INTO {table} VALUES(?, ?)".format(table=cls.TABLE_NAME)
        cursor.execute(sql,(query,out))
        connection.commit()
        connection.close()
        
    def post(self):
        parser.add_argument('query', type=str)
        args = parser.parse_args()
        query=args['query']
        print(query)
        print(len(query))
        if (len(query)==0):
            return {"message": "Enter data"}
        row=self.find_by_name(query)
        if row:
            return jsonify({'query':row['query'],'label':row['label']})
        else:
            X=[]
            X.append(query)
            X_arr=np.asarray(X)
            indices = sentences_to_indices(X_arr, word2id, 207)
            print(indices)
            with graph.as_default():
                pred =model.predict(indices)
                num = np.argmax(pred)
                # Output 'company' or 'fruit' 
                if(num==0):
                    out='computer-company'
                else:
                    out='fruit'
                print("label"+out)    
            try:
                PredictLabel.insert(query,out)
            except:
                return {"message": "An error occurred inserting the item."}
            return jsonify({'query':query,'label':out})
        


# In[114]:


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


# In[115]:


api.add_resource(PredictLabel, '/predictLabel')


# In[116]:


api.add_resource(HelloWorld, '/hello')


# In[117]:


#Main block
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(debug=True,use_reloader=False)


# In[ ]:





# In[ ]:





# In[ ]:




