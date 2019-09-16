#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
from flask import Flask,jsonify,request
from flask_restful import reqparse, abort, Api, Resource
import numpy as np


# In[2]:


class PredictLabel(Resource):
    def post(self,query):
        print(query)
        X=[]
        X.append(query)
        X_arr=np.asarray(X)
        indices = sentences_to_indices(X_arr, word2id, 207)
        print(indices)
        with graph.as_default():
            pred = loaded_model.predict(indices)
            num = np.argmax(pred)
            # Output 'company' or 'fruit' 
            if(num==0):
                out='computer-company'
            else:
                out='fruit'
        #saving data to database        
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        sql = "INSERT INTO {table} VALUES(?, ?)".format(table='responses')
        cursor.execute(sql, (query, out))

        connection.commit()
        connection.close()        
        return jsonify({'output':out})


# In[ ]:




