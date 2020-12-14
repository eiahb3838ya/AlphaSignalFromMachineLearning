# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:18:58 2020

@author: Lantian
"""

import sys
sys.path.append("../../")
from CrossSectionalModelBase import CrossSectionalModelBase
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import pandas as pd
import json

class CrossSectionalModelKNN(CrossSectionalModelBase):
    def __init__(self, jsonpath = None, para = {}):
        #take both jsonpath and the input para into consideration
        if jsonpath is not None:
            with open (jsonpath, 'r') as f:
                para.update(json.load(f))
            with open (jsonpath, 'w') as f:
                json.dump(para, f)
        self.para = para
        self.model = KNeighborsClassifier(**self.para)
    
    def fit(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
     
    def predict(self, X):
        return self.model.predict(X)
    
    def getPara(self):
        return self.para
    
    def getModel(self):
        return self.model
    
    def predictScore(self,x,y):
        return self.model.score(x,y)

if __name__=='__main__':
    from testCSModel import create_classification_dataset 
    X_train, y_train, X_test, y_test = create_classification_dataset()
    
    param = {'weights': 'distance'}
    data = {'n_neighbors': 4}
    with open ('data.json','w') as f:
        json.dump(data,f)
    model = CrossSectionalModelKNN(jsonpath = 'data.json', para = param)
    model.fit(X_train,y_train)
    predy = model.predict(X_test)