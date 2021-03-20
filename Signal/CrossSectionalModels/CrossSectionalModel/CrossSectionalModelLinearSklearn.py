# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:16:04 2020

@author: Mengjie Ye
"""
# import os
# import sys
from sklearn.linear_model import LinearRegression,Ridge,Lasso

# =============================================================================
# abspath = os.path.abspath('.')
# sys.path.append(abspath+'\..')
# try:
#     from .Base.CrossSectionalModelSklearn import CrossSectionalModelSklearn
# except:
#     from Base.CrossSectionalModelSklearn import CrossSectionalModelSklearn
# =============================================================================

from BackTesting.Signal.CrossSectionalModels.Base.CrossSectionalModelSklearn import CrossSectionalModelSklearn

# import matplotlib.pyplot as plt
    
#%% OLS
class CrossSectionalModelOLS(CrossSectionalModelSklearn):
    
    def __init__(self, jsonPath = None, paraDict = {}, json_first = True):
        # super(CrossSectionalModelOLS,self).__init__(jsonPath = None, paraDict = {})
        super().__init__(jsonPath = jsonPath, 
                         paraDict = paraDict, 
                         json_first = json_first)
        self.model = LinearRegression(**self.parameter)


#%% Ridge
class CrossSectionalModelRidge(CrossSectionalModelSklearn):
    
    def __init__(self,jsonPath = None, paraDict = {}, paraGrid = None, json_first = True):
        super().__init__(jsonPath = jsonPath, 
                         paraDict = paraDict, 
                         paraGrid = paraGrid,
                         json_first = json_first)
        self.model = Ridge(**self.parameter)

#%% Lasso  
class CrossSectionalModelLasso(CrossSectionalModelSklearn):
    
    def __init__(self,jsonPath = None, paraDict = {}, paraGrid = None, json_first = True):
        super().__init__(jsonPath = jsonPath, 
                         paraDict = paraDict, 
                         paraGrid = paraGrid,
                         json_first = json_first)
        self.model = Lasso(**self.parameter)

        
#%%

   
    
    
    



     