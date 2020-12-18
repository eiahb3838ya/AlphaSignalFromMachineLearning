# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:44 2020

@author: eiahb
"""




def initialize(): 
    global varList
    varList = []
    
    
def register(name, aVar):
  
    globals()[name] =  aVar
    globals()['varList'].append(name)
    print('{} \n is now in global\n'.format(aVar))
    return(globals()['varList'])
    
def list_vars():
    return(globals()['varList'])