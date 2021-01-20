# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:44 2020

@author: eiahb
"""





def initialize(inputLogger): 
    global logger 
    logger = inputLogger
    logger.info("globalVar has initialized")
    global varList
    varList = []
    
    
def register(name, aVar):
    globals()[name] =  aVar
    globals()['varList'].append(name)
    logger.info('{}:{} is now in global'.format(name, aVar))
    return(globals()['varList'])

def list_vars():
    return(globals()['varList'])