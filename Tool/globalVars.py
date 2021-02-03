# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:44 2020

@author: eiahb
"""





def initialize(inputLogger = None): 
    if inputLogger ==None:
        import logging
        inputLogger = logging.getLogger()
    global varList
    varList = []    
    register("logger", inputLogger)
    register("factor", {})
    register("barra", {})
    register("materialData", {})
    register("utilData", {})

    
def register(name, aVar):
    globals()[name] =  aVar
    globals()['varList'].append(name)
    logger.info('{}:{} is now in global'.format(name, aVar))
    return(globals()['varList'])

def list_vars():
    return(globals()['varList'])