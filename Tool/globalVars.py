# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:33:44 2020

@author: eiahb
"""

import os
from .logger import Logger


def initialize(logName = 'log'): 
    global varList
    varList = []
    
    # global PROJECT_ROOT    
    global PROJECT_ROOT
    # PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'
    cur_path = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.path.join(cur_path, '../')
    loggerFolder = PROJECT_ROOT+"Tool\\log\\"
    fileName = logName
    
    global logger
    logger = Logger(loggerFolder, fileName)
    varList.append('logger')
    
def register(name, aVar):
    globals()[name] =  aVar
    globals()['varList'].append(name)
    logger.info('{}:{} is now in global'.format(name, aVar))
    return(globals()['varList'])

def list_vars():
    return(globals()['varList'])