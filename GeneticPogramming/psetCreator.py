# -*- coding: utf-8 -*-
from deap import gp
import itertools
import numpy.random as random
from inspect import getmembers, isfunction

from GeneticPogramming.primitivefunctions import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions
from Tool import GeneralData, Logger


#%% set up

#PROJECT_ROOT
PROJECT_ROOT = 'C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\'

# logger
loggerFolder = PROJECT_ROOT+"Tool\\log\\"
logger = Logger(loggerFolder, 'log')
#%%
def pset_creator(materialDataNames):
    '''
    get the pset for to create expr

    Parameters
    ----------
    materialDataNames : List
        the list of material data that will be used to create expr.

    Returns
    -------
    pset.

    '''
    inputOfPset = list(itertools.repeat(GeneralData, len(materialDataNames)))
    pset = gp.PrimitiveSetTyped('main', inputOfPset, GeneralData)
    for aName, primitive in [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]:
        # print('add primitive from {:<20}: {}'.format('singleDataFunctions', aName))
        pset.addPrimitive(primitive, [GeneralData], GeneralData, aName)
        
    for aName, primitive in [o for o in getmembers(scalarFunctions) if isfunction(o[1])]:
        # print('add primitive from {:<20}: {}'.format('scalarFunctions', aName))
        pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
        
    for aName, primitive in [o for o in getmembers(singleDataNFunctions) if isfunction(o[1])]:
        # print('add primitive from {:<20}: {}'.format('singleDataNFunctions', aName))
        pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
        
    for aName, primitive in [o for o in getmembers(coupleDataFunctions) if isfunction(o[1])]:
        # print('add primitive from {:<20}: {}'.format('coupleDataFunctions', aName))
        pset.addPrimitive(primitive, [GeneralData, GeneralData], GeneralData, aName)
    
    def return_self(this):
        return(this)
    
    pset.addPrimitive(return_self, [int], int, 'self_int')
    pset.addPrimitive(return_self, [float], float, 'self_float')

    
    # add Arguments
    argDict = {'ARG{}'.format(i):argName for i, argName in enumerate(materialDataNames)}
    pset.renameArguments(**argDict)

    # add EphemeralConstant
    try:
        pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                              ephemeral = lambda: random.uniform(-1, 1),
                              ret_type=float)
        pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                                  ephemeral = lambda: random.randint(1, 10),
                                  ret_type = int)
    except Exception:
        del gp.EphemeralConstant_flaot
        pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                                 ephemeral = lambda: random.uniform(-1, 1),
                                 ret_type=float)
        del gp.EphemeralConstant_int
        pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                                  ephemeral = lambda: random.randint(1, 10),
                                  ret_type = int)
        logger.warning(str(gp.EphemeralConstant_flaot)+"is ready")
        logger.warning(str(gp.EphemeralConstant_int)+"is ready")
        
    return(pset)
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    # materialDataNames
    materialDataNames = [
        'close',
        'high',
        'low',
        'open',
        # 'preclose',
        'amount',
        'volume',
        'pctChange'
    ]
    pset = pset_creator(materialDataNames)
    print(pset.arguments)
    print(pset.mapping)

    
    
    
    
    
    
    
    
    
    
    