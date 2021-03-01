# -*- coding: utf-8 -*-
#%%
from deap import gp
import itertools
import numpy.random as random
from inspect import getmembers, isfunction

from GeneticPogramming.primitivefunctions import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions
from Tool import GeneralData, Logger

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
    # add all functions in singleDataFunctions、scalarFunctions、singleDataNFunctions、coupleDataFunctions
    # it will automately add to the system without any move
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
    # we will add materials in the  materialDataNames to pset automaticly
    argDict = {'ARG{}'.format(i):argName for i, argName in enumerate(materialDataNames)}
    pset.renameArguments(**argDict)

    # add EphemeralConstant
    # 這裡因為 deap 有一個奇怪的設計，就是 EphemeralConstant_flaot 被他定義在 global 裡
    # 如果你添加過，他就會報錯，所以用 try catch 來測試是不是有 run 過
    # 如果他添加過了，我們就刪掉他，再添一次 hhh
    try:
        pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                              ephemeral = lambda: random.uniform(-1, 1),
                              ret_type=float)
        pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                                  ephemeral = lambda: random.randint(2, 5),
                                  ret_type = int)
    except Exception:
        del gp.EphemeralConstant_flaot
        pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                                 ephemeral = lambda: random.uniform(-1, 1),
                                 ret_type=float)
        del gp.EphemeralConstant_int
        pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                                  ephemeral = lambda: random.randint(2, 5),
                                  ret_type = int)
        print(str(gp.EphemeralConstant_flaot)+"is already in gp global, we del it and reAdd anyway")
        print(str(gp.EphemeralConstant_int)+"is already in gp global, we del it and reAdd anyway")
    return(pset)
    
    
if __name__ == '__main__':
    # materialDataNames
    materialDataNames = [
        'close',
        'high',
        'low',
        'open',
        'amount',
        'volume',
        'pctChange'
    ]
    pset = pset_creator(materialDataNames)
    print(pset.arguments)
    print(pset.mapping)

    
    
    
    
    
    
    
    
    
    
    