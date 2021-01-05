# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:03:16 2020

@author: Evan Hu (Yi Fan Hu)

"""

import  random
from inspect import getmembers, isfunction

import numpy as np
from deap import base,creator,gp,tools

from Tool import globalVars
from Tool.GeneralData import GeneralData
from GetData.loadData import load_data
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions

#%% initialize global vars

globalVars.initialize()
loadedData = load_data()
globalVars.list_vars()
#%% add primitives

pset = gp.PrimitiveSetTyped('main', [GeneralData,GeneralData,GeneralData,GeneralData,GeneralData,GeneralData], GeneralData)

pset.renameArguments(ARG0 = 'CLOSE')
pset.renameArguments(ARG1 = 'OPEN')
pset.renameArguments(ARG2 = 'HIGH')
pset.renameArguments(ARG3 = 'LOW')
pset.renameArguments(ARG4 = 'AMOUNT')
pset.renameArguments(ARG5 = 'VOLUME')

for aName, primitive in [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('singleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(scalarFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('scalarFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, float], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(singleDataNFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('singleDataNFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, int], GeneralData, aName)
    
for aName, primitive in [o for o in getmembers(coupleDataFunctions) if isfunction(o[1])]:
    print('add primitive from {:<20}: {}'.format('coupleDataFunctions', aName))
    pset.addPrimitive(primitive, [GeneralData, GeneralData], GeneralData, aName)
    
def return_self(this):
    return(this)

pset.addPrimitive(return_self, [int], int, 'self_int')
pset.addPrimitive(return_self, [float], float, 'self_float')
    


#%% add EphemeralConstant
pset.addEphemeralConstant(name = 'EphemeralConstant_flaot',
                         ephemeral = lambda: random.uniform(-10, 10),
                         ret_type=float)
pset.addEphemeralConstant(name = 'EphemeralConstant_int',
                         ephemeral = lambda: random.randint(1, 180),
                         ret_type = int)


#%%
#creating fitness function and individual
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree,
               fitness = creator.FitnessMin,
               pset = pset) # Individual inherits gp.PrimitiveTree, and add fitness and pset as class vars

#register them
toolbox = base.Toolbox()

# call gp.genHalfAndHalf with following inputs to generate a expr, now we can use toolbox.expr() to call genHalfAndHalf to easily generate a expr
toolbox.register('expr', gp.genHalfAndHalf, pset = pset, min_=1, max_ = 5)  

# call tools.initIterate with following inputs to generate a creator.Individual
# the new func toolbox.inividual is a partial function of tools.initIterate
# the initIterate will call toolbox.expr and put it into creator.Individual which is a class inherits gp.PrimitiveTree
# that is, the output of toolbox.individual is a gp.PrimitiveTree
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)

# the function toolbox.population is a partial function of tools.initRepeat which still needs input variable n 
# and initRepeat calls toolbox.individual n times and put them into creator.list
# therefore the output of toolbox.population(n=n) is a list of creator.Individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

trees = toolbox.population(n=20)
print([str(aTree) for aTree in trees])
trees = toolbox.population(n=20)



#%% how to use compile

toolbox.register("compile", gp.compile, pset=pset)
function = toolbox.compile(trees[0])
toolbox.compile(trees[0])

#%% check out input data
close = globalVars.adj_close
close_np = close.generalData
np.sum(~np.isfinite(close_np[-3:]), axis = 1)
ts_argmax(close, 5)
this = close























