# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:03:16 2020

@author: Evan Hu (Yi Fan Hu)

"""


# import pandas as pd
import numpy as np
from Tool import globals
from Tool.GeneralData import GeneralData
from GetData.loadData import loadData

import operator, random
from deap import base,creator,gp,tools
#%%
globals.initialize()
loadData()
globals.list_vars()
#%%
from GeneticPogramming import scalarFunctions, singleDataFunctions, singleDataNFunctions, coupleDataFunctions
from inspect import getmembers, isfunction

functions_list = [o for o in getmembers(singleDataFunctions) if isfunction(o[1])]
functions_list

#%%

from deap import base,creator,gp,tools

def if_then_else(input, output1, output2):
    return output1 if input else output2

def elementwise_mul(input1, input2):
    # np.multiply(input1.generalData
    return output1 if input else output2

pset = gp.PrimitiveSetTyped('main', [bool, float], float)

# pset.addPrimitive(operator.xor, [bool, bool], bool)
# pset.addPrimitive(operator.mul, [bool, bool], bool)
# pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addTerminal(globals.adj_close, GeneralData)
pset.addTerminal(globals.adj_open, GeneralData)
pset.addTerminal(globals.adj_high, GeneralData)
pset.addTerminal(globals.adj_low, GeneralData)
pset.addTerminal(1, bool)

# pset.renameArguments(ARG0 = 'x')
# pset.renameArguments(ARG1 = 'y')
#%%
pset.addEphemeralConstant('m1',lambda: random.uniform(-1, 1), float)

#%%
#creating fitness function and individual
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMin,
               pset = pset)

#register them
toolbox = base.Toolbox()

toolbox.register('expr', gp.genFull, pset = pset, min_=1, max_ = 3)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.expr)

toolbox.individual(n=1)

#%%
expr = gp.genFull(pset, min_=1, max_=3)
tree = gp.PrimitiveTree(expr)
str(tree)
#'mul(add(x, x), max(y, x))'

#%%
toolbox.register("compile", gp.compile, pset=pset)


function = toolbox.compile(tree)
function(1,2)





















