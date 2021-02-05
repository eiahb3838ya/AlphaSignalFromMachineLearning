'''
A replacement for map using Ray that automates batching to many processors/cluster. Fixes pickle issues such as
DeltaPenalty and addEphemeralConstant when working at scale(many processes on machine or cluster of nodes)
'''
#%%
from time import sleep
from time import time
import os
import ray
from ray.util import ActorPool

#%%
@ray.remote
class Ray_Deap_Map():
    def __init__(self, creator_setup=None, pset_creator = None):
        # issue 946? Ensure non trivial startup to prevent bad load balance across a cluster
        # sleep(0.01)

        # recreate scope from global
        # For GA no need to provide pset_creator. Both needed for GP
        self.creator_setup = creator_setup
        self.psetCreator = pset_creator
        if creator_setup is not None:
            self.creator_setup()
            self.psetCreator()

    def ray_remote_eval_batch(self, f, iterable):
        # iterable, id_ = zipped_input
        # attach id so we can reorder the batches
        return [f(i) for i in iterable]

def ray_deap_map(func, pop, creator_setup, pset_creator):
    n_workers = int(ray.cluster_resources()['CPU'])
    if n_workers == 1:
        results = list(map(func, pop)) #forced eval to time it
    else:
        # many workers
        if len(pop) < n_workers:
            n_workers = len(pop)
        else:
            n_workers = n_workers

    n_per_batch = int(len(pop)/n_workers) + 1
    batches = [pop[i:i + n_per_batch] for i in range(0, len(pop), n_per_batch)]
    actors = [Ray_Deap_Map.remote(creator_setup, pset_creator) for _ in range(n_workers)]
    result_ids = [a.ray_remote_eval_batch.remote(func, b) for a, b in zip(actors,batches)]
    results = ray.get(result_ids)

    return sum(results, [])

#%% 
if __name__ == '__main__':
    from deap import creator, base, gp, tools
    import sys, os
    from functools import partial 
    

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ";" + os.environ.get("PYTHONPATH", "")
    
    ray.init(num_cpus=4, ignore_reinit_error=True)
    def creator_setup():
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness = creator.FitnessMax) 
    
    sys.path.append("..")
    from psetCreator import pset_creator

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
    creator_setup()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_= 1, max_ = 3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(100)
    func = len
    print(ray_deap_map(len, pop, creator_setup, partial(pset_creator, materialDataNames)))
# %%
