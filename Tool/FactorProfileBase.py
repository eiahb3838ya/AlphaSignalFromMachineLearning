# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:18:05 2020

@author: Evan
@reviewer: Robert
"""
from abc import abstractmethod
import abc

class FactorProfileBase(object, metaclass=abc.ABCMeta):
    def __init__(self):
        """
        Parameter
        ----------------------------
        functionName: str
        datasetName_list: list[str]
        parameters_dict: dict()
        """
        
        self.factorName = ""
        self.functionName = ""
        self.reliedDatasetNames = []
        self.parameters = {}
        
        
        
    @abstractmethod        
    def get_relied_dataset(self):
        pass
        
    def get_factor_kwargs(self, verbose = 0):
        """
        Parameter
        -------------------------------
        verbose: boolean, if verbose is True, return factorName, parameters,dataset; otherwise, return return factorName, parameters
        """
        out = dict()
        out.update({'factorName':self.factorName})
        out.update({'functionName':self.functionName})
        out.update({'reliedDatasetNames':self.reliedDatasetNames})
        out.update({'parameters':self.parameters})
        
        if verbose == 0:
            return(out)
        elif verbose == 1:
            out.update({'dataset':self.dataset})
            return(out)
        else:
            raise ValueError('verbose can only be 0 or 1.')
        
