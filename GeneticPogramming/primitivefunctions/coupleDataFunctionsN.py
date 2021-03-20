#%%
import numpy as np
from Tool.GeneralData import GeneralData
from GeneticPogramming import utils
from scipy.stats.stats import pearsonr   
import copy
import warnings

warnings.filterwarnings("ignore")

#%%
#TODO
# 𝑡𝑠_𝑐𝑜𝑟𝑟(𝑎, 𝑏, 𝑐) 3 过去 c 天 a 和 b 的相关系数

#%%
# def ts_corr(this: GeneralData, that: GeneralData, rollingDaysN: int = 2) -> GeneralData:
#     assert this.generalData.shape == that.generalData.shape
#     assert rollingDaysN > 0

#     outputToReturn = copy.copy(this)
#     toStride2DArray = outputToReturn.generalData
#     strided = utils.get_strided(toStride2DArray, rollingDaysN)
#     std = np.nanstd(strided, axis = 1)
#     outputToReturn.generalData = std
#     return outputToReturn

    # outputToReturn = copy.copy(this)
    # outputToReturn.generalData = np.maximum(this.generalData, that.generalData)
    # return outputToReturn


#%% simple test
if __name__ == '__main__':
    PROJECT_ROOT = "c:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\"
    import os
    os.chdir(PROJECT_ROOT)
    from Tool import globalVars
    from GetData import load_data
    
    globalVars.initialize()
    load_data("materialData",
        os.path.join(os.path.join(PROJECT_ROOT,"data"), "h5")
    )
#%%
    this = globalVars.materialData['close']
    that = globalVars.materialData['open']
    
    # 沒救
    # this_stride = utils.get_strided(this.generalData, 5)
    # that_stride = utils.get_strided(that.generalData, 5)
    # output = np.ndarray((that_stride.shape[0], that_stride.shape[2]))
    # for i in range(this_stride.shape[0]):
    #     print(i)
    #     output[i] = utils.rowwise_corrcoef(this_stride[i].T, that_stride[i].T)

# %%
