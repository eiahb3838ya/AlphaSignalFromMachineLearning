# %%
import pandas as pd
import os

data_path = "C:\\Users\\eiahb\\Documents\\MyFiles\\WorkThing\\tf\\01task\\GeneticProgrammingProject\\AlphaSignalFromMachineLearning\\data\\newdata"


# %%
os.listdir(data_path)
# %%
a_file = os.listdir(data_path)[-1]

pd.read_csv(os.path.join(data_path, a_file), index_col=0)


# %%
