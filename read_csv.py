import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
if __name__ == '__main__':

    df=pd.read_csv("./PolSF/SF-RISAT/sim_results_combined.csv")
    comp_rgb=[]
    comp_rgb.append(df.iloc[7:,0].values)
    comp_rgb.append(df.iloc[7:,1].values)
    comp_rgb.append(df.iloc[7:,2].values)
    comp_rgb=np.stack(comp_rgb,axis=0)
    comp_rgb=comp_rgb.reshape(384,384,3)
    abs_rgb=np.abs(comp_rgb)
    plt.imshow(abs_rgb)
    plt.show()
