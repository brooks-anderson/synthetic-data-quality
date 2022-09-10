"""
Split the reduced transactions file into a training and holdout set.
"""

import pandas as pd
import numpy as np
import os

os.chdir("../data")

merged = pd.read_csv('merged_avs.csv.gz')

np.random.seed(11)
holdout_ids = np.random.choice(merged.id, size=400, replace=False)

train = merged.loc[~merged.id.isin(holdout_ids)]
holdout = merged.loc[merged.id.isin(holdout_ids)]

train.to_csv('train.csv.gz', index=False, compression='gzip')
holdout.to_csv('holdout.csv.gz', index=False, compression='gzip')
