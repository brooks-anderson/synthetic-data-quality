"""
Objective: Create a subset of the Acquire Valued Shoppers dataset that
is usable on a single machine.
"""

import pandas as pd
import numpy as np
import os

# update path to data
os.chdir("../data")

offers = pd.read_csv('offers.csv.gz')
customers = pd.read_csv('trainHistory.csv.gz')
trans_iter = pd.read_csv('transactions.csv.gz', chunksize=100000)

num_rows = 0
chunk_count = 0
transactions = pd.DataFrame()
for t in trans_iter:
    if chunk_count % 50 == 0:
        print(f"Chunk Count: {chunk_count}/3500")

    cat_t = t.loc[t.category.isin(offers.category)]
    df = cat_t.merge(customers, on=['id', 'chain'],
                     how='inner')

    transactions = pd.concat([transactions, df])
    num_rows += len(df)
    chunk_count += 1

print(f"Num rows: {num_rows}")

# sample customer ids
np.random.seed(11)
sample_ids = np.random.choice(transactions.id.unique(), size=2000,
                              replace=False)
transactions = transactions.loc[transactions.id.isin(sample_ids)]

# if there are more than 5000 (?) transactions per id, we assume the id
# represents an institution, not an individual

transactions_vc = transactions.value_counts('id')
institution_ids = transactions_vc.loc[
    transactions_vc.values > 5000].index
transactions = transactions.loc[~transactions.id.isin(institution_ids)]

# drop dept, repeater, market due to logical substitutions
transactions = transactions.drop(['dept', 'repeater', 'market'], axis=1)

# write to file
transactions.to_csv('merged_AVS.csv.gz', index=False,
                    compression='gzip')
