#!/usr/bin/env python3

import pandas as pd
from sklearn.datasets import load_boston


#b Start define frequency_table.
def frequency_table(
    X,
    sort=True,
    ascending=False,
    bins=10,
    dropna=False,
):
    count = X.apply(
        lambda x: x.value_counts(
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )
    ).T.stack().reset_index()
    count.columns = ['Variable', 'Value', 'Count']
    count['Proportion'] = X.apply(
        lambda x: x.value_counts(
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
            normalize=True,
        )
    ).T.stack().to_numpy()
    return count
#b End define frequency_table.


#b Define create example data.
def create_example_data():
    boston = load_boston()
    data = pd.DataFrame(
        boston['data'],
        columns=boston['feature_names'],
    )
    return data
#b End create example data.


#b Define main.
def main():
    data = create_example_data()
    freq_table = frequency_table(data)
    freq_table.to_csv('output/freq_table.csv', index=None)
#b End main.

if __name__ == '__main__':
    main()

