import numpy as np
import pandas as pd
from sklearn import preprocessing
from datetime import datetime

df = pd.read_csv("genderdone.csv")


def encoder(dataframe):

    cat_colname = dataframe.select_dtypes(
        include=['object']).columns.values
    cat_uniq_colname = [col for col in dataframe[cat_colname]
                        if (dataframe[col].nunique() == 3)]
    unique_rows = np.unique(np.array(dataframe[cat_uniq_colname].values))

    ohe = preprocessing.OneHotEncoder(dtype=int)
    col_trans = ohe.fit_transform(dataframe[cat_uniq_colname]).toarray()

    dataframe_encod = pd.DataFrame(data=col_trans,
                                   index=range(len(col_trans)),
                                   columns=unique_rows)
    dataframe = dataframe.drop(cat_uniq_colname, axis=1)
    # Concat mean filled table and dropped dataframe
    dataframe = pd.concat([dataframe, dataframe_encod], axis=1)

    return dataframe


start_time = datetime.now()
dataframe_last = encoder(df)
end_time = datetime.now()
dataframe_last.to_csv("Multi-Gender-done.csv",index=False)
print('Duration: {}'.format(end_time - start_time))
