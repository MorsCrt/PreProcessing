import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime

df = pd.read_csv("eksikveriler.csv")


def valmeanfiller(dataframe):
    global df
    # missval_colname = Which columns have missing values and numerical
    missval_colname = np.array([col for col in df.columns
                                if df[col].isnull().any()])
    numeric_colname = df.select_dtypes(
        include=['float64', 'int64']).columns.values
    # Intersection of the numeric_colname and missval_colname
    inter_colnames = np.intersect1d(missval_colname, numeric_colname)
    # Saving order name for after concat
    all_col_name = df.columns.values

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    df_misscol = dataframe[inter_colnames].values  # List to dataframe
    imputer = imputer.fit(df_misscol)
    df_misscol = imputer.transform(df_misscol)

    df_filled = pd.DataFrame(data=df_misscol,
                             index=range(len(df_misscol)),
                             columns=inter_colnames)

    df = df.drop(inter_colnames, axis=1)  # Drop columns(have missing values) on the main dataframe
    df = pd.concat([df, df_filled], axis=1)  # Concat mean filled table and dropped dataframe
    df = df[all_col_name]

    return df


start_time = datetime.now()
x = valmeanfiller(df)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# missval_colname = np.array([col for col in df.columns
#                        if df[col].isnull().any()])
#
# numeric_colname = df.select_dtypes(
#     include=['float64','int64']).columns.values
#
# inter_colnames = np.intersect1d(missval_colname,numeric_colname)
#
# print(inter_colnames)
# df_misscol = df[missval_colname].values
# print(df.select_dtypes(include=['float64','int64']).columns.values)
#


# missval_colname = np.array(df.columns[df.isna().any()].tolist())


# all_col_name = df.columns.values #Saving order name for after concat
# print(all_col_name)
#
# xy = df.dtypes = 'int64'
# print(xy)
# index_no = [dataframe.columns.get_loc(col) for col in missval_colname]  # Get missval_colname index

# for col, no in zip(missval_colname, index_no):
#     df.insert(no, col, df_filled)
#     print(df)


# for i in range(len(missval_colname)):
#     print(missval_colname[i])
#
# for col,no in zip(missval_colname, index_no ):
#     print("col",col)
#     print("index no",no)


# missval_colname = [col for col in df.columns if df[col].isnull().any()]
# print("Column Names: ", missval_colname)
#
# imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
# yas = df[missval_colname].values
#
# imputer = imputer.fit(yas)
# yas = imputer.transform(yas)
#
# sonuc = pd.DataFrame(data=yas,index=range(len(yas)),columns=missval_colname)
#
# print(sonuc)
