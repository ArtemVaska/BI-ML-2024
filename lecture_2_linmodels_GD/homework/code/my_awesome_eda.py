import numpy as np
import pandas as pd


def fill_df_dtypes_dict(df: pd.DataFrame) -> dict:
    """
    Fills df_dtypes_dict with column names and it dtypes. Also predict possible types of data.

    Additional function for run_eda.

    :param df: a dataframe to analyze
    :return: a dictionary with information about column types
    """

    df_dtypes_dict = {}
    for i in range(len(df.dtypes)):
        df_dtypes_dict[df.dtypes.index[i]] = [df.dtypes.iloc[i].name]

    for column_name, column_type in df_dtypes_dict.items():
        if column_type[0] == 'object':
            if df[column_name].nunique() <= 10:
                df_dtypes_dict[column_name].append('factor')
            else:
                df_dtypes_dict[column_name].append('string')
        elif column_type[0] == 'int64':
            if df[column_name].nunique() <= 5:
                df_dtypes_dict[column_name].append('probably factor')
            else:
                df_dtypes_dict[column_name].append('numerical')
        else:
            df_dtypes_dict[column_name].append('numerical')
    return df_dtypes_dict


def process_factor_features(df: pd.DataFrame, df_dtypes_dict: dict) -> dict:
    """
    Analyzes all factor features in dataframe from input.

    Additional function for run_eda.

    :param df: a dataframe to analyze
    :param df_dtypes_dict: a dictionary from fill_df_dtypes_dict function
    :return: a dictionary with information about factor features
    """

    df_factor_dict = {}
    for column_name, column_type in df_dtypes_dict.items():
        if column_type[1] == 'factor' or column_type[1] == 'probably factor':
            df_factor_dict[column_name] = []
            for factor_value in df[column_name].unique():
                counts = df[df[column_name] == factor_value].shape[0]
                frequences = counts / df.shape[0]
                df_factor_dict[column_name].append([factor_value, counts, frequences])
    return df_factor_dict


def process_numeric_features(df: pd.DataFrame, df_dtypes_dict: dict) -> dict:
    """
    Analyzes all numeric features in dataframe from input.

    Additional function for run_eda.

    :param df: a dataframe to analyze
    :param df_dtypes_dict: a dictionary from fill_df_dtypes_dict function
    :return: a dictionary with information about numeric features
    """

    df_numeric_dict = {}
    for column_name, column_type in df_dtypes_dict.items():
        if column_type[1] == 'numerical':
            iqr = df[column_name].quantile(0.75) - df[column_name].quantile(0.25)
            is_outlier = np.abs((df[column_name] - df[column_name].median()) / iqr) < 1.5
            n_outliers = (~is_outlier).sum()

            describe_var = df[column_name].describe().to_list()
            df_numeric_dict[column_name] = [
                describe_var[3],  # min
                describe_var[7],  # max
                describe_var[1],  # mean
                describe_var[2],  # std
                describe_var[4],  # q0.25
                describe_var[5],  # median
                describe_var[6],  # q0.75
                n_outliers  # number of outliers
                ]
    return df_numeric_dict


def run_eda(df: pd.DataFrame) -> None:
    """
    Prints various information about dataframe from input. Check README for more information.

    Additional functions:
        - fill_df_dtypes_dict
        - process_factor_features
        - process_numeric_features

    :param df: a dataframe to analyze
    :return: everything prints on the screen
    """

    print('Viber! I mean WhatsApp... Sorry, Hello!\n')
    print(f'Your dataframe has {df.shape[0]} rows and {df.shape[1]} columns.\n')

    df_dtypes_dict = fill_df_dtypes_dict(df)
    print('List of column names and types:')
    for column_name, column_type in df_dtypes_dict.items():
        print(f'  {column_name}: {column_type[0]} -- {column_type[1]}')

    df_factor_dict = process_factor_features(df, df_dtypes_dict)  #
    print(f'\nDataframe has {len(df_factor_dict)} factor features:')
    for column_name in df_factor_dict:
        print(f'  {column_name}:')
        for factor_value, counts, frequences in df_factor_dict[column_name]:
            print(f'    {factor_value}: count = {counts}, frequence = {frequences:.2f}')

    df_numeric_dict = process_numeric_features(df, df_dtypes_dict)
    print(f'\nDataframe has {len(df_numeric_dict)} numerical features:')
    for column_name in df_numeric_dict:
        print(f'  {column_name}:')
        print(f'    Min: {df_numeric_dict[column_name][0]}')
        print(f'    Max: {df_numeric_dict[column_name][1]}')
        print(f'    Mean: {df_numeric_dict[column_name][2]:.2f}')
        print(f'    Standart deviation: {df_numeric_dict[column_name][3]:.2f}')
        print(f'    Quantile 1: {df_numeric_dict[column_name][4]}')
        print(f'    Median: {df_numeric_dict[column_name][5]}')
        print(f'    Quantile 3: {df_numeric_dict[column_name][6]}')
        if df_numeric_dict[column_name][7] != 0:
            print(f'    Number of outliers (+- 1.5 IQR): {df_numeric_dict[column_name][7]}')

    print(f'\nTotal NA in dataframe: {df.isna().sum().sum()}')
    print(f'There are {(df.isna().sum() > 0).sum()} rows:')
    df_na = df.isna().sum()[df.isna().sum() > 0]
    for i in range(len(df_na)):
        print(f'  {df_na.index[i]}: {df_na.values[i]}')

    print(f'\nDataframe has {len(df)-len(df.drop_duplicates())} duplicates.')
    pass
