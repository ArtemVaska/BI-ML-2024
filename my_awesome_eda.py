import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

plt.rcParams.update({'font.weight': 'normal'})


def define_types(df, category_threshold):
    column_types = []
    for column in df.columns:
        if df[column].nunique() <= category_threshold:
            column_types.append("categorical")
        else:
            match df[column].dtype:
                case "int64":
                    column_types.append("numerical")
                case "float64":
                    column_types.append("numerical")
                case "object":
                    column_types.append("string")
                case "datetime64[ns]":
                    column_types.append("datetime")

    df_types = pd.DataFrame(columns=["dtype", "type"])
    df_types["dtype"] = df.dtypes
    df_types["type"] = column_types

    return df_types


def correlation_heatmap(df, features):
    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1)
    sns.heatmap(df[features].corr(), annot=True, cmap="crest", fmt=".2f")
    plt.title("Correlation heatmap", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=10)
    plt.show()


def histogram_boxplot(df, feature):
    plt.figure(figsize=(6, 3))
    sns.set(style="whitegrid")
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], bins=30, color="#0077b6", ec="k")
    plt.xlabel(feature, fontsize=8, fontweight="bold")
    plt.ylabel("Count", fontsize=8, fontweight="bold")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(False)

    # boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[feature], color="#0077b6")
    plt.xlabel(feature, fontsize=8, fontweight="bold")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(False)
    plt.show()


def na_barplot(proportions):
    plt.figure(figsize=(5, 2.5))
    sns.set(style="whitegrid")
    sns.barplot(x=proportions.index, y=proportions, color="#0077b6", ec="k")
    plt.title("Proportion of missing values for each variable", fontsize=10, fontweight="bold")
    plt.xlabel("Features", fontsize=8, fontweight="bold")
    plt.ylabel("Proportion of NAs", fontsize=8, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


def run_eda(df: pd.DataFrame, category_threshold=5) -> None:
    """
    Launches EDA.

    :param df: table with data
    :param category_threshold: threshold for categorical variables
    :return: prints the EDA of the data to the stdout
    """
    # greeting
    print(f"Here is EDA for your dataframe ;)\n")

    # number of observations(rows) and parameters(features, columns)
    n_observations = df.shape[0]
    n_features = df.shape[1]
    print(f"Observations (rows): {n_observations}")
    print(f"Parameters (features, columns): {n_features}\n")

    # data types
    print(f"Data types:")
    df_types = define_types(df, category_threshold)
    categorical_features = df_types.query("type == 'categorical'").index
    numerical_features = df_types.query("type == 'numerical'").index
    print(tabulate(df_types, headers="keys", tablefmt="pretty", floatfmt=".2f", colalign=("left", "center")))

    # statistic for categorical features
    print(f"\nStatistics for categorical features:")
    for feature in categorical_features:
        counts = df[feature].value_counts()
        freqs = counts / n_observations
        df_count = pd.DataFrame({"count": counts, "frequency": round(freqs, 2)})
        print(f"{tabulate(df_count, headers="keys", tablefmt="pretty", floatfmt=".2f")}")

    # statistic for numerical features
    print(f"\nDescriptive statistic for numerical features:")
    numerical_features_subset = numerical_features[~numerical_features.str.contains('id')]
    numerical_features_subset_stat = df.loc[:, numerical_features_subset].describe().round(2)
    print(f"{tabulate(numerical_features_subset_stat, headers="keys", tablefmt="pretty", floatfmt=".2f")}\n")

    # correlation heatmap
    correlation_heatmap(df, numerical_features_subset)

    # outliers
    print(f"\nOutliers for numerical features:")
    outliers = {}
    for feature in numerical_features_subset:
        q1, q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = sum(~df[feature].between(lower_bound, upper_bound))
        outliers[feature] = n_outliers
    print(f"{tabulate(outliers.items())}\n")

    # histograms and boxplots for each feature
    for feature in numerical_features_subset:
        histogram_boxplot(df, feature)

    # NAs
    print(f"\nMissing values (NAs):")
    total_na = df.isna().sum().sum()
    rows_na = df[df.isnull().any(axis=1)].shape[0]
    columns_na = df.columns[df.isnull().any()].values
    print(f"Total NA: {total_na}")
    print(f"Rows with NA: {rows_na}")
    print(f"Columns with NA: {columns_na}\n")

    # plot proportion of missing values for each variable
    proportions = df.isnull().mean()
    na_barplot(proportions)

    # duplicates
    print(f"\nNumber of duplicates (rows): {df.duplicated().sum()}")
