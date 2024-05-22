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


def correlation_heatmap(df, features, correlation_figsize):
    plt.figure(figsize=correlation_figsize)
    sns.set(font_scale=1)
    sns.heatmap(df[features].corr(), annot=True, cmap="crest", fmt=".2f")
    plt.title("Correlation heatmap", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=10)
    plt.show()


def histogram_boxplot(df, feature, hist_box_figsize):
    plt.figure(figsize=hist_box_figsize)
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


def na_barplot(proportions, barplot_figsize):
    plt.figure(figsize=barplot_figsize)
    sns.set(style="whitegrid")
    sns.barplot(x=proportions.index, y=proportions, color="#0077b6", ec="k")
    plt.title("Proportion of missing values for each variable", fontsize=10, fontweight="bold")
    plt.xlabel("Features", fontsize=8, fontweight="bold")
    plt.ylabel("Proportion of NAs", fontsize=8, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


def run_eda(df: pd.DataFrame,
            category_threshold=5,
            correlation_figsize=(6, 5),
            hist_box_figsize=(8, 4),
            barplot_figsize=(5, 2.5)
            ) -> None:
    """
    Launches EDA.

    :param df: table with data
    :param category_threshold: threshold for categorical variables
    :param correlation_figsize: size of correlation heatmap
    :param hist_box_figsize: size of histogram-boxplot
    :param barplot_figsize: size of barplot with NAs
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
    numerical_features = df_types.query("type == 'numerical'").index
    categorical_features = df_types.query("type == 'categorical'").index
    string_features = df_types.query("type == 'string'").index
    print(tabulate(df_types, headers="keys", tablefmt="pretty", floatfmt=".2f", colalign=("left", "center")))
    print(f"\nString features: {string_features.tolist()}")
    print(f"Categorical features: {categorical_features.tolist()}")
    print(f"Numerical features: {numerical_features.tolist()}")

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
    correlation_heatmap(df, numerical_features_subset, correlation_figsize)

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
        histogram_boxplot(df, feature, hist_box_figsize)

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
    na_barplot(proportions, barplot_figsize)

    # duplicates
    print(f"\nNumber of duplicates (rows): {df.duplicated().sum()}")
