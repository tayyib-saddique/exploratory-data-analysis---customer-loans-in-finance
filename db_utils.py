from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import missingno as msno
import statsmodels.api as sm
import seaborn as sns
import yaml


def load_db_credentials(): 
    conf = yaml.safe_load(Path('credentials.yaml').read_text())
    return conf

class RDSDatabaseConnector():
    def __init__(self, database_credentials_dict) -> None:
        self.credentials = database_credentials_dict

    def connect_to_engine(self):
        engine = create_engine(f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}")
        engine.connect() 
        return engine 
    
    def create_df_from_database(self, table):
        connect_to_database = self.connect_to_engine()
        print("Connected to the database")

        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, connect_to_database)
        return df
    
    def to_csv(self, df):
        return df.to_csv('loan_payments.csv', index=False)

class DataFrameInfo():
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def info(self):
        return self.df.info()
    
    def describe(self):
        return self.df.describe()
    
    def shape(self):
        return self.df.shape()
    
    def data_types(self):
        return self.df.dtypes
    
    def column_describe(self, col : pd.Series):
        return self.df[col].describe()

    def count_nulls_in_columns(self, col :pd.Series):
        return self.df[col].isnull().sum()
    
    def percent_nulls_in_columns(self, col: pd.Series):
        return (self.df[col].isnull().sum() / self.df[col].size)

    def outliers_in_columns(self, col: pd.Series):
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*1.5)
        lower_bound = Q1 + (IQR*1.5)
        outliers = np.where((self.df[col] < lower_bound) | (self.df[col] > upper_bound))[0]
        return outliers

class DataTransform():
    def __init__(self, df : pd.DataFrame):
        self.df = df
    
    def convert_to_float(self, col):
        self.df[col] = self.df[col].astype(float)
        
    def convert_to_datetime(self, col):
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')


class DataFrameTransform():
    def __init__(self, df : pd.DataFrame):
        self.df = df

    def impute_mean(self, col : pd.Series):
        mean = self.df[col].mean()
        self.df[col].fillna(mean, inplace= True)
        return self.df[col]

    def impute_median(self, col : pd.Series):
        median = self.df[col].median()
        self.df[col].fillna(median, inplace= True)
        return self.df[col]
    
    def log_transform(self, col: pd.Series):
        new_col = self.df[col].map(lambda i: np.log(i) if i > 0.0 else 0.0)
        return self.df[new_col]
    
    def remove_outliers(self, col: pd.Series):
        outliers = DataFrameInfo.outliers_in_columns(self, col)
        self.df = self.df.drop(outliers)
        return self.df[col]
    
    def remove_correlated_columns(self, threshold = 0.95):
        col_corr = set()
        corr_matrix = Plotter.visualise_correlation_matrix(self).abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] 
                    col_corr.add(colname)
                    if colname in self.df.columns:
                        del self.df[colname]
        return self.df

class Plotter():
    def __init__(self, df : pd.DataFrame):
        self.df = df

    def qq_plot(self, col: pd.Series):
        """Generates Q-Q plot for column data."""
        return sm.qqplot(self.df[col])
     
    def histogram(self, col : pd.Series):
        return sns.histplot(self.df[col])
        
    def visualise_removed_NaN(self):
        return msno.bar(self.df)

    def visualise_skew(self, col: pd.Series):
        skew = self.df[col].skew()
        ax = self.df[col].plot.kde(bw_method=0.5)
        return skew, ax

    def visualise_outliers(self, col: pd.Series):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        flierprops = dict(marker='o', markerfacecolor='green', markersize=2,
                  linestyle='none')
        axs[0].boxplot(self.df[col], flierprops = flierprops)

        transformed_data = DataFrameTransform.remove_outliers(self, col)
        axs[1].boxplot(transformed_data)
        
        axs[0].set_title('Original Data')
        axs[1].set_title('Transformed Data')
        return fig, axs
    
    def visualise_correlation_matrix(self):
        return self.df.corr()


if __name__ == "__main__":
    credentials = load_db_credentials()
    db_connector = RDSDatabaseConnector(credentials)
    df_loan_payments = db_connector.create_df_from_database('loan_payments')
    db_connector.to_csv(df_loan_payments)
    data = pd.read_csv('loan_payments.csv') 
