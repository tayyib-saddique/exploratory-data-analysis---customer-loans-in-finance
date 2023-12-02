from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import missingno as msno
import statsmodels.api as sm
from scipy import stats
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
        '''Returns general info about the dataframe'''
        return self.df.info()
    
    def describe(self):
        '''Returns descriptive statistics for the dataframe'''
        return self.df.describe()
    
    def shape(self):
        '''Returns shape of the dataframe'''
        return self.df.shape()
    
    def data_types(self):
        '''Returns all datatypes'''
        return self.df.dtypes
    
    def count_nulls(self):
        return self.df.isna().sum()
    
    def measure_skew(self):
        return self.df.skew(numeric_only=True)

    def column_describe(self, col : pd.Series):
        '''Highlights basic descriptive statistics for a column'''
        return self.df[col].describe()
    
    def check_nulls(self, col : pd.Series):
        if self.df[col].isna().sum() != 0:
            return True
        return False
    
    def check_skew_col(self, col : pd.Series):
        if self.df[col].skew() > 1:
            return True
        return False

    def count_nulls_in_columns(self, col :pd.Series):
        '''Returns the total count of NaN values in a column'''
        return self.df[col].isnull().sum()
    
    def percent_nulls_in_columns(self, col: pd.Series):
        '''Returns the percentage of NaN values in a column'''
        return (self.df[col].isnull().sum() / self.df[col].size)

    def outliers_in_columns(self, col: pd.Series):
        '''Highlights outliers in column based on bounds from quartiles'''
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
        '''Converts all values in column into float'''
        self.df[col] = self.df[col].astype(float)
        
    def convert_to_datetime(self, col):
        '''Forcefully converts all values in column into datetime format'''
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
class DataFrameTransform():
    def __init__(self, df : pd.DataFrame):
        self.df = df

    def impute_mean(self, col : pd.Series):
        '''Replaces all NaN values with the mean value of the column'''
        mean = self.df[col].mean()
        self.df[col].fillna(mean, inplace= True)
        return self.df[col]

    def impute_median(self, col : pd.Series):
        '''Replaces all NaN values with the median value of the column'''
        median = self.df[col].median()
        self.df[col].fillna(median, inplace= True)
        return self.df[col]
    
    def remove_NaN(self, col: pd.Series):
        return self.df[col].dropna()
    
    def log_transform(self, col: pd.Series):
        '''Transforms distribution of data using log transformation'''
        self.df[col] = self.df[col].map(lambda i: np.log(i) if float(i) > 0.0 else 0.0)
        return self.df[col]
    
    def box_cox_transform(self, col: pd.Series):
        '''Performes Box-Cox transformation'''
        return stats.boxcox(self.df[col])
        
    def yeo_johnson_transform(self, col: pd.Series):
        '''Performes Yeo-Johnson transformation'''
        return stats.yeojohnson(self.df[col])

    def remove_outliers(self, col: pd.Series):
        '''Removes any outliers from dataframe'''
        outliers = DataFrameInfo.outliers_in_columns(self, col)
        self.df = self.df.drop(outliers)
        return self.df[col]
    
    def remove_correlated_columns(self, threshold = 0.99):
        '''Removes any columns with a correlation threshold > 0.95 by default unless otherwise specified'''
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
        '''Generates QQ Plot for column'''
        return sm.qqplot(self.df[col])
     
    def histogram(self, col : pd.Series):
        '''Generates Histogram for column'''
        return sns.histplot(self.df[col])
        
    def visualise_NaN(self):
        '''Visualises NaN data'''
        return msno.bar(self.df)

    def visualise_skew(self, col: pd.Series):
        '''Visualises skew in distribution'''
        skew = self.df[col].skew()
        ax = self.df[col].plot.kde(bw_method=0.5)
        return skew, ax

    def visualise_outliers(self, col: pd.Series):
        '''Visualises outliers using boxplots showing transformation in data'''
        fig, axs = plt.subplots(nrows=1, ncols=2)
        flierprops = dict(marker='o', markerfacecolor='green', markersize=2,
                  linestyle='none')
        axs[0].boxplot(self.df[col], flierprops = flierprops)

        transformed_data = DataFrameTransform.remove_outliers(self, col)
        axs[1].boxplot(transformed_data)
        
        axs[0].set_title(f'Original Data - {col}')
        axs[1].set_title(f'Transformed Data - {col}')
        return fig, axs
    
    def visualise_correlation_matrix(self):
        '''Visualises correlation matrix'''
        return self.df.corr()


if __name__ == "__main__":
    credentials = load_db_credentials()
    db_connector = RDSDatabaseConnector(credentials)
    df_loan_payments = db_connector.create_df_from_database('loan_payments')
    db_connector.to_csv(df_loan_payments)
    data = pd.read_csv('loan_payments.csv')

