# Libraries and Utilities

import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------- load data -------------
df = pd.read_csv('/Users/bella/Desktop/Github/CRM_Analytics/data.csv',
                 encoding = 'unicode_escape',
                 dtype ={'CustomerID':str,
                         'InvoiceID': str},
                 parse_dates = ['InvoiceDate'],
                 infer_datetime_format=True)

# --------- check data -------------
def check_data(dataframe,head=5):
    print(" shape ".center(70,'-'))
    print(f'Rows: {dataframe.shape[0]}')
    print(f'Rows: {dataframe.shape[1]}')
    print(" TYPES ".center(70, '-'))
    print(dataframe.dtypes)
    print(" HEAD ".center(70, '-'))
    print(dataframe.head(head))
    print(" MISSING VALUES ".center(70,'-'))
    print(dataframe.isnull().sum())
    print(" DUPLICATED VALUES ".center(70, '-'))
    print(dataframe.duplicated().sum())
    print(" QUANTILES ".center(70, '-'))
    numeric_df = dataframe.select_dtypes(include=['number'])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

#check_data(df)

#  --------- Understand the data (charts) -------------
''' 
# Number of Orders by Countries
world_map = df[['CustomerID', 'InvoiceNo', 'Country']
              ].groupby(['CustomerID', 'InvoiceNo', 'Country']
                       ).count().reset_index(drop = False)
countries = world_map['Country'].value_counts()
data = dict(type='choropleth',
            locations = countries.index,
            locationmode = 'country names',
            z = countries,
            text = countries.index,
            colorbar = {'title':'Orders'},
            colorscale='Viridis',
            reversescale = False)

layout = dict(title={'text': "Number of Orders by Countries",
                     'y':0.9,
                     'x':0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'},
              geo = dict(resolution = 50,
                         showocean = True,
                         oceancolor = "LightBlue",
                         showland = True,
                         landcolor = "whitesmoke",
                         showframe = True),
             template = 'plotly_white',
             height = 600,
             width = 1000)

choromap = go.Figure(data = [data], layout = layout)
plot(choromap, validate = False)
'''
# Descriptive Statistics
def desc_stats(dataframe):
    desc_df = pd.DataFrame(index=dataframe.columns,
                           columns=dataframe.describe().T.columns,
                           data=dataframe.describe().T)
    # size
    f, ax = plt.subplots(figsize=(10,
                                  desc_df.shape[0] * 2))
    sns.heatmap(desc_df,
                annot=True,
                cmap="Greens",
                fmt='.2f',
                ax=ax,
                linecolor='white',
                linewidths=1.1,
                cbar=False,
                annot_kws={"size": 12})
    plt.xticks(size=18)
    plt.yticks(size=14,
               rotation=0)
    plt.title("Descriptive Statistics", size=14)
    plt.show()

#desc_stats(df.select_dtypes(include=['number']))


# --------- Data Preprocessing -------------

def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    #Detects outliers with IQR method and replaces with thresholds
    df_ = dataframe.copy()
    quartile1 = df_[variable].quantile(q1)
    quartile3 = df_[variable].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    df_.loc[(df_[variable] < low_limit), variable] = low_limit
    df_.loc[(df_[variable] > up_limit), variable] = up_limit

    return df_

def ecommerce_preprocess(dataframe):
    df_ = dataframe.copy()
    # Missing Values
    df_ = df_.dropna()
    # Cancelled Orders & Quantity
    df_ = df_[~df_['InvoiceNo'].str.contains('C', na=False)]
    df_ = df_[df_['Quantity'] > 0]
    # Replacing Outliers
    df_ = replace_with_thresholds(df_, "Quantity", q1=0.01, q3=0.99)
    df_ = replace_with_thresholds(df_, "UnitPrice", q1=0.01, q3=0.99)
    # Total Price
    df_["TotalPrice"] = df_["Quantity"] * df_["UnitPrice"]

    return df_

df = ecommerce_preprocess(df)
desc_stats(df.select_dtypes(include = [float, int]))

# save the processed data to new file
df.to_csv('/Users/bella/Desktop/Github/CRM_Analytics/processed_data.csv', index=False)

