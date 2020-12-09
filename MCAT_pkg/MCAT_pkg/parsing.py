import numpy as np
import pandas as pd


def separate_categories(df, return_col = "time to catastrophe (s)"):
    ''' Separates fluorescent labeled and not labeled into
    two separate numpy arrays

    Parameters
    _______
    df : pandas DataFrame
    contains a column called "labeled" with the fluorescence information,
    as well as all other relevant information

    return_col : str (optional)
    the name of the column of the dataframe to return values from, defaults
    to "time to catastrophe (s)"

    Returns
    _______
    tuple containing separated data from the given column; fluorescent
    labeled first, then not labeled
    '''
    labeled = df.loc[df["labeled"] == True, return_col].values
    not_labeled = df.loc[df["labeled"] == False, return_col].values
    return (labeled, not_labeled)

def parse_df(filename):
    '''Parses microtubule catastrophe dataframe

    Parameters
    ______
    filename : str
    the filename of the data as a csv to read from

    Returns 
    ______
    pandas DataFrame containing parsed data, with NaN values filtered out
    '''
    df_raw = pd.read_csv((filename), comment = "#")
    df_parsed = pd.melt(df_raw, value_vars=list(df.columns.values),
                    var_name = "Concentration (uM)",
                    value_name = "Time to catastrophe (s)")
    final_df = df_parsed.dropna()
    return final_df
