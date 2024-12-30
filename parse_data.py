# libraries
import pandas as pd
import time


# fbref table link


def extract_data(df: pd.DataFrame, url_df: str, file_name: str) -> pd.DataFrame:
    """
    Extracts data from a given url and returns a pandas DataFrame
    """

    df = pd.read_html(url_df)[0]
    df.columns = [' '.join(col).strip() for col in df.columns]
    df = df.reset_index(drop=True)
    new_columns = []
    for col in df.columns:
        if 'level_0' in col:
            new_col = col.split()[-1]  # takes the last name
        else:
            new_col = col
        new_columns.append(new_col)
    # rename columns
    df.columns = new_columns
    df = df.fillna(0)
    print(f"\n\n\n\n*******************************************\nColumns and values in the {file_name}: \n\n\n\n\n")
    print(df.head())
    print(df.keys())
    df.to_excel(f'data/{file_name}.xlsx', index=False)
    return df

def main():
    url_df_possesion = 'https://fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats'
    urf_df_stats= "https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats"
    url_df_gca= 'https://fbref.com/en/comps/Big5/gca/squads/Big-5-European-Leagues-Stats'

    df_possesion = extract_data(df=pd.DataFrame(), url_df=url_df_possesion, file_name='squadPossesion')
    # delay the extraction of the data for http requests to be made
    time.sleep(5)
    df_stats = extract_data(df=pd.DataFrame(), url_df=urf_df_stats, file_name='squadStats')
    time.sleep(5)
    df_gca = extract_data(df=pd.DataFrame(), url_df=url_df_gca, file_name='squadGCA')
    new_columns=[]
    print("\n\nAdjusting the columns in the squad dataframes: \n")
    for col in df_stats.columns:
        if 'level_0' in col:
            new_col = col.split()[-1]  # takes the last name
        else:
            new_col = col
            new_col = ''.join(col.split())  # Remove all spaces
        new_columns.append(new_col)
    # rename columns
    df_stats.columns = new_columns


    # Merge the dataframes together
    df = pd.merge(df_possesion, df_stats, on='Squad')
    df = pd.merge(df, df_gca, on='Squad')
    df.to_excel('data/squadData.xlsx', index=False)
    # Rename the GF column to Goals
    df = df.rename(columns={'GF': 'Goals'})
    # Rename the GA column to Goals and Assists
    df = df.rename(columns={'GA': 'Goals and Assists'})

    print("*******************************************\nColumns and values in the Main DataFrame: \n")
    print(f"\n\n\n\n\n\nDescription:\n{df.describe()}")
    print(f"\n\n\n\n\n\nColumns:\n{df.keys()}")
    print(f"\n\n\n\n\n\n\n\n\n")
if __name__ == '__main__':
    print("Calling main: ")
    main()
    print("Done")
    