"""
Process data
"""

import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge disaster messages and categories data from CSV files.

    Parameters:
    messages_filepath (str): The file path to the CSV file containing disaster messages.
    categories_filepath (str): The file path to the CSV file containing disaster categories.

    Returns:
    pd.DataFrame: A dataframe containing the merged disaster messages and categories data.
    """
    # Load data_messages from messages_filepath
    data_messages = pd.read_csv(messages_filepath)
    # Load data_categories from categories_filepath
    data_categories = pd.read_csv(categories_filepath)
    # Merge 2 dataframes
    output_dataframe = data_messages.merge(data_categories, on=["id"])

    return output_dataframe


def clean_data(df_concat):
    """
    Clean the merged disaster messages and categories dataframe.

    Parameters:
    df_concat (pd.DataFrame): A dataframe containing merged disaster messages and categories data.

    Returns:
    pd.DataFrame: A cleaned dataframe with the specified operations applied.
    """
    # Create df_categories from the category columns and separate them with ;
    df_categories = df_concat["categories"].str.split(";", expand=True)
    # Create new column names for df_categories from the first row of df_categories
    category_colnames = df_categories.iloc[0].apply(lambda x: x[:-2]).tolist()

    # Update column names for each category
    df_categories.columns = category_colnames

    df_categories = df_categories.apply(lambda x: x.astype(str).str[-1]).astype(int)
    df_concat = pd.concat([df_concat, df_categories], axis=1)

    # Drop the categories column
    df_concat.drop("categories", axis=1, inplace=True)

    # Remove rows in df_concat when related column has value 2
    df_concat = df_concat[df_concat["related"] != 2]

    # Remove rows where the message column has a NaN value
    df_concat = df_concat.dropna(subset=["message"])

    return df_concat


def save_data(df, database_filename):
    """
    Save the cleaned disaster messages and categories data to a SQLite database.

    Parameters:
    df (pd.DataFrame): A dataframe containing cleaned disaster messages and categories data.
    database_filename (str): The filename of the SQLite database to save the data to.

    Returns:
    None
    """
    engine_data = create_engine("sqlite:///" + str(database_filename))
    df.to_sql("disaster_messages", engine_data, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath,
                categories_filepath,
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
