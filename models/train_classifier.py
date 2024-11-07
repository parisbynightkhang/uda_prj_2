"""
Train the model
"""

import pickle
import re
import sys

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Load data from a SQLite database and extract features and target variables for
    a disaster response machine learning model.

    Parameters:
    database_filepath (str): The path to the SQLite database file containing the disaster messages data.

    Returns:
    features (pandas.Series): A series of messages from the disaster messages data.
    target (pandas.DataFrame): A dataframe containing the categories of disaster messages as columns.
    category_names (list): A list of category names extracted from the target dataframe.
    """
    # Load data from database
    engine_data = create_engine("sqlite:///" + str(database_filepath))
    df_disaster_messages = pd.read_sql_table("disaster_messages", con=engine_data)

    # Extract features and target variables
    features = df_disaster_messages["message"]
    target = df_disaster_messages.drop(["message", "genre", "id", "original"], axis=1)
    category_names = target.columns.tolist()

    return features, target, category_names


def tokenize(text):
    """
    Tokenizes a given text by converting it to lowercase, replacing non-letter/numeric characters with spaces,
    removing extra spaces, parsing the text using nltk's word_tokenize, initializing a lemmatizer from nltk,
    and iterating through each token to lemmatize, normalize uppercase and lowercase letters,
    and remove leading and trailing spaces.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list_token (list): A list of tokens obtained after tokenizing the input text.
    """
    # Convert to lowercase, replace non-letter/numeric characters with spaces and remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Parsing text using nltk's word_tokenize
    word_token = word_tokenize(text)

    # Initialize word_net from nltk
    word_net = WordNetLemmatizer()

    # Iterate through each token
    list_token = []
    for token in word_token:
        # Lemmatize, normalize uppercase and lowercase letters, and remove leading and trailing spaces
        clean_token = word_net.lemmatize(token).lower().strip()
        list_token.append(clean_token)

    return list_token


def build_model():
    """
    This function creates a machine learning pipeline for text classification using GridSearchCV.
    The pipeline includes CountVectorizer, TfidfTransformer, and MultiOutputClassifier.
    GridSearchCV is used for hyperparameter tuning.

    Parameters:
    None

    Returns:
    gscv (GridSearchCV): A GridSearchCV object with the defined pipeline and hyperparameters.
    """
    # Create the pipeline model
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize, token_pattern=None)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    # Create the hyperparameter
    parameters = {
        "tfidf__use_idf": (True, False),
        "clf__estimator__n_estimators": [10, 20],
        "clf__estimator__min_samples_split": [2, 4],
    }

    # Create the gridsearch object
    clf = GridSearchCV(pipeline, param_grid=parameters)

    return clf


def evaluate_model(model, X_test, Y_test, category_names):  # noqa
    """
    Evaluates a trained machine learning model using classification reports for each category.

    Parameters:
    model (sklearn.pipeline.Pipeline): The trained machine learning model to be evaluated.
    X_test (pandas.Series): A series of test messages used to evaluate the model.
    Y_test (pandas.DataFrame): A dataframe containing the true categories of the test messages.
    category_names (list): A list of category names extracted from the target dataframe.

    Returns:
    None. The function prints the classification report for each category.
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, ": ", classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Saves a trained machine learning model to a pickle file.

    Parameters:
    model (sklearn.pipeline.Pipeline): The trained machine learning model to be saved.
    model_filepath (str): The path to the pickle file where the model will be saved.

    Returns:
    None. The function saves the trained model to the specified pickle file.
    """
    pickle.dump(model.best_estimator_, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
