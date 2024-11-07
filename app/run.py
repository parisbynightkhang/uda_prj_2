"""
Flask application
"""

import json

import joblib
import pandas as pd
import plotly
from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """
    Tokenizes a given text into individual words, normalizes them, and lemmatizes them.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list: A list of tokenized, normalized, and lemmatized words.
    """
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


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("disaster_messages", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    # Calculate the distribution of classes with value 1
    class_1_distribution = df.drop(
        ["id", "message", "original", "genre"], axis=1
    ).sum() / len(df)
    # Sort class_1_distribution
    class_1_distribution = class_1_distribution.sort_values(ascending=False)

    # Calculate the distribution of classes with value 0
    class_0_distribution = 1 - class_1_distribution

    # Get the names of the categories
    categories_name = list(class_0_distribution.index)

    # create visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [
                Bar(x=categories_name, y=class_1_distribution, name="Class=1"),
                Bar(x=categories_name, y=class_0_distribution, name="Class=0"),
            ],
            "layout": {
                "title": "Distribution of classes",
                "yaxis": {"title": "Distribution"},
                "xaxis": {"title": "Class"},
                "barmode": "stack",
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html",
        query=query,
        classification_result=classification_results,
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
