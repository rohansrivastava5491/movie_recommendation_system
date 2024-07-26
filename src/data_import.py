import pandas as pd

def import_data():
    movies = pd.read_csv("data/raw/movies.csv")
    ratings = pd.read_csv("data/raw/ratings.csv")
    return movies, ratings
