import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


def load_item(filename):
    columns = ("movie id", "movie title", "release date", "video release date",
               "IMDb URL", "unknown", "Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime", "Documentary", "Drama",
               "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
               "Romance", "Sci-Fi", "Thriller", "War", "Western")
    data = pd.read_csv(filename, names=columns, skipinitialspace=True,
                       sep="|", na_values="?", index_col=False,
                       encoding='latin-1')
    return data


def load_user(filename):
    columns = ("user id", "age", "gender", "occupation", "zip code")
    data = pd.read_csv(filename, names=columns, skipinitialspace=True,
                       sep="|", na_values="?", index_col=False,
                       encoding='latin-1')
    return data


def load_data(filename):
    columns = ("user id", "movie id", "rating", "timestamp")
    data = pd.read_csv(filename, names=columns, skipinitialspace=True,
                       sep="\t", na_values="?", index_col=False,
                       encoding='latin-1')
    return data


def load_test(filename):
    columns = ("test_id", "user id", "movie id")
    data = pd.read_csv(filename, names=columns, skipinitialspace=True,
                       sep="\t", na_values="?", index_col=False,
                       encoding='latin-1')
    return data


def make_full_dataset(datafile):
    items = load_item("data/u.item")
    users = load_user("data/u.user")
    data = load_data(datafile)
    full = data.merge(items, how='left', on='movie id')
    full = full.merge(users, how='left', on='user id')
    return full


def five_fold(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user id', 'movie id', 'rating']],
                                reader)
    algo = SVD()
    out = cross_validate(
        algo,
        data,
        measures=[
            'RMSE',
            'MAE'],
        cv=5,
        verbose=True)
    print(out)


full = make_full_dataset("data/u_train.data")
five_fold(full)
print(full.head())
