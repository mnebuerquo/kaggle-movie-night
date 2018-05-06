import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import csv
import time
import sys


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
    return algo.fit(data.build_full_trainset())


def predict(test, algo):
    output = [('index', 'rating')]
    for index, row in test.iterrows():
        prediction = algo.predict(row['user id'], row['movie id'])
        output.append((row['test_id'], prediction.est))
    return output


def save(output):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'kmn-sa-{}.csv'.format(timestr)
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(output)


def main(args):
    full = make_full_dataset("data/u_train.data")
    algo = five_fold(full)
    test = load_test("data/u_test.data")
    output = predict(test, algo)
    save(output)


if __name__ == "__main__":
    main(sys.argv[1:])
