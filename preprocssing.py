from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from auxiliary_functions import get_x_any_y_advanced_creative, get_x_any_y, get_x_any_y_years
import datetime

X_COLUMNS = ['year', 'month', 'day','day literal', 'hour']
Y_COLUMN = 'loadrank'

TEST_SIZE = 0.3


def get_x_any_y(df, dates, y_column):
    x, y = [], []
    for date in dates:
        day_df = df[df['date'] == date]
        x.append(day_df.drop([y_column, 'date', 'year'], axis=1).to_numpy())
        y.append(day_df[y_column].to_numpy())
    return x, y


def prepare():
    apr = pd.read_csv("datasets/uber-raw-data-apr14.csv")
    may = pd.read_csv("datasets/uber-raw-data-may14.csv")
    jun = pd.read_csv("datasets/uber-raw-data-jun14.csv")
    jul = pd.read_csv("datasets/uber-raw-data-jul14.csv")
    aug = pd.read_csv("datasets/uber-raw-data-aug14.csv")
    sep = pd.read_csv("datasets/uber-raw-data-sep14.csv")

    data = [apr,may,jun,jul,aug,sep]
    data = pd.concat(data)

    mm = []
    dd = []
    yy = []
    hrs = []
    dow = []
    dow_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_dict = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}

    dt_time = data['Date/Time'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        mm.append(dt_time[i].split(" ")[0].split("/")[0])

    dt_time = data['Date/Time'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        dd.append(dt_time[i].split(" ")[0].split("/")[1])

    dt_time = data['Date/Time'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        yy.append(dt_time[i].split(" ")[0].split("/")[2])

    dt_time = data['Date/Time'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        hrs.append(dt_time[i].split(" ")[1].split(":")[0])

    for i in range(len(mm)):
        dow.append(dow_list[datetime.date(int(yy[i]),
                                          int(mm[i]),
                                          int(dd[i])).weekday()])


    data['month'] = mm
    data['day'] = dd
    data['year'] = yy
    data['hour'] = hrs
    data['day literal'] = dow
    data = data.replace(dow_dict)
    data['date'] = data['day'] + data['month'] + data['year']
    del data['Date/Time']
    data.to_csv("datasets/uber.csv")

def prepare_dataset():
    #prepare()
    df = pd.read_csv("datasets/uber.csv")
    df = df[['year','month','day','day literal','date','hour']]
    df_agg = df.groupby(['year','month','day','day literal','date','hour']).size().reset_index(name='counts')
    return df_agg


def prepare_categorized_dataset():
    #prepare()
    df = pd.read_csv("datasets/uber.csv")
    df = df[['year', 'month', 'day','day literal', 'date', 'hour']]
    df_agg = df.groupby(['year', 'month', 'day','day literal','date', 'hour']).size().reset_index(name='counts')
    max_count = max(df_agg['counts'])
    # print(max_count)
    max_range = range(max_count)
    quantile = np.quantile(max_range, q=[0.25, 0.5, 0.75])
    df_agg['loadrank'] = df_agg["counts"].apply(
        lambda x: 0 if x <= quantile[0] else (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3))))
    # print(df_agg)
    df_agg.to_csv("datasets/uber_hour_categorized.csv")
    print(df_agg)
    return df_agg
#prepare()



def prepare_train_test(categorized=False, scale=True, **kwargs):
    seed = kwargs.get("seed", 57)

    df = prepare_categorized_dataset()

    X = df.drop([Y_COLUMN,'counts', 'date','year'], axis=1)
    Y = df[Y_COLUMN]

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = TEST_SIZE, random_state = seed)

    if scale:
        scalar = StandardScaler()
        scalar.fit(train_x)

        scaled_train_x = scalar.transform(train_x)
        scaled_test_x = scalar.transform(test_x)
        return scaled_test_x, test_y, scaled_train_x, train_y

    return test_x, test_y, train_x, train_y


def prepare_grouped_data(categorized=False, scale=True):
    df = prepare_categorized_dataset()
    dates_in_data = np.unique(df['date'])
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=TEST_SIZE,
                                             random_state=57)

    train_x, train_y = get_x_any_y(df, train_days, Y_COLUMN)
    test_x, test_y = get_x_any_y(df, test_days, Y_COLUMN)

    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.drop([Y_COLUMN, 'counts', 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y