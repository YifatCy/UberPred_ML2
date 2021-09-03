from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from auxiliary_functions import get_x_any_y_advanced_creative, get_x_any_y, get_x_any_y_years
import datetime
plt.style.use('ggplot')
pd.set_option('display.max_rows', 100,'display.max_columns',None)


X_COLUMNS = ['hour', 'month','day', 'day_literal', 'spd', 'vsb','temp', 'dewp', 'slp', 'sd', 'hday']
Y_COLUMN = 'loadrank'

TEST_SIZE = 0.3


def get_x_any_y(df, dates, y_column):
    x, y = [], []
    for date in dates:
        day_df = df[df['date'] == date]
        x.append(day_df.drop([y_column,'pickups','date','year'], axis=1).to_numpy())
        y.append(day_df[y_column].to_numpy())
    return x, y


def prepare():
    data= pd.read_csv("datasets/uber_nyc_enriched.csv",index_col=False)
    mm = []
    dd = []
    yy = []
    hrs = []
    dow = []
    dow_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    dt_time = data['pickup_dt'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        mm.append(dt_time[i].split(" ")[0].split("-")[1])

    dt_time = data['pickup_dt'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        dd.append(dt_time[i].split(" ")[0].split("-")[2])

    dt_time = data['pickup_dt'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        yy.append(dt_time[i].split(" ")[0].split("-")[0])

    dt_time = data['pickup_dt'].copy(deep=True).to_numpy()
    for i in range(len(dt_time)):
        hrs.append(int(dt_time[i].split(" ")[1].split(":")[0]))

    for i in range(len(mm)):
        dow.append(dow_list[datetime.date(int(yy[i]),
                                          int(mm[i]),
                                          int(dd[i])).weekday()])

    data['month'] = mm
    data['day'] = dd
    data['year'] = yy
    data['hour'] = hrs
    data['day_literal'] = dow
    data['date'] = data['year'] + data['month'] + data['day']
    del data['pickup_dt']
    del data['pcp01']
    del data['pcp06']
    del data['pcp24']
    data.to_csv("datasets/uber_new.csv",index=False)
#prepare()


def change_to_numbers():
    dow_dict = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    borough_dict = {"Bronx": 0, "Brooklyn": 1, "EWR": 2, "Manhattan": 3, "Queens": 4, "Staten Island": 5, "NA": 6}
    df = pd.read_csv("datasets/uber_new.csv")
    df = df.replace(dow_dict)
    df = df.replace(borough_dict)
    df = df.replace({'Y': 1, 'N': 0})
    df.to_csv("datasets/uber_hour_changing_num.csv", index=False)
    return df


def group_by_borough(df):
    df1 = df.groupby(['date', 'hour', 'month', 'day', 'day_literal', 'year', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'sd', 'hday'], observed=True)['pickups'].sum().reset_index()
    return df1


def prepare_categorized_dataset():
    prepare()
    df = change_to_numbers()
    # to group_by_borough
    df = group_by_borough(df)
    max_count = max(df['pickups'])
    # print(max_count)
    max_range = range(max_count)
    quantile = np.quantile(max_range, q=[0.25, 0.5, 0.75])
    df['loadrank'] = df["pickups"].apply(
        lambda x: 0 if x <= quantile[0] else (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3))))
    # print(df_agg)
    df.to_csv("datasets/uber_hour_categorized.csv",index=False)
    return df
#prepare_categorized_dataset()


def prepare_train_test(categorized=False, scale=True, **kwargs):
    seed = kwargs.get("seed", 57)

    df = prepare_categorized_dataset()
    X = df.drop([Y_COLUMN,"pickups", 'date','year'], axis=1)
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
        train_set_x = train_set.drop([Y_COLUMN, "pickups", 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y