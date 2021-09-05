from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from auxiliary_functions import get_x_any_y_advanced_creative, get_x_any_y, get_x_any_y_years
import datetime
import itertools

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

def get_x_any_y_years(df, years, y_column):
    x, y = [], []
    for year in years:
        years_df = df[df['year'] == year]
        x.append(years_df.drop([y_column,'date', 'year'], axis=1).to_numpy())
        y.append(years_df[y_column].to_numpy())
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
    del data['pickup_dt']  ## TODO check if it affects the others
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

def group_by_ignore_borough(df):
    df1 = df.groupby(['date', 'hour', 'month', 'day', 'day_literal', 'year', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'sd', 'hday'], observed=True)['pickups'].sum().reset_index()
    return df1

def prepare_categorized_dataset():
    prepare()
    df = change_to_numbers()
    # to group_by_borough
    df = group_by_ignore_borough(df)
    max_count = max(df['pickups'])
    # print(max_count)
    max_range = range(max_count)
    quantile = np.quantile(max_range, q=[0.25, 0.5, 0.75])
    df['loadrank'] = df["pickups"].apply(
        lambda x: 0 if x <= quantile[0] else (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3))))
    # print(df_agg)
    df.to_csv("datasets/uber_hour_categorized.csv",index=False)
    return df

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

def prepare_grouped_data(scale=True):
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

def value_to_index(lst):

    res = 0
    for index, value in enumerate(reversed(lst)):
        res += value * 3 ** index

    return res

def over_sampling():
    ## over sampling
    data = prepare_categorized_dataset()
    num_0 = len(data[data[Y_COLUMN] == 0])
    num_1 = len(data[data[Y_COLUMN] == 1])
    num_2 = len(data[data[Y_COLUMN] == 2])
    num_3 = len(data[data[Y_COLUMN] == 3])
    #ratio = num_1 / num_0
    new_data = pd.DataFrame(columns=data.columns)
    cnt = 0
    for idx,point in data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        if point[Y_COLUMN] == 0 or point[Y_COLUMN] == 1:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            continue
        p = np.random.rand()
        if p > 0.5:
            cnt += 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
        else:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
    print('cnt ', cnt)
    print('over sampling ',len(new_data))
    #new_data = shuffle(new_data)
    for i in ['date','hour', 'month','day', 'day_literal', 'year','spd', 'temp', 'dewp', 'hday','pickups','loadrank']:
        new_data[i] = new_data[i].astype(int)
    new_data.to_csv('over_sampling.csv')
    print('over_sampling',new_data)
    df = new_data.groupby('loadrank')['pickups'].count().reset_index(name='sum_uber_pickups')
    df = df.set_index('loadrank')
    df.plot.bar(rot=0, title=f"Hist of Hours per Category [0, 1, 2, 3]", alpha=0.7, color='salmon')
    plt.ylabel('Sum Hours')
    plt.xlabel('Category')
    #plt.savefig('figures/hist_per_each_category.png')
    plt.show()
    return new_data

def under_sampling():
    ## under sampling
    data = prepare_categorized_dataset()
    # num_0 = len(data[data['song_popularity'] == 0])
    # num_1 = len(data[data['song_popularity'] == 1])
    # ratio = num_1 / num_0
    new_data = pd.DataFrame(columns=data.columns)
    cnt = 0
    for idx,point in data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        if point[Y_COLUMN] == 3 or point[Y_COLUMN] == 2:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            continue
        p = np.random.rand()
        if p >= 0.5:
            cnt += 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
    print('cnt ', cnt)
    print('under sampling ', len(new_data))
    #new_data = shuffle(new_data)
    for i in ['date','hour', 'month','day', 'day_literal', 'year','spd', 'temp', 'dewp', 'hday','pickups','loadrank']:
        new_data[i] = new_data[i].astype(int)
    print('under_sampling',new_data)
    new_data.to_csv('under_sampling.csv')
    print('over_sampling',new_data)
    df = new_data.groupby('loadrank')['pickups'].count().reset_index(name='sum_uber_pickups')
    df = df.set_index('loadrank')
    df.plot.bar(rot=0, title=f"Hist of Hours per Category [0, 1, 2, 3]", alpha=0.7, color='salmon')
    plt.ylabel('Sum Hours')
    plt.xlabel('Category')
    #plt.savefig('figures/hist_per_each_category.png')
    plt.show()
    return new_data

def prepare_grouped_data_over(categorized=False, scale=True):
    over_sampling()
    df = pd.read_csv("over_sampling.csv", index_col=False)
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

def prepare_grouped_data_under(categorized=False, scale=True):
    under_sampling()
    df = pd.read_csv("under_sampling.csv", index_col=False)
    dates_in_data = np.unique(df['date'])
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=TEST_SIZE,
                                             random_state=57)

    train_x, train_y = get_x_any_y(df, train_days, Y_COLUMN)
    test_x, test_y = get_x_any_y(df, test_days, Y_COLUMN)
    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.drop([Y_COLUMN,"pickups", 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y

def group_by_borough(df):
    df1 = df.groupby(['borough','date', 'hour', 'month', 'day', 'day_literal', 'year', 'spd', 'vsb', 'temp', 'dewp',
                      'slp', 'sd', 'hday'], observed=True)['pickups'].sum().reset_index()
    return df1

def prepare_categorized_dataset_creative():
    prepare()
    df = change_to_numbers()
    df = group_by_borough(df)
    max_count = max(df['pickups'])
    max_range = range(max_count)
    quantile = np.quantile(max_range, q=[0.25, 0.5, 0.75])
    df['loadrank'] = df["pickups"].apply(
        lambda x: 0 if x <= quantile[0] else (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3))))
    # print(df_agg)
    df.to_csv("datasets/uber_hour_categorized_by_borough.csv",index=False)
    return df

def get_boroughs_dict():
    return {"Bronx": 0, "Brooklyn": 1, "EWR": 2, "Manhattan": 3, "Queens": 4, "Staten Island": 5, "NA": 6}

def get_x_any_y_creative(df, dates, y_column):
    x, y = [], []
    day_bor_df = pd.DataFrame()
    dict = get_boroughs_dict()
    for bor in dict.values():
        for date in dates:
            day_df = df[df['date'] == date]
            day_bor_df = day_df[day_df['borough'] == bor]
            print(day_bor_df)
            x.append(day_bor_df.drop([y_column,'pickups','date','year'], axis=1).to_numpy())
            y.append(day_bor_df[y_column].to_numpy())
    return x, y

def prepare_grouped_data_creative(scale=True):
    df = prepare_categorized_dataset_creative()
    dates_in_data = np.unique(df['date'])
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=TEST_SIZE,
                                             random_state=57)

    train_x, train_y = get_x_any_y_creative(df, train_days, Y_COLUMN)
    test_x, test_y = get_x_any_y_creative(df, test_days, Y_COLUMN)
    print(train_x,'\n',train_y,'\n',test_x,'\n',test_y)
    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.drop([Y_COLUMN, "pickups", 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y
prepare_grouped_data_creative()
