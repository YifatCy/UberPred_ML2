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
        x.append(day_df.drop([y_column,'pickup_dt','pickups','date','year'], axis=1).to_numpy())
        y.append(day_df[y_column].to_numpy())
    return x, y


def get_x_any_y_years(df, years, y_column):
    x, y = [], []
    for year in years:
        years_df = df[df['year'] == year]
        x.append(years_df.drop([y_column, 'pickup_dt','date', 'year'], axis=1).to_numpy())
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
    #del data['pickup_dt']  ## TODO check if it affects the others
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
    #df1 = df.groupby(['date', 'hour', 'month', 'day', 'day_literal', 'year', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'sd', 'hday'], observed=True)['pickups'].sum().reset_index()
    #df1.to_csv("1.csv")
    # TODO check if it affects the others
    df1 = df.groupby(['pickup_dt','date', 'hour', 'month', 'day', 'day_literal', 'year', 'spd', 'vsb', 'temp', 'dewp', 'slp', 'sd', 'hday'], observed=True)['pickups'].sum().reset_index()
    #df2.drop(['pickup_dt'],axis=1).to_csv("2.csv")
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


def prepare_train_test(categorized=False, scale=True, **kwargs):
    seed = kwargs.get("seed", 57)

    df = prepare_categorized_dataset()
    X = df.drop([Y_COLUMN,"pickup_dt","pickups", 'date','year'], axis=1)
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
        train_set_x = train_set.drop([Y_COLUMN, "pickup_dt","pickups", 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)
        print(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y

def prepare_dataset_advanced():
    df = prepare_categorized_dataset()

    #df = pd.read_csv(r'datasets/uber_nyc_enriched.csv')
    #df["cnt_categories"] = df["pickups"].apply(lambda x: 0 if x < 450 else (1 if x < 1400 else 2))
    #df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))
    #df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    #df["date"] = df['timestamp'].apply(lambda x: int(x.strftime('%d%m%Y')))
    #df["month"] = df['timestamp'].apply(lambda x: int(x.strftime('%m')))
    #df["day"] = df['timestamp'].apply(lambda x: int(x.strftime('%d')))
    df['pickup_dt'] = pd.to_datetime(df['pickup_dt'])
    df['year_week'] = df['pickup_dt'].apply(lambda x: str(x.isocalendar()[0])+'_'+str(x.isocalendar()[1]))
    df['week_day'] = df['pickup_dt'].apply(lambda x: int(x.isocalendar()[2]))

    df.drop(['pickup_dt', 'date'], axis=1, inplace=True)
    return df

prepare_dataset_advanced()

def prepare_grouped_data_advanced(num_of_hours):

    df = prepare_dataset_advanced()
    dates_in_data = df['year_week'].unique()
    train_weeks, test_weeks = train_test_split(dates_in_data,
                                               test_size=TEST_SIZE,
                                               random_state=57)

    train_x, train_y = get_x_any_y_advanced_creative(df, train_weeks, Y_COLUMN, num_of_hours)
    test_x, test_y = get_x_any_y_advanced_creative(df, test_weeks, Y_COLUMN, num_of_hours)


    return train_x, train_y, test_x, test_y


def divide_data_to_two_years(scale=True):
    df = prepare_categorized_dataset()

    train_years = [2015]
    test_years = [2016]

    train_x, train_y = get_x_any_y_years(df, train_years, Y_COLUMN)
    test_x, test_y = get_x_any_y_years(df, test_years, Y_COLUMN)

    if scale:
        train_set = df[df['year'].isin(train_years)]
        train_set_x = train_set.drop([Y_COLUMN,'pickup_dt', 'date', 'year'], axis=1)

        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(year) for year in train_x]
        scaled_test_x = [scalar.transform(year) for year in test_x]

        return np.array(scaled_train_x), np.array(train_y), np.array(scaled_test_x), np.array(test_y)

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def value_to_index(lst):

    res = 0
    for index, value in enumerate(reversed(lst)):
        res += value * 3 ** index

    return res

def get_x_any_y_advanced_creative(df, weeks, y_column, k):

    x, y = {}, {}

    assert 24 % k == 0

    for week_index, week in enumerate(weeks):

        x[week_index] = []
        y[week_index] = []

        x_week = {i: [] for i in range(24//k)}
        y_week = {i: [] for i in range(24//k)}

        week_df = df[df['year_week'] == week]
        for week_day in sorted(week_df['week_day'].unique()):

            x_day_array = week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).to_numpy()

            y_hours_labels = {week_df[week_df['week_day'] == week_day]['hour'].iloc[i]: week_df[week_df['week_day'] == week_day][y_column].iloc[0] for i in range(len(week_df[week_df['week_day'] == week_day].index))}
            x_hours_vectors = {week_df[week_df['week_day'] == week_day]['hour'].iloc[i]: week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).iloc[0].to_numpy() for i in range(len(week_df[week_df['week_day'] == week_day].index))}


            hour_index = list(week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).columns).index('hour')

            # No missing hours:
            if len(x_day_array) == 24:
                for block_index in range(24 // k):
                    hours_vectors = [x_hours_vectors[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]
                    x_week[block_index].append(list(itertools.chain(*hours_vectors)))

                    y_week[block_index].append(value_to_index([y_hours_labels[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]))

            # missing hours. should discard windows with missing hours.
            else:
                missing_hours = [hour for hour in range(24) if hour not in x_day_array[:, hour_index]]
                missing_blocks = set()

                for block_index in range(24 // k):
                    for block_hour in range(k * block_index, k * (block_index + 1), 1):
                        if block_hour in missing_hours:
                            missing_blocks.add(block_index)

                for block_index in range(24 // k):
                    if block_index not in missing_blocks:

                        hours_vectors = [x_hours_vectors[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]
                        x_week[block_index].append(list(itertools.chain(*hours_vectors)))

                        y_week[block_index].append(value_to_index([
                            y_hours_labels[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]))


        for block_index in range(24 // k):
            x[week_index].append(x_week[block_index])
            y[week_index].append(np.array(y_week[block_index]))


    return x, y
