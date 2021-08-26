from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from auxiliary_functions import get_x_any_y_advanced_creative, get_x_any_y, get_x_any_y_years
import datetime

TEST_SIZE = 0.3
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
    del data['Date/Time']

    data.to_csv("datasets/uber.csv")

def base_agg():
    #prepare()
    df = pd.read_csv("datasets/uber.csv")
    df = df[['year','month','day','hour']]
    df_agg = df.groupby(['year','month','day','hour']).size().reset_index(name='counts')
    max_count = max(df_agg['counts'])
    print(max_count)
    max_range = range(max_count)
    quantile = np.quantile(max_range,q=[0.25,0.5,0.75])
    df_agg["loadrank"] = df_agg["counts"].apply(lambda x: 0 if x <= quantile[0] else (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3))))


base_agg()

# def show_hist():
#     df = pd.read_csv(r'london_merged.csv')
#     df["cnt"].hist(figsize=(5, 5), grid=False, bins=100)
#     buckets = 3
#     colors = ["red", "orange", "green"]
#     for i in range(1, buckets + 1):
#         print(df["cnt"].quantile(q=i / buckets))
#         plt.vlines(x=df["cnt"].quantile(q=i / buckets), ymin=0, ymax=1750, colors=colors[i - 1],
#                    label=f"{int(df['cnt'].quantile(q=i / buckets))} - {round(i / buckets, 2)} quantile")
#
#     plt.title("histogram of bicycles count")
#     plt.legend()
#     plt.xlabel("bike counts")
#     plt.show()


# def prepare_dataset():
#     df = pd.read_csv(r'london_merged.csv')
#     df["cnt_categories"] = df["cnt"].apply(lambda x: 0 if x < 450 else (1 if x < 1400 else 2))
#     df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))
#
#     df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
#     df["date"] = df['timestamp'].apply(lambda x: int(x.strftime('%d%m%Y')))
#     df["month"] = df['timestamp'].apply(lambda x: int(x.strftime('%m')))
#     df["day"] = df['timestamp'].apply(lambda x: int(x.strftime('%d')))
#
#     df["year"] = df['timestamp'].apply(lambda x: int(x.strftime('%Y')))
#
#     df.drop(['timestamp', 'cnt'], axis=1, inplace=True)
#     return df


# def prepare_categorized_dataset():
#     df = prepare()
#     for col in df:
#         if col == 'date':
#             continue
#         if len(df[col].unique()) > 24:
#             lower_barrier = df[col].quantile(q=1 / 3)
#             higher_barrier = df[col].quantile(q=2 / 3)
#             df[f'{col}_categorized'] = df[col]. \
#                 apply(lambda x: 'low' if x < lower_barrier else 'medium' if x < higher_barrier else 'high')
#             df.drop(col, axis=1, inplace=True)
#     return df

#
# def prepare_train_test(categorized=False, scale=True, **kwargs):
#     seed = kwargs.get("seed", 57)
#
#     df = prepare_categorized_dataset() if categorized else prepare_dataset()
#
#
#     X = df.drop(['agg', 'date', 'year'], axis=1)
#     Y = df['agg']
#
#     train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = TEST_SIZE, random_state = seed)
#
#     if scale:
#         scalar = StandardScaler()
#         scalar.fit(train_x)
#
#         scaled_train_x = scalar.transform(train_x)
#         scaled_test_x = scalar.transform(test_x)
#         return scaled_test_x, test_y, scaled_train_x, train_y
#
#     return test_x, test_y, train_x, train_y
#
#
# def prepare_grouped_data(categorized=False, scale=True):
#     df = prepare_categorized_dataset() if categorized else prepare_dataset()
#     dates_in_data = df['date'].unique()
#
#     train_days, test_days = train_test_split(dates_in_data,
#                                              test_size=TEST_SIZE,
#                                              random_state=57)
#
#     train_x, train_y = get_x_any_y(df, train_days, 'agg')
#     test_x, test_y = get_x_any_y(df, test_days, 'agg')
#
#     if scale:
#         train_set = df[df['date'].isin(train_days)]
#         train_set_x = train_set.drop(['agg', 'date', 'year'], axis=1)
#         scalar = StandardScaler()
#         scalar.fit(train_set_x)
#
#         scaled_train_x = [scalar.transform(day) for day in train_x]
#         scaled_test_x = [scalar.transform(day) for day in test_x]
#         return scaled_train_x, train_y, scaled_test_x, test_y
#
#     return train_x, train_y, test_x, test_y
#
#
# def prepare_dataset_advanced():
#     df = pd.read_csv(r'london_merged.csv')
#     df["cnt_categories"] = df["cnt"].apply(lambda x: 0 if x < 450 else (1 if x < 1400 else 2))
#     df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))
#
#     df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
#     df["date"] = df['timestamp'].apply(lambda x: int(x.strftime('%d%m%Y')))
#     df["month"] = df['timestamp'].apply(lambda x: int(x.strftime('%m')))
#     df["day"] = df['timestamp'].apply(lambda x: int(x.strftime('%d')))
#     df['year_week'] = df['timestamp'].apply(lambda x: str(x.isocalendar()[0])+'_'+str(x.isocalendar()[1]))
#     df['week_day'] = df['timestamp'].apply(lambda x: int(x.isocalendar()[2]))
#
#     df.drop(['timestamp', 'cnt', 'date'], axis=1, inplace=True)
#     return df
#
#
# def prepare_grouped_data_advanced(num_of_hours):
#
#     df = prepare_dataset_advanced()
#     dates_in_data = df['year_week'].unique()
#     train_weeks, test_weeks = train_test_split(dates_in_data,
#                                                test_size=TEST_SIZE,
#                                                random_state=57)
#
#     train_x, train_y = get_x_any_y_advanced_creative(df, train_weeks, Y_COLUMN, num_of_hours)
#     test_x, test_y = get_x_any_y_advanced_creative(df, test_weeks, Y_COLUMN, num_of_hours)
#
#
#     return train_x, train_y, test_x, test_y
#
#
# def divide_data_to_two_years(categorized=False, scale=True):
#     df = prepare_categorized_dataset() if categorized else prepare_dataset()
#
#     train_years = [2015]
#     test_years = [2016]
#
#     train_x, train_y = get_x_any_y_years(df, train_years, Y_COLUMN)
#     test_x, test_y = get_x_any_y_years(df, test_years, Y_COLUMN)
#
#     if scale:
#         train_set = df[df['year'].isin(train_years)]
#         train_set_x = train_set.drop([Y_COLUMN, 'date', 'year'], axis=1)
#
#         scalar = StandardScaler()
#         scalar.fit(train_set_x)
#
#         scaled_train_x = [scalar.transform(year) for year in train_x]
#         scaled_test_x = [scalar.transform(year) for year in test_x]
#
#         return np.array(scaled_train_x), np.array(train_y), np.array(scaled_test_x), np.array(test_y)
#
#     return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
