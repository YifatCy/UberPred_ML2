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
    num_0 = data[data[Y_COLUMN] == 0]
    #print('num_0',num_0)
    num_1 = data[data[Y_COLUMN] == 1]
    num_2 = data[data[Y_COLUMN] == 2]
    num_3 = data[data[Y_COLUMN] == 3]
    y_2 = np.unique(num_2['date'])
    y_3 = np.unique(num_3['date'])
    count=0
    new = pd.DataFrame(columns=data.columns)
    for i in y_2:
        p = np.random.rand()
        if p > 0.5:
            count += 1
            temp = data[data['date']==i]
            temp = temp.replace(2015,str(2016))
            temp['date'] = temp.apply(lambda row: str(row['year'])+str(row['month'])+str(row['day']),axis=1)
            #print(temp)
            new = pd.concat([new,temp])

    for i in y_3:
        p = np.random.rand()
        if p > 0.5:
            count += 1
            temp = data[data['date']==i]
            temp = temp.replace(2015,str(2016))
            temp['date'] = temp.apply(lambda row: str(row['year'])+str(row['month'])+str(row['day']),axis=1)
            #print(temp)
            new = pd.concat([new,temp])
    new_data=pd.concat([new,data])
    print('cnt ', count)
    print('over sampling ',len(new_data))
    #new_data = shuffle(new_data)
    for i in ['date','hour', 'month','day', 'day_literal', 'year','spd', 'temp', 'dewp', 'hday','pickups','loadrank']:
        new_data[i] = new_data[i].astype(int)
    new_data.to_csv('over_sampling.csv',index=False)
    print('over_sampling',new_data)
    df = new_data.groupby('loadrank')['pickups'].count().reset_index(name='sum_uber_pickups')
    df = df.set_index('loadrank')
    df.plot.bar(rot=0, title=f"Hist of Hours per Category  [0, 1, 2, 3]\n Over", alpha=0.7, color='salmon')
    plt.ylabel('Sum Hours')
    plt.xlabel('Category')
    #plt.savefig('figures/hist_per_each_category.png')
    plt.show()
    return new_data

def hist_of_loads():
    df = pd.read_csv("datasets/uber_hour_categorized_by_borough1.csv")
    df = df[['borough','loadrank']]


    for b in ['Bronx','Brooklyn','EWR','Manhattan','Queens','Staten Island']:
        df1 = df[df['borough'] == b ]
        df1 = df1['loadrank']
        plt.hist(df1,label=b,alpha=0.5)
    plt.xticks((0,1,2,3))
    plt.legend(loc='upper right')
    plt.show()
    #df.plot.hist(x='loadrank',)
    #plt.hist(df)
    #df.plot.hist(columns='borough')
    #df.plot.bar(rot=0, title=f"Hist of Hours per Category  [0, 1, 2, 3]\n", alpha=0.7, color='salmon')
    #plt.ylabel('Sum Hours')
    #plt.xlabel('Category')
    #plt.savefig('figures/hist_per_each_category.png')
    #df.plot.bar()
    #plt.show()
hist_of_loads()
def under_sampling():
    ## under sampling
    data = prepare_categorized_dataset()
    num_0 = data[data[Y_COLUMN] == 0]
    #print('num_0',num_0)
    num_1 = data[data[Y_COLUMN] == 1]
    num_not = pd.concat([num_1,num_0])
    #print(num_not)
    num_not_date = list(np.unique(num_not[['date']]))
    #print(len(num_not_date))
    num_2 = data[data[Y_COLUMN] == 2]
    num_3 = data[data[Y_COLUMN] == 3]
    y_2 = np.unique(num_2['date'])
    y_3 = np.unique(num_3['date'])
    y_not = []
    for i in num_not_date:
        if i in y_2:
            continue
        if i in y_3:
            continue
        else:
            y_not += [i]
    #print(len(y_not))
    #print(y_not)
    new_data = pd.DataFrame(columns=data.columns)
    for i in y_2:
        temp = data[data['date'] == i]
        new_data = pd.concat([new_data,temp])
    for i in y_3:
        temp = data[data['date'] == i]
        new_data = pd.concat([new_data,temp])
    cnt = 0
    for i in y_not:
        p = np.random.rand()
        if p >= 0.5:
            cnt += 1
            temp = data[data['date'] == i]
            new_data = pd.concat([new_data,temp])
    #new_data = shuffle(new_data)
    for i in ['date','hour', 'month','day', 'day_literal', 'year','spd', 'temp', 'dewp', 'hday','pickups','loadrank']:
        new_data[i] = new_data[i].astype(int)
    new_data.to_csv('under_sampling.csv',index=False)
    df = new_data.groupby('loadrank')['pickups'].count().reset_index(name='sum_uber_pickups')
    df = df.set_index('loadrank')
    df.plot.bar(rot=0, title=f"Hist of Hours per Category  [0, 1, 2, 3]\n Under", alpha=0.7, color='salmon')
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
    quantile = np.quantile(max_range, q=[1/7, 2/7, 3/7, 4/7, 5/7, 6/7])
    df['loadrank'] = df["pickups"].apply(
        lambda x: 0 if x <= quantile[0] else
        (1 if x <= quantile[1] else (2 if x <= quantile[2] else (3 if x<= quantile[3] else (4 if x <= quantile[4] else (5 if x <= quantile[5]
            else 6 ))))))
    # print(df_agg)
    df.to_csv("datasets/uber_hour_categorized_by_borough.csv",index=False)
    return df

def get_boroughs_dict():
    return {"Bronx": 0, "Brooklyn": 1, "EWR": 2, "Manhattan": 3, "Queens": 4, "Staten Island": 5, "NA": 6}

def get_boroughs_dict_reversed():
    return {0: "Bronx", 1:"Brooklyn", 2: "EWR", 3: "Manhattan", 4: "Queens", 5: "Staten Island", 6: "NA"}

def get_x_any_y_creative(df, dates, y_column):
    x, y = [], []
    day_bor_df = pd.DataFrame()
    dict = get_boroughs_dict()
    for bor in dict.values():
        for date in dates:
            day_df = df[df['date'] == date]
            day_bor_df = day_df[day_df['borough'] == bor]
            if not day_bor_df.empty:
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
    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.drop([Y_COLUMN, "pickups", 'date','year'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y
