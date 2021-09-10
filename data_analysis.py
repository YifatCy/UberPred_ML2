from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from preprocssing import prepare_categorized_dataset
import seaborn as sns
plt.style.use('ggplot')


def hist_per_hours(df):
    df = df.groupby(['hour'])['pickups'].sum().reset_index(name='sum_uber_pickups')
    df = df[['sum_uber_pickups']]
    print(max(df['sum_uber_pickups']))
    df.plot.bar(rot=0,title=f"Uber Pickups by Hour",alpha=0.7,color='royalblue')
    plt.ylabel('Sum pickups / 1M')
    plt.xlabel('Hour')
    plt.savefig('figures/hist_per_hours.png')
    plt.show()


def hist_per_week_day(df):
    df = df.groupby(['day_literal'])['pickups'].sum().reset_index(name='sum_uber_pickups')
    df['Day'] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = df.set_index('Day')
    df = df[['sum_uber_pickups']]
    df.plot.bar(rot=0,title=f"Uber Pickups by Week Day",alpha=0.7,color='teal')
    plt.ylabel('Sum pickups / 1M')
    plt.savefig('figures/hist_per_week_day.png')
    plt.show()

def hist_per_each_category(df):
    df = df.groupby('loadrank')['pickups'].count().reset_index(name='sum_uber_pickups')
    df = df.set_index('loadrank')
    df[['sum_uber_pickups']]
    df.plot.bar(rot=0, title=f"Hist of Hours per Category [0, 1, 2, 3]", alpha=0.7,color='salmon')
    plt.ylabel('Sum Hours')
    plt.xlabel('Category')
    plt.savefig('figures/hist_per_each_category.png')
    plt.show()


def heat_map_day_hours_pickups(df):
    print(df.groupby(['day_literal', 'hour'])['pickups'].mean())
    dow_dict = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    print(df)
    corrected_dict = {v: k for k, v in dow_dict.items()}

    #df['day_literal'] = df['day_literal'].replace(corrected_dict)
    print(df)
    weekly_data = df.groupby(['day_literal', 'hour'])['pickups'].mean()
    #print(weekly_data.index[:][0])
    #weekly_data['day_literal'] = weekly_data['day_literal'].replace(corrected_dict)

    weekly_data = weekly_data.unstack(level=0)
    weekly_data = weekly_data.rename(columns=corrected_dict)
    plt.figure(figsize=(15, 10))
    sns.heatmap(weekly_data, cmap="Blues",annot_kws={"size": 25})
    #sns.set(font_scale=1.4)
    _ = plt.title('Heatmap of average pickups in time vs day grid')
    plt.savefig('figures/heatmap.png')
    plt.show()



df = prepare_categorized_dataset()
print(df)
hist_per_hours(df)
hist_per_week_day(df)
hist_per_each_category(df)
heat_map_day_hours_pickups(df)

nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
nyc.head(5)