import pycrfsuite
import numpy as np
from preprocssing import prepare_grouped_data
from preprocssing import *
X_COLUMNS = ['date','hour', 'month','day', 'day_literal', 'spd', 'vsb','temp', 'dewp', 'slp', 'sd', 'hday']
Y_COLUMN = 'loadrank'


def hour_features(hours, i, last):
    current_hour = hours[i]
    last_week_hour = last[i]
    features = [col + '=' + str(val) for col, val in zip(X_COLUMNS, current_hour)]
    features.append('bias')
    features_last = ['last_' + col + '=' + str(val) for col, val in zip(X_COLUMNS, last_week_hour)]
    features.extend(features_last)
    if i > 0:
        prev_hour = hours[i-1]
        prev_features = ['prev_'+col + '=' + str(val) for col, val in zip(X_COLUMNS, prev_hour)]
        features.extend(prev_features)
    else:
        features.append('BOS')
    return features


def date_features(day, dict2):
    day1 = int(day[0][0])
    model = [hour_features(day, i,dict2[str(day1)]) for i in range(len(day))]
    return model


def train_model(dict2,X_train, y_train, path='crf_model_adv.crfsuite'):
    X_train = [date_features(x,dict2) for x in X_train]
    y_train = [[str(label) for label in y] for y in y_train]
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 150,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(path)


def evaluate_model(dict2,X_test, y_test, path='crf_model_adv.crfsuite'):
    X_test = [date_features(x,dict2) for x in X_test]
    y_test = [[str(label) for label in y] for y in y_test]
    tagger = pycrfsuite.Tagger()
    tagger.open(path)
    accuracy = 0
    for date, labels in zip(X_test, y_test):
        predicted = np.array(tagger.tag(date))
        corrected = np.array(labels)
        accuracy += np.sum(predicted == corrected) / (24 * len(X_test))
    print(f'Accuracy for crf advance model: {accuracy}')


def pre_dict_last_week():
    dict1 = prepare()
    df = prepare_categorized_dataset()
    del df['year']
    del df['pickups']
    del df['loadrank']
    dict2 = {}
    dates_in_data = np.unique(df['date'])
    for i in dates_in_data:
        date = int(dict1[str(i)])
        hours = list(df[df['date'] == date]['hour'])
        temp = df[df['date'] == date]
        list1 = []
        for h in hours:
            list1 += [list(temp[temp['hour'] == h].iloc[0])]
        dict2[str(i)] = list1
    return dict2

def pre_dict_last_week_over():
    dict1 = prepare()
    df = pd.read_csv("over_sampling.csv", index_col=False)
    df1 = prepare_categorized_dataset()
    del df['year']
    del df['pickups']
    del df['loadrank']
    dict2 = {}
    dates_in_data = np.unique(df['date'])
    for i in dates_in_data:
        date = int(dict1[str(i)])
        hours = list(df1[df1['date'] == date]['hour'])
        temp = df1[df1['date'] == date]
        list1 = []
        for h in hours:
            list1 += [list(temp[temp['hour'] == h].iloc[0])]
        dict2[str(i)] = list1
    #print(dict2)
    return dict2

def pre_dict_last_week_under():
    dict1 = prepare()
    df = pd.read_csv("under_sampling.csv", index_col=False)
    df1 = prepare_categorized_dataset()
    del df['year']
    del df['pickups']
    del df['loadrank']
    dict2 = {}
    dates_in_data = np.unique(df['date'])
    for i in dates_in_data:
        date = int(dict1[str(i)])
        hours = list(df1[df1['date'] == date]['hour'])
        temp = df1[df1['date'] == date]
        list1 = []
        for h in hours:
            list1 += [list(temp[temp['hour'] == h].iloc[0])]
        dict2[str(i)] = list1
    return dict2

if __name__ == '__main__':
    # main
    model_location = 'crf_model_adv.crfsuite'
    dict2 = pre_dict_last_week()
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=False)
    train_model(dict2,X_train, y_train, path=model_location)
    evaluate_model(dict2,X_test, y_test, path=model_location)
    model_location = 'crf_model_adv.crfsuite'
    dict2 = pre_dict_last_week_under()
    print('under')
    X_train, y_train, X_test, y_test = prepare_grouped_data_under(scale=False)
    train_model(dict2,X_train, y_train, path=model_location)
    evaluate_model(dict2,X_test, y_test, path=model_location)
    model_location = 'crf_model_adv.crfsuite'
    dict2 = pre_dict_last_week_over()
    print('over')
    X_train, y_train, X_test, y_test = prepare_grouped_data_over(scale=False)
    train_model(dict2,X_train, y_train, path=model_location)
    evaluate_model(dict2,X_test, y_test, path=model_location)

