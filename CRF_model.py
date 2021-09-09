import pycrfsuite
import numpy as np
from preprocssing import prepare_grouped_data

X_COLUMNS = ['hour', 'month','day', 'day_literal', 'spd', 'vsb','temp', 'dewp', 'slp', 'sd', 'hday']
Y_COLUMN = 'loadrank'


def hour_features(hours, i):
    current_hour = hours[i]
    features = [col + '=' + str(val) for col, val in zip(X_COLUMNS, current_hour)]
    features.append('bias')
    if i > 0:
        prev_hour = hours[i-1]
        prev_features = ['prev_'+col + '=' + str(val) for col, val in zip(X_COLUMNS, prev_hour)]
        features.extend(prev_features)
    else:
        features.append('BOS')
    return features


def date_features(day):
    model = [hour_features(day, i) for i in range(len(day))]
    return model


def train_model(X_train, y_train, path='crf_model.crfsuite'):
    X_train = [date_features(x) for x in X_train]
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


def evaluate_model(X_test, y_test, path='crf_model.crfsuite'):
    X_test = [date_features(x) for x in X_test]
    y_test = [[str(label) for label in y] for y in y_test]
    tagger = pycrfsuite.Tagger()
    tagger.open(path)
    accuracy = 0
    for date, labels in zip(X_test, y_test):
        predicted = np.array(tagger.tag(date))
        corrected = np.array(labels)
        accuracy += np.sum(predicted == corrected) / (24 * len(X_test))
    print(f'Accuracy for crf basic model: {accuracy}')


if __name__ == '__main__':
    # main
    model_location = 'crf_model.crfsuite'
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=False)
    train_model(X_train, y_train, path=model_location)
    evaluate_model(X_test, y_test, path=model_location)

