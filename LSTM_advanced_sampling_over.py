import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from preprocessing import *
from preprocessing import train_test_split
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class LSTM_Tagger(nn.Module):
    def __init__(self, vector_embedding_dim, hidden_dimension, classes_num):
        super(LSTM_Tagger, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device =torch.device("cpu")
        self.lstm = nn.LSTM(input_size=vector_embedding_dim, hidden_size=hidden_dimension,
                            num_layers=2, bidirectional=True, batch_first=False)
        self.hidden_to_count = nn.Linear(hidden_dimension * 2, classes_num)

    def forward(self, hours_array, hidden_layer=False):
        tens_hours = torch.from_numpy(hours_array).float().to(self.device)
        lstm_out, _ = self.lstm(tens_hours.view(tens_hours.shape[0], 1, -1))
        if hidden_layer:
            return lstm_out
        weights_class = self.hidden_to_count(lstm_out.view(tens_hours.shape[0], -1))
        count_scores = F.log_softmax(weights_class, dim=1)
        return count_scores


def train(verbose=True, hidden_dimension=100, x_train=None, y_train=None, x_test=None, y_test=None, epochs=40):
    if x_train is None:
        x_train, y_train, x_test, y_test = prepare_grouped_data(scale=True)
    epochs = epochs
    hidden_dimension = hidden_dimension
    vector_embed_dim = x_train[0].shape[1]
    accum_grad_steps = 50
    count_type = 4
    model = LSTM_Tagger(vector_embed_dim, hidden_dimension, count_type)
    with_cuda = torch.cuda.is_available()
    torch_device = torch.device("cuda:0" if with_cuda else "cpu")
    if with_cuda:
        model.cuda()
    loss_function = nn.NLLLoss()
    opt = optim.Adam(model.parameters(), lr=0.01)
    if verbose:
        print("start-train")
    accuracy_list = []
    loss_l = []
    epochs = epochs
    best_accuracy = 0
    for epoch in range(epochs):
        accuracy = 0
        loss_printable = 0
        indx = 0
        for day_index in np.random.permutation(len(x_train)):
            indx += 1
            hours_array = x_train[day_index]
            counts_tensor = torch.from_numpy(y_train[day_index]).to(torch_device)
            counts_scores = model(hours_array)
            loss = loss_function(counts_scores, counts_tensor)
            loss /= accum_grad_steps
            loss.backward()
            if indx % accum_grad_steps == 0:
                opt.step()
                model.zero_grad()
            loss_printable += loss.item()
            _, indices = torch.max(counts_scores, 1)
            accuracy += np.mean(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())

        if verbose:
            loss_printable = accum_grad_steps * (loss_printable / len(x_train))
            accuracy = accuracy / len(x_train)
            accuracy_list.append(float(accuracy))
            loss_l.append(float(loss_printable))
            test_accuracy = estimation(model, torch_device, x_test, y_test)
            best_accuracy = test_accuracy if test_accuracy > best_accuracy else best_accuracy
            e_interval = indx
            print("Epoch {} completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
                  .format(epoch + 1, np.mean(loss_l[-e_interval:]),
                          np.mean(accuracy_list[-e_interval:]), test_accuracy))
    return model, best_accuracy


def estimation(model, device, x_test, y_test):
    accuracy = 0
    with torch.no_grad():
        for day_index in range(len(x_test)):
            hours_array = x_test[day_index]
            tensor_count = torch.from_numpy(y_test[day_index]).to(device)
            scores_count = model(hours_array)
            _, indx = torch.max(scores_count, 1)
            accuracy += np.sum(tensor_count.to("cpu").numpy() == indx.to("cpu").numpy())
        accuracy = accuracy / (len(x_test) * len(x_test[0]))
    return accuracy

def model_loading(model_fname):
    with open(f'dumps/{model_fname}', 'rb') as f:
        model = pickle.load(f)
    return model

def model_saving(model, model_fname):
    with open(f'dumps/{model_fname}', 'wb') as f:
        pickle.dump(model, f)

def FindMaxLength(lst):
    maxLength = max(len(x) for x in lst)
    return maxLength


def LSTM_error_per_hour(model):
    _, _, x_test, y_test = prepare_grouped_data(scale=True)
    counts = np.zeros(24)
    errors = np.zeros(24)
    for x, y in zip(x_test, y_test):
        _, pred = torch.max(model(x), 1)
        for indx in range(len(x)):
            if pred[indx] != y[indx]:
                errors[indx] += 1
            counts[indx] += 1

    rate_errors = errors / np.sum(errors)
    plt.bar(np.arange(1, 25), rate_errors)
    plt.xticks(np.arange(1, 25))
    plt.title('LSTM Error Distribution - hourly')
    plt.show()


if __name__ == '__main__':

    x_train, y_train, x_test_val, y_test_val = prepare_grouped_data(scale=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=2 / 3, random_state=57)
    print('start-validation')
    best_accuracy = 0
    best_dimension = 50
    epochs = 40
    for hidden_dimension in [50, 100, 200]:
        print('_________________')
        print(f'hidden dimension: {hidden_dimension}')
        _, accuracy = train(verbose=True, hidden_dimension=hidden_dimension,
                            x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val, epochs=epochs)
        best_accuracy, best_dimension = (accuracy, hidden_dimension) if accuracy > best_accuracy else (best_accuracy, best_dimension)
    print(f'the best accuracy: {best_accuracy}\tBest dim: {best_dimension}')
    model, accuracy = train(verbose=True, hidden_dimension=best_dimension,
                            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs)
    print(f'model test accuracy: {accuracy}')

