import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import heapq
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def sign(yi):
    if int(yi) == 1:
        return 1
    else:
        return -1


def knn(X_test, X_train, Y_train, k=5):

    coeff = np.zeros((X_train.shape[1], 1))

    # for each train vector
    for ix in range(coeff.shape[0]):

        if not np.isnan(float(Y_train[ix])):

            v1 = np.asarray(X_train[:, ix], dtype='float')
            coeff[ix] = sp.stats.pearsonr(v1, np.asarray(X_test, dtype='float'))[0]

        else:
            coeff[ix] = -100.0

    return heapq.nlargest(k, range(len(coeff)), coeff.take)


def get_prediction(output_knn, Y_train):
    neighbours = []
    for ik in output_knn:
        neighbours.append(float(Y_train[ik]))
    num_1 = np.sum(neighbours)
    avg = np.mean(neighbours)

    return avg

################ Data preparation ##############

# Load the data into variable 'd'
d = pd.read_csv('data.csv', header=None)

# get the data values as a numpy matrix and store in the variable 'data'
data = d.values

# extract the data for drugs in another variable
drugs = data[1:6, :]

# data for genes and cell lines
cell_line_names = data[0, 1:]
cell_lines = data[6:, 1:]


cell_data = data[6:, 1:]
#################################################

for jx in range(drugs.shape[0]):
# for jx in range(1):
    drug_data = drugs[jx, 1:]
    # print drug_data.shape
    drug_name = drugs[jx, 0]

    predictions_k3 = []
    predictions_k5 = []
    predictions_k7 = []
    real_k3 = []
    real_k5 = []
    real_k7 = []

    for ix in range(cell_data.shape[1]):
        t1 = cell_data[:, :ix]
        t2 = cell_data[:, ix+1:]
        x_train = np.hstack((t1, t2))
        x_test = cell_data[:, ix]

        t1 = drug_data[:ix]
        t2 = drug_data[ix+1:]
        y_train = np.hstack((t1, t2))
        y_test = drug_data[ix]

        # print 'X, y train:', x_train.shape, y_train.shape
        # print 'X, y test:', x_test.shape, y_test

        # skip if y_test is nan
        if np.isnan(float(y_test)):
            continue
        preds_index_k3 = knn(x_test, x_train, y_train, k=3)
        preds_index_k5 = knn(x_test, x_train, y_train, k=5)
        preds_index_k7 = knn(x_test, x_train, y_train, k=7)
        # print preds_index
        pred_k3 = get_prediction(preds_index_k3, y_train)
        pred_k5 = get_prediction(preds_index_k5, y_train)
        pred_k7 = get_prediction(preds_index_k7, y_train)
        # print pred
        predictions_k3.append(pred_k3)
        predictions_k5.append(pred_k5)
        predictions_k7.append(pred_k7)
        real_k3.append(int(y_test))
        real_k5.append(int(y_test))
        real_k7.append(int(y_test))
    predictions_k3 = np.asarray(predictions_k3)
    predictions_k5 = np.asarray(predictions_k5)
    predictions_k7 = np.asarray(predictions_k7)
    real_k3 = np.asarray(real_k3)
    real_k5 = np.asarray(real_k3)
    real_k7 = np.asarray(real_k3)
    # print real.shape, predictions.shape
    # print real
    # print predictions
    f_k3, t_k3, _3 = roc_curve(real_k3, predictions_k3)
    f_k5, t_k5, _5 = roc_curve(real_k5, predictions_k5)
    f_k7, t_k7, _7 = roc_curve(real_k7, predictions_k7)
    # print f
    # print t
    auc_score_k3 = auc(f_k3, t_k3)
    auc_score_k5 = auc(f_k5, t_k5)
    auc_score_k7 = auc(f_k7, t_k7)

    plt.figure(jx)
    plt.plot(f_k3, t_k3, 'r', marker='o', label='ROC curve (area = %0.2f)' % auc_score_k3)
    plt.plot(f_k5, t_k5, 'g', marker='o', label='ROC curve (area = %0.2f)' % auc_score_k5)
    plt.plot(f_k7, t_k7, 'b', marker='o', label='ROC curve (area = %0.2f)' % auc_score_k7)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


