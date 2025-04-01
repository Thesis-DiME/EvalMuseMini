import json
import csv
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import torch


def load_csv_as_dict_list(file_path):
    dict_list = []
    with open(file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict_list.append(row)
    return dict_list


def load_json_as_dict_list(file_path):
    with open(file_path, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def load_data(file_path, file_type):
    if file_type == "csv":
        return load_csv_as_dict_list(file_path)
    elif file_type == "json":
        return load_json_as_dict_list(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def compute_metrics(y_pred, y):
    """
    compute metrics btw predictions & labels
    """
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method="asymptotic")[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), np.std(y_pred)]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)

    # compute PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    return SRCC, KRCC, PLCC, RMSE


def binarize(lst):
    return [1 if x >= 0.5 else 0 for x in lst]


def calculate_accuracy(true_list, pred_list):
    true_bin = binarize(true_list)
    pred_bin = binarize(pred_list)

    correct = sum([1 for t, p in zip(true_bin, pred_bin) if t == p])
    loss = torch.mean(torch.abs(torch.tensor(true_list) - torch.tensor(pred_list)))

    accuracy = correct / len(true_list)
    return accuracy, loss


def get_index(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i : i + len_list1] == list1:
            return i
    return 0


def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]
