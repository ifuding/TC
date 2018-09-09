import sklearn
import pandas as pd
import numpy as np
from time import gmtime, strftime
from feature_engineer import timer

nrs = np.random.RandomState(0)

def lcc_sample(labels, preds, input_data, C = 1):
    """
    Param:
    labels shape: (n_sample,)
    preds shape: (n_sample,)
    input_data shape: (n_sample, feature_dim)
    C: times based on accepte_rate
    return:
    data after sampling
    """
    accept_rate = np.abs(labels - preds) * C
    bernoulli_z = nrs.binomial(1, np.clip(accept_rate, 0, 1))
    select_ind = [i for i in range(bernoulli_z.shape[0]) if bernoulli_z[i] == 1]
    sample_data = input_data[select_ind, :]
    sample_labels = labels[select_ind]
    weight = np.ones(len(labels))
    adjust_weight_ind = [i for i in range(len(accept_rate)) if accept_rate[i] > 1]
    weight[adjust_weight_ind] = accept_rate[adjust_weight_ind]
    weight = weight[select_ind]
    print('-----LCC Sampling Before All: {} Pos: {} Neg: {}'.format(len(labels), np.sum(labels == 1), np.sum(labels == 0)))
    print('-----LCC Sampling After All: {} Pos: {} Neg: {}'.format(len(sample_labels), np.sum(sample_labels == 1), np.sum(sample_labels == 0)))
    print('-----LCC Sampling Rate: {}'.format(float(len(sample_labels)) / float(len(labels))))
    return sample_data, sample_labels, weight


def neg_sample(input_data, labels, C = 1):
    """
    Param:
    labels shape: (n_sample,)
    preds shape: (n_sample,)
    input_data shape: (n_sample, feature_dim)
    C: neg_number = C * pos_number   
    return:
    data after sampling
    """
    with timer("Negative sampling"):
        print('Negative sampling...')
        pos_ind = np.where(labels == 1)[0]
        neg_ind = np.where(labels == 0)[0]
        accept_rate = float(C * len(pos_ind)) / float(len(neg_ind))
        neg_select_ind = nrs.choice(neg_ind, len(pos_ind) * C, replace = True)
        select_ind = np.append(pos_ind, neg_select_ind)
        nrs.shuffle(select_ind)
        sample_data = input_data[select_ind, :]
        sample_labels = labels[select_ind]
        sample_neg_ind = np.where(sample_labels == 0)[0]
        weight = np.ones(len(sample_labels))
        weight[sample_neg_ind] = 1.0 / accept_rate
        print('-----Neg Sampling Before All: {} Pos: {} Neg: {}'.format(len(labels), np.sum(labels == 1), np.sum(labels == 0)))
        print('-----Neg Sampling After All: {} Pos: {} Neg: {}'.format(len(sample_labels), np.sum(sample_labels == 1), np.sum(sample_labels == 0)))
        print('-----Neg Sampling Rate: {}'.format(float(len(sample_labels)) / float(len(labels))))
    return sample_data, sample_labels, weight
