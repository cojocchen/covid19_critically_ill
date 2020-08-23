#coding:utf-8

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
from sksurv.metrics import concordance_index_censored
from scipy.stats import fisher_exact, chisquare, mannwhitneyu

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

FEATURE_LIST = [
                'Number.of.comorbidities',
                'Lactate.dehydrogenase',
                'Age',
                'NLR',
                'Creatine.kinase',
                'Direct.bilirubin',
                'Malignancy',
                'X.ray.abnormality',
                'COPD',
                'Dyspnea',]
SURVIVAL_THRESHOLD = (0.02750352, 0.3114235)

class Analysiser():
    def __init__(self,
            fname_pred,
            fname_truth,
            varname_pred='survival_30days',
            varname_truth='critically_ill',
            varname_time='critically_ill_time',
            ):
        self.fname_pred = fname_pred
        self.data = pd.read_csv(self.fname_pred)

        self.fname_truth = fname_truth
        tmp_df = pd.read_csv(self.fname_truth)
        self.data[varname_truth] = tmp_df[varname_truth]
        self.data[varname_time] = tmp_df[varname_time]

        self.var_pred = varname_pred
        self.var_event = varname_truth
        self.var_time = varname_time

        self.n_bootstraps=1000
        self.random_seed=42

        # will only analysis masked rows
        self.data_mask = np.ones(len(self.data))>0

    def check_time(self, default_time=0, time_delta=0.01):
        if np.min(self.data[self.var_time].values) <= 0:
            print('DATA WARNING: time cannot be smaller or equal to 0. Set to %f for analysis.'%time_delta)
            tmp = self.data[self.var_time].values
            tmp[tmp<0] = 0
            self.data[self.var_time] = tmp+time_delta

    def plot_rocs(self, var_pred_list=None, verbose=False):
        if var_pred_list == None:
            var_list = [(self.var_pred, self.var_event)]
        else:
            var_list = [(x, self.var_event) for x in var_pred_list]
        fpr = dict()
        tpr = dict()
        plt.figure()
        colors = ['aqua', 'darkorange', 'cornflowerblue']
        for i, var in enumerate(var_list):
            var_pred, var_event = var
            fpr[var_pred], tpr[var_pred], _ = \
                    roc_curve(self.data[var_event][self.data_mask].values,
                            self.data[var_pred][self.data_mask].values)
            plt.plot(fpr[var_pred], tpr[var_pred], label=var_pred,
                    color=colors[i])
        plt.legend(loc="lower right")
        plt.show()

    def analysis_auc(self, verbose=False):
        y_pred = self.data[self.var_pred][self.data_mask].values
        y_true = self.data[self.var_event][self.data_mask].values
        if len(y_pred) <= 1:
            return 0,0,0
        roc_value_raw = roc_auc_score(y_true, y_pred)

        bootstrapped_scores = []
        rng = np.random.RandomState(self.random_seed)
        for i in range(self.n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            if verbose:
                print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("AUC value: %.3f (0.95 CI, %.3f-%.3f)"%(roc_value_raw, confidence_lower, confidence_upper))
        return(roc_value_raw, confidence_lower, confidence_upper)

    def analysis_survival_recall(self, cut_values=SURVIVAL_THRESHOLD,
            verbose=False):
        res = []
        for cut in cut_values:
            thr = 1-cut
            y_pred = self.data[self.var_pred][self.data_mask].values<thr
            y_true = self.data[self.var_event][self.data_mask].values
            if len(y_pred) <= 1:
                res += [0,0,0]
                continue
            recall_value_raw = recall_score(y_true, y_pred)

            bootstrapped_scores = []
            rng = np.random.RandomState(self.random_seed)
            for i in range(self.n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(y_pred), len(y_pred))
                if len(np.unique(y_true[indices])) < 2:
                    # We need at least one positive and one negative sample for Recall
                    # to be defined: reject the sample
                    continue
                score = recall_score(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)
                if verbose:
                    print("Bootstrap #{} Recall: {:0.3f}".format(i + 1, score))

            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()
            confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
            print("Recall value (cut-off %f): %.3f (0.95 CI, %.3f-%.3f)"%(thr, recall_value_raw, confidence_lower, confidence_upper))
            res += [recall_value_raw, confidence_lower, confidence_upper]
        return res

    def analysis_c_index(self, verbose=False):
        y_pred = self.data[self.var_pred][self.data_mask].values
        y_true = self.data[self.var_event][self.data_mask].values.astype(bool)
        y_time = self.data[self.var_time][self.data_mask].values

        if len(y_pred) <= 1:
            return 0,0,0

        cidx_value_raw = concordance_index_censored(y_true,y_time,y_pred)[0]

        bootstrapped_scores = []
        rng = np.random.RandomState(self.random_seed)
        for i in range(self.n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = concordance_index_censored(y_true[indices],y_time[indices],y_pred[indices])[0]
            bootstrapped_scores.append(score)
            if verbose:
                print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("C-Index value: %.3f (0.95 CI, %.3f-%.3f)"%(cidx_value_raw, confidence_lower, confidence_upper))
        return (cidx_value_raw, confidence_lower, confidence_upper)

def evaluate_prediction(
        fname_pred,
        fname_truth,
        fname_output
        ):
    '''
    Evaluate prediction results.
    Arguments:
        fname_pred,
        fname_truth,
        basename_output
    Returns:
        results will be saved in fname_output
    '''
    print('Evaluate survival rate prediction')
    # put prediction file name here:
    ana = Analysiser(fname_pred, fname_truth, fname_output)
    result = pd.DataFrame()
    # your prediction should be in a column, fill the name of the column below to analysis
    for var in ['survival_30days']:
        res = {'prediction':var}
        ana.var_pred=var
        print('\n--Analyzing %s'%var)
        # count positives
        res['datasize'] = np.sum(ana.data_mask)
        res['positive_count'] = np.sum(ana.data[ana.var_event][ana.data_mask].values)
        if var == 'survival_30days':
            tmp = ana.analysis_survival_recall()
            for thr_idx in range(len(res)//3):
                res['recall_cut%d'%thr_idx] = tmp[thr_idx*3]
                res['recall_cut%d_lower'%thr_idx] = tmp[thr_idx*3+1]
                res['recall_cut%d_upper'%thr_idx] = tmp[thr_idx*3+2]
        tmp = ana.analysis_auc()
        res['auc'] = tmp[0]
        res['auc_lower'] = tmp[1]
        res['auc_upper'] = tmp[2]
        tmp = ana.analysis_c_index()
        res['c-index'] = tmp[0]
        res['c-index_lower'] = tmp[1]
        res['c-index_upper'] = tmp[2]
        result = result.append(res, ignore_index=True)
    result.to_csv(fname_output,sep=',',index=False,encoding='utf-8-sig')

def print_help():
    print('Usage: python evaluation.py prediction.csv truth.csv')

if __name__=='__main__':
    if len(sys.argv) < 3:
        print_help()
    else:
        evaluate_prediction(sys.argv[1], sys.argv[2], sys.argv[1]+'_evaluation.csv')
