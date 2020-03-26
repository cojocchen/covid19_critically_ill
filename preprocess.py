#coding:utf-8
import os
import sys
import numpy as np
import pandas as pd

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

def imputate(
        fname_input,
        fname_output):
    '''
    Normalize input values
    Arguments:
        fname_input: string, input csv file name
            file should contain following columns (raw value, NOT normalized value):
                'patient_ID'
                'Number.of.comorbidities'
                'Lactate.dehydrogenase'
                'Age'
                'NLR'
                'Creatine.kinase'
                'Direct.bilirubin'
                'Malignancy'
                'X.ray.abnormality'
                'COPD'
                'Dyspnea'
        fname_output: string, output csv file name
    Return:
        results will be saved in fname_output
    '''
    os.system('Rscript --vanilla ./data_imputation.R %s %s'%(fname_input,fname_output))

def normalize(raw):
    '''
    Normalize input values
    Arguments:
        raw: a dictionary containing raw values/arrays of target features.
            Will look for:
            raw['Number.of.comorbidities']
            raw['Lactate.dehydrogenase']
            raw['Age']
            raw['NLR']
            raw['Creatine.kinase']
            raw['Direct.bilirubin']
            raw['Malignancy']
            raw['X.ray.abnormality']
            raw['COPD']
            raw['Dyspnea']
    Return:
        res: a dictionary containing normalized values/arrays of target features.
    '''
    # the type of data is a dictionary, keys are the features, values are values
    # TO-DO: use external json file to store and load mean/std values
    mean = {'Number.of.comorbidities':0.366352,
            'Lactate.dehydrogenase':259.740330,
            'Age':48.879804,
            'NLR':5.374749,
            'Creatine.kinase':98.198436,
            'Direct.bilirubin':3.762461,
            }
    std = {'Number.of.comorbidities':0.766475,
           'Lactate.dehydrogenase':77.955030,
           'Age':12.861063,
           'NLR':2.445621,
           'Creatine.kinase':49.690099,
           'Direct.bilirubin':1.389199
           }
    res = {}
    res['Malignancy'] = raw['Malignancy']
    res['X.ray.abnormality'] = raw['X.ray.abnormality']
    res['COPD'] = raw['COPD']
    res['Dyspnea'] = raw['Dyspnea']
    res['Number.of.comorbidities'] = (raw['Number.of.comorbidities'] - mean['Number.of.comorbidities'])/std['Number.of.comorbidities']
    res['Lactate.dehydrogenase'] = (raw['Lactate.dehydrogenase'] - mean['Lactate.dehydrogenase'])/std['Lactate.dehydrogenase']
    res['Age'] = (raw['Age'] - mean['Age'])/std['Age']
    res['NLR'] = (raw['NLR'] - mean['NLR'])/std['NLR']
    res['Creatine.kinase'] = (raw['Creatine.kinase'] - mean['Creatine.kinase'])/std['Creatine.kinase']
    res['Direct.bilirubin'] = (raw['Direct.bilirubin'] - mean['Direct.bilirubin'])/std['Direct.bilirubin']
    return res

def count_missing_values_table(df):
    # count total missing values
    mis_val = df[FEATURE_LIST].isnull().sum().sum()
    return mis_val

def preprocess(
        fname_input,
        fname_output):
    '''
    Preprocess file in the following order:
        imputation (if contains missing value of features)
        normalization (only for none-binary value)
    Arguments:
        fname_input: string, input csv file name
            file should contain following columns (raw value, NOT normalized value):
                'patient_ID'
                'Number.of.comorbidities'
                'Lactate.dehydrogenase'
                'Age'
                'NLR'
                'Creatine.kinase'
                'Direct.bilirubin'
                'Malignancy'
                'X.ray.abnormality'
                'COPD'
                'Dyspnea'
        fname_output: string, output csv file name
    Return:
        results will be saved in fname_output
    '''
    # load file and check
    print('Preprocess data')
    df = pd.read_csv(fname_input)
    for var in FEATURE_LIST:
        if var not in df:
            print('DATA ERROR: need variable:',var)
            exit(0)
    print('--date format check: pass')
    # imputate missing value if necessary
    if count_missing_values_table(df) > 0:
        print('--date imputation: required')
        imputate(fname_input, fname_output)
        df = pd.read_csv(fname_output)
        print('--date imputation: done')
    # normalize features
    dict_raw = {}
    for var in FEATURE_LIST:
        dict_raw[var] = df[var]
    dict_norm = normalize(dict_raw)
    for var in FEATURE_LIST:
        df[var] = dict_norm[var]
    print('--date normalization: done')
    df.to_csv(fname_output,sep=',',index=False,encoding='utf-8-sig')
    print('--file saved to:',fname_output)

def print_help():
    print('Usage: python preprocess.py input.csv')

def test_script():
    preprocess(
            './data/test.csv',
            './data/test.csv_normalized.csv'
            )
    ## raw data of patient # 1 for test
    #test_data={'Malignancy':0,
    #        'X.ray.abnormality':0,
    #        'COPD':0,
    #        'Dyspnea':0,
    #        'Number.of.comorbidities':0,
    #        'Lactate.dehydrogenase':433,
    #        'Age':47,
    #        'NLR':10.32,
    #        'Creatine.kinase':429,
    #        'Direct.bilirubin':5.3}
    ##after normalization
    #normalized_data = normalize(test_data)
    #print(normalized_data)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print_help()
    else:
        preprocess(sys.argv[1], sys.argv[1]+'_processed.csv')
