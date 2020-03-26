# Early Triage of Critically-ill COVID-19 Patients using Deep Learning

This package provides an implementation of the prediction and analysis of early triage of critically-ill COVID-19 patients using deep learning

## Setup

### Dependencies

Processing pipelines are implemented in python.
If there is missing value, data imputation requires R.

#### Python 3.6+
* pandas 0.23.0
* tensorflow 1.14
* sklearn 0.21.3
* sksurv 0.11
* scipy 1.4.1

#### R 3.6.2+
* mice_3.8.0
* dplyr_0.8.5

## Data

Data are all in csv format. Results will be saved in csv format as well.

### Input file format

A csv file in the following formate is needed for prediction:
patient_ID | X.ray.abnormality | Dyspnea | Number.of.comorbidities | Malignancy | COPD | Age | NLR | Lactate.dehydrogenase | Direct.bilirubin	| Creatine.kinase
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1 |	1 |	0 |	0 |	0 |	0 |	59 |	3.428571429 |	201 |	1.7 |	26
6 |	 |	0 |	0 |	0 |	0 |	65 |	6.545454545 |	255 |	3.3 |	
9 |	 |	0 |	4 |	0 |	0 |	61 |	3.888888889 |	211 |	3.2 |	113

If value is unknow, you can leave the cell empty. A imputation method will be used to fill the value. But please note that complete values will result in better prediction performance.

Explaination of each value is as following:
column | data_type | unit | explain
--- | --- | --- | ---
patient_ID | any | | ID of patients 
X.ray.abnormality | binary | 0/1 | 1 indicate x ray abornmality
Dyspnea | binary | 0/1 | 1 indicate dyspnea
Number.of.comorbidities | intiger | 0-9 | count of comorbidities
Malignancy | binary | 0/1 | 1 indicate cancer (maligant)
COPD | binary | 0/1 | 1 indicate COPD
Age | intiger | 0-150 | age of patient
NLR | float | ratio | Neutrophil-Lymphocyte Ratio (NLR)
Lactate.dehydrogenase | float | U/I | Lactate dehydrogenase test result
Direct.bilirubin	| float | μmol/l | Direct bilirubin test result
Creatine.kinase | float | U/l | Creatine kinase test result

For more  details, please refer to paper.

### Truth file format

To evaluate the result, a right censored file containing the truth of each observation is required. A csv file in the following format is needed:
patient_ID |	critically_ill |	critically_ill_time
--- | --- | ---
1 |	0 |	5.0
2 |	0 |	10.0
3 |	0 |	6.0

All values are needed and the samples need to be in the same order as input csv file (patient_ID will NOT be checked in analysis). Explaination of each value is as following:
column | data_type | unit | explain
--- | --- | --- | ---
patient_ID | any | | ID of patients 
critically_ill | 	binary | 0/1 | 1 indicate critically-ill patient, right censored
critically_ill_time | float | days | number of days from data collection to the event

Truth file is not needed if you only want to predict the survival ratio.

### Model checkpoints

All models should be stored in models fold. Please unzip the models.zip file and make sure the files follows rule:

* code_root
  * models
    * fold_0
      * model.meta 
    * fold_1
      * model.meta
    * fold_2
      * model.meta
    * fold_3
      * model.meta
    * fold_4
      * model.meta
    * imputation.csv

## Preprocess

Preprocessing code will impute missing value and normalize results. To run the code:
```
python preprocess.py input.csv
```
"input.csv" can be replaced with any csv file that meets input file format requirement. A file with name [INPUT]_processed.csv will be saved containing the processed file. Use this file as the input of next step.

## Prediction

Prediction code will predict risk scores and survival probability for each sample.
```
python preprocess.py input.csv_processed.csv
```
"input.csv" can be replaced with the name of your input file. A file with name [INPUT]_processed.csv_prediction.csv will be saved containing the prediction results. Use this file as the input of next step.

5 values are generated:
Value | Explaination
--- | ---
score_dl | risk score of deep learning model
score_final | final risk score of nomogram
survival_5days | 5 days critically ill probability
survival_10days | 10 days critically ill probability
survival_30days | 30 days critically ill probability

## Evaluation

Evaluation code will analysis predictions. To run:
```
python evaluation.py input.csv_processed.csv_prediction.csv truth.csv
```
"input.csv" can be replaced with the name of your input file. "truth.csv" can be replaced with the truth file that meet the format requirement. A file with name [INPUT]_processed.csv_prediction.csv_evaluation.csv will be saved containing the evaluation results. Use this file as the input of next step.

Following analysis will be conducted on "survival_30days" with 95% confidence inter computed as well:
 Value | Explaination
--- | ---
c-index | C-index of survival analysis
auc | ROAUC value assuming all right-censored samples are negatives
recall | recall value with different cut-offs assuming all right-censored samples are negatives

## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.
