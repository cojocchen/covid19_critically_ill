library(mice)
library(dplyr)

args=commandArgs(T)

#workingDir = ""
#setwd(workingDir); 
refDir = "models/imputation.csv"
method ='pmm'

#load data
train_data <- read.table(refDir,header=F, check.names=F, sep=",") #1590 obs of 65 variables
train_data_df <- data.frame(train_data, row.names=train_data[,1]) #set first column (patient_ID) as index
train_data_df <- train_data_df[,-1]
count_train <- nrow(train_data_df)

#external validation imputation
#should keep same feature sequence in external validation csv
ex_test_data <- read.table(args[1],header = T, check.names=F, sep=",")
#ex_test_data <- read.table("imputation_test_data_noheader.csv",header = F, check.names=F, sep=",")
ex_test_data_df <- data.frame(ex_test_data, row.names = ex_test_data[,1])
ex_test_data_df <- ex_test_data_df[,-1]
count_ex_test <- nrow(ex_test_data_df)

ex_test_imputed_data = ex_test_data_df[1,]


#rename train data colnames
names(ex_test_data_df)
names(train_data_df) <- c("Malignancy","Age","X.ray.abnormality","Dyspnea","COPD","Lactate.dehydrogenase","Direct.bilirubin","Creatine.kinase","Number.of.comorbidities","NLR")
names(train_data_df)

#if only use imputaed training data instead of re-imputing training data
completeTrainData <- train_data_df
sapply(completeTrainData, function(x) sum(is.na(x)))# check whether all na are filled

for (i in seq(1, count_ex_test, by=1)) {
  print(i)
  ex_total_df <- bind_rows (completeTrainData, ex_test_data_df[i,])
  ex_imputedData <- mice(ex_total_df, m=1,maxit=20, method=method,seed=500) #List of 21
  ex_completeData<- complete(ex_imputedData)
  ex_test_imputed_data <- bind_rows(ex_test_imputed_data, ex_completeData[count_train+1,]) #the obs of train + 1 the line of test data
}
ex_test_imputed_data <- ex_test_imputed_data[-1,]
sapply(ex_test_imputed_data, function(x) sum(is.na(x)))# check whether all na are filled

write.csv(ex_test_imputed_data, paste(args[2],sep="") )
