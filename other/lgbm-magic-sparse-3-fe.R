set.seed(9002)

library(data.table)
library(caret)
library(lightgbm)
library(pROC)
library(dplyr)
library(matrixStats)

# load data
feature_names <- sprintf('var_%d', 0:199)
fake_rows <- fread('../input/fake-rows/fake_rows.csv')[[1]]
target <- fread('../input/santander-customer-transaction-prediction/train.csv', select='target')[[1]]
train <- fread('../input/santander-customer-transaction-prediction/train.csv', select=feature_names)
test <- fread('../input/santander-customer-transaction-prediction/test.csv', select=feature_names)


# sparsify features
full <- rbind(train, test[-fake_rows])
for (f in feature_names) {
  counts <- full[, .N, by=f]
  for (n in 1:3) {
    keep <- counts[N > n][[f]]    
    set(train,
        j=sub('var', paste0('sparse', n), f),
        value=ifelse(train[[f]] %in% keep, train[[f]], NA))
    set(test,
        j=sub('var', paste0('sparse', n), f),
        value=ifelse(test[[f]] %in% keep, test[[f]], NA))    
  }
}

# add overlap/unique counts


trainX <- train
testX <-  test
trainY <- target

rm(train, test, target)
gc()

fold_num <- 10
folds <- createFolds(factor(trainY), k = fold_num, list = FALSE)

# vectors to save AUC to validation fold and predictions on test 
AUC_valid_folds <- vector(mode = "numeric",  length = fold_num)
pred_test_folds <- vector(mode = "list", length = fold_num)

for (fld in 1:fold_num) {
    cat(paste0("Fold number ", fld, "...\n"))
    
    newtrain <- trainX[folds != fld, ]
    newlabel <- trainY[folds != fld]
    dtrain <- lgb.Dataset(data = as.matrix(newtrain), label = as.matrix(newlabel))
    dvalid <- lgb.Dataset(data = as.matrix(trainX[folds == fld, ]), label = as.matrix(trainY[folds == fld]))
    
    lgb_params <- list(objective = "binary", 
                       boost="gbdt",
                       metric="auc",
                       boost_from_average="false",
                       num_threads=4,
                       learning_rate = 0.005,
                       num_leaves = 11,
                       max_depth=-1, 
                       tree_learner = "serial",
                       feature_fraction = 0.05,
                       bagging_freq = 15,
                       bagging_fraction = 0.4,
                       min_data_in_leaf = 80,
                       min_sum_hessian_in_leaf = 10.0)
    
    lgb_model <- lgb.train(params = lgb_params,
                          data = dtrain,
                          nrounds = 1000000,
                          list(val = dvalid),
                          eval_freq = 10000, 
                          eval = "auc",
                          early_stopping_rounds = 3000,
                          seed = 44000
    )
    
    pred_valid <- predict(lgb_model, as.matrix(trainX[folds == fld, ]))
    roc_obj <- roc(response = trainY[folds == fld], pred_valid)
    AUC_valid_folds[fld] <- auc(roc_obj) 
    cat(paste0("Auc ", AUC_valid_folds[fld], "\n"))
    test_pred <- predict(lgb_model, as.matrix(testX))
    pred_test_folds[[fld]] <- test_pred
}
cat(paste("Average AUC:", round(mean(AUC_valid_folds),6), "\nStandard deviation AUC:", round(sd(AUC_valid_folds),6))) #incorrect AUC calculation

final_pred <- rowMeans(sapply(pred_test_folds, rank))

submission <- read.csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
submission$target <- final_pred
write.csv(submission, file = "submission.csv", row.names=F)