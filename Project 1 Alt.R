# Libraries
library(caret)
library(ROSE)
library(randomForest)
library(rpart)
library(rpart.plot)
library(forecast)
library(treeClust)
library(car)
library(dplyr)
library(tidyr)
#------------------------------------------------------------------------------
# Load Data
car1 <- read.csv("/Users/reyn/Documents/SU SR Year/Fall Quarter/Data Mining/Projects/car_test_7.csv", header = TRUE)  
car2 <- read.csv("/Users/reyn/Documents/SU SR Year/Fall Quarter/Data Mining/Projects/car_train_class_7.csv", header = TRUE)

car2$price_nom <- as.factor(car2$price_nom)
car2$is_new <- as.factor(car2$is_new)
car2$has_accidents <- as.factor(car2$has_accidents)
car2$frame_damaged <- as.factor(car2$frame_damaged)

str(car2)
table(car2$price_nom)
#------------------------------------------------------------------------------
#Training/Validation Set
set.seed(666)

training_index <- sample(1:nrow(car2), 0.6 * nrow(car2))
validation_index <- setdiff(1:nrow(car2), training_index)

training_df <- car2[training_index, ]
validation_df <- car2[validation_index, ]

#------------------------------------------------------------------------------
# Normalize and kNN
# omit the unneccesarry variables
# year , horsepower , owner count, 
names(training_df)
training_df <- training_df[ , -c(1:14, 16:20, 22, 24:25, 27:34, 36, 38:40, 42:53, 55:58)]
validation_df <- validation_df[ , -c(1:14, 16:20, 22, 24:25, 27:34, 36, 38:40, 42:53, 55:58)]
str(training_df)
nrow(training_df)
nrow(validation_df)
sum(is.na(training_df))
sum(is.na(validation_df))

train_norm <- training_df
valid_norm <- validation_df
names(training_df)

norm_values <- preProcess(training_df[, -c(9)],
                          method = c("center",
                                     "scale"))
train_norm[, -c(9)] <- predict(norm_values,
                               training_df[, -c(9)])
head(train_norm)

valid_norm[, -c(9)] <- predict(norm_values,
                               validation_df[, -c(9)])
head(valid_norm)

car1_norm <- predict(norm_values, car1)
car1_norm

names(car1_norm)
car1_norm <- car1_norm[ , -c(1:13, 15:19, 21, 23:24, 26:33, 35, 37:39, 41:52, 54:58)]
car1_norm <- replace(car1_norm, car1_norm=='', NA) #convert missing strings to NA
car1_norm <- drop_na(car1_norm) #drops row of NA values

train_norm <- drop_na(train_norm)
valid_norm <- drop_na(valid_norm)


#------------------------------------------------------------------------------
# kNN = 3
knn_model_k3 <- caret::knn3(price_nom ~ ., data = train_norm, k = 3)
knn_model_k3
knn_pred_k3_train <- predict(knn_model_k3, newdata = train_norm[, -c(9)], 
                             type = "class")
head(knn_pred_k3_train)

confusionMatrix(knn_pred_k3_train, as.factor(train_norm[, 9]),
                positive = "1")

knn_pred_k3_valid <- predict(knn_model_k3, newdata = valid_norm[, -c(9)], type = "class")
head(knn_pred_k3_valid)

knn_pred_k3_valid_prob <- predict(knn_model_k3, newdata = valid_norm[, -c(9)])
head(knn_pred_k3_valid_prob)

confusionMatrix(knn_pred_k3_valid, as.factor(valid_norm[, 9]),
                positive = "1")

car_predict_3 <- predict(knn_model_k3, 
                         newdata = car1_norm,
                         type = "class")
car_predict_3

car_predict_prob_3 <- predict(knn_model_k3, 
                              newdata = car1_norm,
                              type = "prob")
car_predict_prob_3

ROSE::roc.curve(valid_norm$price_nom, knn_pred_k3_valid)
#------------------------------------------------------------------------------
# kNN = 7
knn_model_k7 <- caret::knn3(price_nom ~ ., data = train_norm, k = 7)
knn_model_k7
knn_pred_k7_train <- predict(knn_model_k7, newdata = train_norm[, -c(9)], type = "class")
head(knn_pred_k7_train)

confusionMatrix(knn_pred_k7_train, as.factor(train_norm[, 9]),
                positive = "1")

knn_pred_k7_valid <- predict(knn_model_k7, newdata = valid_norm[, -c(9)], type = "class")
head(knn_pred_k7_valid)

knn_pred_k7_valid_prob <- predict(knn_model_k7, newdata = valid_norm[, -c(9)])
head(knn_pred_k7_valid_prob)

confusionMatrix(knn_pred_k7_valid, as.factor(valid_norm[, 9]),
                positive = "1")



car_predict_7 <- predict(knn_model_k7, 
                         newdata = car1_norm,
                         type = "class")
car_predict_7
car_predict_prob_7 <- predict(knn_model_k7, 
                                  newdata = car1_norm,
                                  type = "prob")
car_predict_prob_7

ROSE::roc.curve(valid_norm$price_nom, knn_pred_k7_valid)
#------------------------------------------------------------------------------
#Classification Tree - Confusion Matrix, ROC, and prediction
class_tree <- rpart(price_nom ~ mileage + has_accidents + is_new + 
                      frame_damaged + year + horsepower + seller_rating + owner_count,
                    data = training_df, method = "class",
                    maxdepth = 30)
prp(class_tree, cex = 0.8, tweak = 1)

# Confusion Matrix from Classification Tree - change y
class_tree_train_predict <- predict(class_tree, training_df,
                                    type = "class")

t(t(head(class_tree_train_predict,10)))
training_df$price_nom <- as.factor(training_df$price_nom)


confusionMatrix(class_tree_train_predict, training_df$price_nom, positive = "1")

class_tree_valid_predict <- predict(class_tree, validation_df,
                                    type = "class")
summary(class_tree_valid_predict)

t(t(head(class_tree_valid_predict,10)))
validation_df$price_nom <- as.factor(validation_df$price_nom)
confusionMatrix(class_tree_valid_predict, validation_df$price_nom, positive = "1")

class_tree_valid_predict_prob <- predict(class_tree, validation_df,
                                         type = "prob")

head(class_tree_valid_predict_prob)

# ROC - change 
ROSE::roc.curve(validation_df$price_nom, class_tree_valid_predict)

# predict the model
price_class_tree <- predict(class_tree, newdata = car1,
                            type = "class")
price_class_tree

price_class_tree <- predict(class_tree, newdata = car1,
                            type = "prob")
price_class_tree
#------------------------------------------------------------------------------
# Weighted 
# kNN = 3
training_df_rose <- ROSE(price_nom ~ mileage + has_accidents + is_new + 
                           frame_damaged + year + horsepower + seller_rating + owner_count,
                         data = training_df, seed = 666)$data
table(training_df_rose$price_nom)

train_norm_2 <- training_df_rose
valid_norm_2 <- validation_df

train_norm_2 <- drop_na(train_norm_2)
valid_norm_2 <- drop_na(valid_norm_2)

names(train_norm_2)

knn_model_k3_rose <- caret::knn3(price_nom ~ ., data = train_norm_2, k = 3)
knn_model_k3_rose

knn_pred_k3_train_rose <- predict(knn_model_k3_rose, newdata =
                              train_norm_2[,-c(9)],
                            type = "class")
head(knn_pred_k3_train_rose)

confusionMatrix(knn_pred_k3_train_rose, as.factor(train_norm_2[, 9]),
                positive = "1")

knn_pred_k3_valid_rose <- predict(knn_model_k3_rose, 
                            newdata = valid_norm_2[, -c(9)],
                            type = "class")
head(knn_pred_k3_valid_rose)

confusionMatrix(knn_pred_k3_valid_rose, as.factor(valid_norm_2[, 9]),
                positive = "1")

ROSE::roc.curve(valid_norm_2$price_nom, knn_pred_k3_valid_rose)

#kNN = 7

knn_model_k7_rose <- caret::knn3(price_nom ~ ., data = train_norm_2, k = 7)
knn_model_k7_rose

knn_pred_k7_train_rose <- predict(knn_model_k7_rose, newdata =
                                    train_norm_2[,-c(9)],
                                  type = "class")
head(knn_pred_k7_train_rose)

confusionMatrix(knn_pred_k7_train_rose, as.factor(train_norm_2[, 9]),
                positive = "1")

knn_pred_k7_valid_rose <- predict(knn_model_k7_rose, 
                                  newdata = valid_norm_2[, -c(9)],
                                  type = "class")
head(knn_pred_k7_valid_rose)

confusionMatrix(knn_pred_k7_valid_rose, as.factor(valid_norm_2[, 9]),
                positive = "1")

ROSE::roc.curve(valid_norm_2$price_nom, knn_pred_k7_valid_rose)
#------------------------------------------------------------------------------
# Weighted Class Tree
class_tree_rose <- rpart(price_nom ~ mileage + has_accidents + is_new + 
                           frame_damaged + year + horsepower + seller_rating + owner_count,
                         data = training_df_rose, method = "class",
                         maxdepth = 30)
rpart.plot(class_tree_rose, type = 5)

class_tree_rose_train_predict <- predict(class_tree_rose, training_df_rose,
                                         type = "class")
summary(class_tree_rose_train_predict)

confusionMatrix(class_tree_rose_train_predict, training_df_rose$price_nom, positive = "1")

class_tree_rose_valid_predict <- predict(class_tree_rose, validation_df,
                                         type = "class")
summary(class_tree_rose_valid_predict)

confusionMatrix(class_tree_rose_valid_predict, validation_df$price_nom, positive = "1")

ROSE::roc.curve(validation_df$price_nom, class_tree_rose_valid_predict)

price_class_tree_rose <- predict(class_tree_rose, newdata = car1,
                                 type = "class")
price_class_tree_rose

price_class_tree_rose <- predict(class_tree_rose, newdata = car1,
                                 type = "prob")
price_class_tree_rose
#------------------------------------------------------------------------------
# According to our research, the two major factors that determine a car's price
# is mileage and the condition of the car. Since the condition of the car is
# vague, we will use other factors to determine the condition of the car, such as
# whether the frame of the car is damaged (frame_damaged), if the car has had accidents
# (has_accidents), if the car is new (is_new), and when the car was first released (year).
# Other minor factors that we thought were important in this model was horsepower,
# the number of owners of the car(owner_count), and the selling rate of the car. We
# think this is important because horsepower can determine a good engine, which is a main
# component of a car, owner count can tell us if the car is new or used, and the selling
# rate will tell us the demand and quality of the car. 

# After looking at both the unbalanced and balanced training models of the kNN 
# and the Classification tree, the best model we chose was the balanced classification tree
# as it was one of the more accurate models in terms of ROC Curve, validation test, and was 
# able to predict all of the observations. For classification trees, missing data would not 
# create a huge problem in the predictions and for the variables we chose, there was important 
# missing data. We believe that whether the frame of the car is damaged or whether it has 
# gotten into an accident is a major factor in the condition of the car, and therefore the price. 
# Since only the classification tree was able to run all the observations, we believe it is 
# the most accurate of the model. Also all results for cars 2, 3, 4, and 5 were the same, but
# the balanced classification tree was able to predict cars 1 and 6 prices, while the kNN 
# could not. Lastly, the weighted classification tree had one of the best ROC curves, as 
# it had one of the greatest area under the curve values of the models we created.
# The area under the curve of the ROC measures the separability, which indicates a good model 
# performance the closer to 1 the AUC is. The AUC can have a value of 0 to 1.

# Our data also needed to be balanced because of we thought the data's proportion
# was overwhelmingly leaning to one class. Since it was mostly one class, this could
# lead to inaccuracies, such as the lower sensitivity values we received in the unbalanced 
# models. So by balancing the data, it could lead to better results, and higher percentages.
# Also, since we only balanced the training data, the confusion matrix models' positive
# predicted values (precision) rose for the training data, while the validation data's 
# precision went down. Precision is the metric of the model's ability to predict
# the true positives compared to all the predicted positives. Using a balanced 
# classification tree training set with an unbalanced validation set, this created more
# values of 1 in the validations set, but more inaccurate predictions. 

# Sensitivity is the metric of the model's ability to predict true positive values for
# each category. The sensitivity of the model increased for both models for the 
# weighted training set compared to the regular training set, however it decreased for
# both models for the validation set. Specificity is the metric of the model's ability 
# to predict true negative values for each category. The specificity of the model increased 
# for both models for the weighted training set compared to the regular training set, 
# however it also decreased for both models for the validation set. This could
# possibly be due to the validation set not being weighted compared to the training set.

# Note that some of the limitations of using a classification tree is that it is prone
# to overfitting. Overfitting is when the model fits the training data, and has a 
# difficult time to predict new records. This can be seen in both the kNN and classification
# tree models as most of the data from the car2 dataframe was majority 0 or negative.
# The negative predicted value was very high, while the positive predicted value was
# very low. This shows that overfitting occured in the final predictions and may not be
# that accurate when predicting positive or 1 values. 

# Lastly, we want a model with the highest sensitivity, specificity, positive predicited value, 
# negative predicted value, and accuracy; however no model will ever be perfect and 
# there will always be limitations to any model that we use. 

# With that being said, our model predicted the price of cars 1 and 6 were 1s and cars 
# 2, 3, 4, and 5 were 0s. This means that cars 1 & 6 the price should be high, while 
# cars 2, 3, 4, and 5 should be priced low. 
