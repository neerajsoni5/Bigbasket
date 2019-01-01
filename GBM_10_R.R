
library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(readr)
library(readxl)


test<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Test_u94Q5KV.csv")
train<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Train.csv")
str(train)

test$Item_Outlet_Sales<-0
test$Source<-"test"
train$Source<-"train"

Combi<-rbind(train,test)

#write.csv(Combi,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv",row.names = F)

#map<-read_excel("C:/Users/SoniNe02/Desktop/Others/Bigbasket/check.xlsx",sheet=1)

#Combi1<-read.csv("C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv")
library(stringr)
Combi$ItemId<-as.factor(str_sub(Combi$Item_Identifier, start = 1L, end = 2))


str(Combi)

library(plyr)
# let's replace them accordingly


levels(Combi$Item_Fat_Content)

Combi$Item_Fat_Content <- revalue(Combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "low fat" = "Low Fat", "reg" = "Regular"))



Combi$Item_Visibility[Combi$Item_Visibility==0]<-NA
Combi$Outlet_Year<-(2018-Combi$Outlet_Establishment_Year)
#Combi$Item_MRP<-log(Combi$Item_MRP)
#Combi$Item_Outlet_Sales<-log(Combi$Item_Outlet_Sales)

summary(Combi)

train_1<-Combi[which(Combi$Source=="train"),]
summary(train_1)
str(train_1)
train_1<-train_1[,c(2:7,9:11,14:15,12)]



#train_1<-train[,c(2:12)]
#train_1$Item_Outlet_Sales<-log(train_1$Item_Outlet_Sales)

#install.packages("AmesHousing")



#set.seed(123)
#ames_split <- initial_split(AmesHousing::make_ames(), prop = .7)
#ames_train <- training(ames_split)
#ames_test  <- testing(ames_split)

names(train_1)
str(train_1)
##gbm
# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = Item_Outlet_Sales ~ .,
  distribution = "gaussian",
  data = train_1,
  n.trees = 3000,
  interaction.depth = 1,
  shrinkage = 0.001,###Closse to one
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

#print results
print(gbm.fit)

# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))
#we see that the minimum CV RMSE is 29133 (this means on average our model is about $29,133 off from the actual sales price)

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")


##Run a model without log transformation
test_1<-Combi[which(Combi$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:13)]
#test_1<-na.omit(test_1)
str(test_1)



# predict values for test data
pred <- predict(gbm.fit, n.trees = gbm.fit$n.trees, test_1[,c(2:13)])
test_1$Item_Outlet_Sales<-(pred)

sample<-test_1[,c("Item_Identifier","Outlet_Identifier","Item_Outlet_Sales")]
sum(sample$Item_Outlet_Sales)

write.csv(sample,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission20.csv",row.names=F)

###1300
############################################################################################################33
#the plot also illustrates that the CV error is still decreasing at 10,000 trees.

##Now
#increase the learning rate to take larger steps down the gradient descent, 
#reduce the number of trees (since we are reducing the learning rate), 
#and increase the depth of each tree from using a single split to 3 splits.
#This model takes about 90 seconds to run and achieves a significantly lower RMSE than our initial model with only 1,260 trees.



###############################################################################################################



# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1,3,5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(train_1), nrow(train_1))
random_ames_train <- train_1[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Item_Outlet_Sales ~.,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = Item_Outlet_Sales ~.,
  distribution = "gaussian",
  data = train_1,
  n.trees =23,
  interaction.depth = 3,
  shrinkage = 0.3,
  n.minobsinnode = 5,
  bag.fraction = 0.8, 
  train.fraction = 0.75,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

print(gbm.fit.final)

options(scipen=999)
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)


# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit.final$cv.error)
# get MSE and compute RMSE
sqrt(gbm.fit.final$cv.error[min_MSE])
## [1] 23112.1

##Run a model without log transformation
test_1<-Combi[which(Combi$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:13)]
#test_1<-na.omit(test_1)
str(test_1)



# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, test_1[,c(2:13)])
test_1$Item_Outlet_Sales<-(pred)

sample<-test_1[,c("Item_Identifier","Outlet_Identifier","Item_Outlet_Sales")]
sum(sample$Item_Outlet_Sales)

write.csv(sample,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission21.csv",row.names=F)
##1164(Without missing value imputtion)
# results
caret::RMSE(pred, ames_test$p)
##############################################################################################################################################
#####With missing value imputation

test<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Test_u94Q5KV.csv")
train<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Train.csv")
str(train)

test$Item_Outlet_Sales<-0
test$Source<-"test"
train$Source<-"train"

Combi<-rbind(train,test)

#write.csv(Combi,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv",row.names = F)

#map<-read_excel("C:/Users/SoniNe02/Desktop/Others/Bigbasket/check.xlsx",sheet=1)

#Combi1<-read.csv("C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv")
library(stringr)
Combi$ItemId<-as.factor(str_sub(Combi$Item_Identifier, start = 1L, end = 3))


str(Combi)

library(plyr)
# let's replace them accordingly


levels(Combi$Item_Fat_Content)

Combi$Item_Fat_Content <- revalue(Combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "low fat" = "Low Fat", "reg" = "Regular"))


# weights and standard deviations by item identifier
weightsByItem <- as.data.frame( ddply(na.omit(Combi), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Weight), 
                                      sd=sd(Item_Weight)))

# we can now use these values to fill in the missing weight values:
Combi$Item_Weight <- ifelse(is.na(Combi$Item_Weight), 
                            weightsByItem$mean[
                              match(Combi$Item_Identifier, weightsByItem$Item_Identifier)], Combi$Item_Weight)



Combi$Outlet_Size[Combi$Outlet_Size==""]<-"Small"
class(Combi$Outlet_Size)
levels(Combi$Outlet_Size)


Combi$Item_Visibility[(Combi$Item_Visibility==0)]<-NA

#Visibility and standard deviations by item identifier
VisByItem <- as.data.frame( ddply(na.omit(Combi), 
                                  ~Item_Identifier, 
                                  summarise, 
                                  mean=mean(Item_Visibility), 
                                  sd=sd(Item_Visibility)))


# we can now use these values to fill in the missing vis values:
Combi$Item_Visibility <- ifelse(is.na(Combi$Item_Visibility), 
                                VisByItem$mean[
                                  match(Combi$Item_Identifier, VisByItem$Item_Identifier)], Combi$Item_Visibility)

Combi$Outlet_Year<-(2018-Combi$Outlet_Establishment_Year)


summary(Combi)

train_1<-Combi[which(Combi$Source=="train"),]
summary(train_1)
str(train_1)
train_1<-train_1[,c(2:7,9:11,14:15,12)]

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1,3,5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(train_1), nrow(train_1))
random_ames_train <- train_1[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Item_Outlet_Sales ~.,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = Item_Outlet_Sales ~.,
  distribution = "gaussian",
  data = train_1,
  n.trees =50,
  interaction.depth = 3,
  shrinkage = 0.10,
  n.minobsinnode = 5,
  bag.fraction = 1, 
  train.fraction = 0.75,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

print(gbm.fit.final)

options(scipen=999)
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)


# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit.final$cv.error)
# get MSE and compute RMSE
sqrt(gbm.fit.final$cv.error[min_MSE])
## [1] 23112.1

##Run a model without log transformation
test_1<-Combi[which(Combi$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:13)]
#test_1<-na.omit(test_1)
str(test_1)



# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, test_1[,c(2:13)])
test_1$Item_Outlet_Sales<-(pred)

sample<-test_1[,c("Item_Identifier","Outlet_Identifier","Item_Outlet_Sales")]
sum(sample$Item_Outlet_Sales)

write.csv(sample,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission22.csv",row.names=F)

######################1159
###########################################################################################################################################
##With item id =2, with 3 was 4th highest influencer
test<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Test_u94Q5KV.csv")
train<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Train.csv")
str(train)

test$Item_Outlet_Sales<-0
test$Source<-"test"
train$Source<-"train"

Combi<-rbind(train,test)

#write.csv(Combi,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv",row.names = F)

#map<-read_excel("C:/Users/SoniNe02/Desktop/Others/Bigbasket/check.xlsx",sheet=1)

#Combi1<-read.csv("C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv")
library(stringr)
Combi$ItemId<-as.factor(str_sub(Combi$Item_Identifier, start = 1L, end = 2))


str(Combi)

library(plyr)
# let's replace them accordingly


levels(Combi$Item_Fat_Content)

Combi$Item_Fat_Content <- revalue(Combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "low fat" = "Low Fat", "reg" = "Regular"))


# weights and standard deviations by item identifier
weightsByItem <- as.data.frame( ddply(na.omit(Combi), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Weight), 
                                      sd=sd(Item_Weight)))

# we can now use these values to fill in the missing weight values:
Combi$Item_Weight <- ifelse(is.na(Combi$Item_Weight), 
                            weightsByItem$mean[
                              match(Combi$Item_Identifier, weightsByItem$Item_Identifier)], Combi$Item_Weight)



Combi$Outlet_Size[Combi$Outlet_Size==""]<-"Small"
class(Combi$Outlet_Size)
levels(Combi$Outlet_Size)


Combi$Item_Visibility[(Combi$Item_Visibility==0)]<-NA

#Visibility and standard deviations by item identifier
VisByItem <- as.data.frame( ddply(na.omit(Combi), 
                                  ~Item_Identifier, 
                                  summarise, 
                                  mean=mean(Item_Visibility), 
                                  sd=sd(Item_Visibility)))


# we can now use these values to fill in the missing vis values:
Combi$Item_Visibility <- ifelse(is.na(Combi$Item_Visibility), 
                                VisByItem$mean[
                                  match(Combi$Item_Identifier, VisByItem$Item_Identifier)], Combi$Item_Visibility)

Combi$Outlet_Year<-(2018-Combi$Outlet_Establishment_Year)
summary(Combi)

unique(Combi$Outlet_Year)
table(Combi$Outlet_Year)

Combi$OutltNew<- as.factor(ifelse(Combi$Outlet_Year<15,"New","Old"))
a<-aggregate(Item_Outlet_Sales ~ Item_Type+Outlet_Identifier,data=Combi,mean)
names(a)[3]<-"avg"
Combi$Outlet_Establishment_Year<-NULL

Combi2<-merge(Combi,a,by=c("Item_Type","Outlet_Identifier"),all.x=T)

#Combi$
str(Combi2)

train_1<-Combi2[which(Combi2$Source=="train"),]
summary(train_1)
str(train_1)
train_1<-train_1[,c(1:2,4:10,13:15,11)]

# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1,3,5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(train_1), nrow(train_1))
random_ames_train <- train_1[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = Item_Outlet_Sales ~.,
    distribution = "gaussian",
    data = random_ames_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}


hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)

#for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = Item_Outlet_Sales ~.,
  distribution = "gaussian",
  data = train_1,
  n.trees =36,
  interaction.depth = 3,
  shrinkage = 0.30,
  n.minobsinnode = 15,
  bag.fraction = 1, 
  train.fraction = 0.75,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

print(gbm.fit.final)

options(scipen=999)
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)


# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit.final$cv.error)
# get MSE and compute RMSE
sqrt(gbm.fit.final$cv.error[min_MSE])
## [1] 23112.1

##Run a model without log transformation
test_1<-Combi[which(Combi$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:13)]
#test_1<-na.omit(test_1)
str(test_1)



# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, test_1[,c(2:13)])
test_1$Item_Outlet_Sales<-(pred)

sample<-test_1[,c("Item_Identifier","Outlet_Identifier","Item_Outlet_Sales")]
sum(sample$Item_Outlet_Sales)

write.csv(sample,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission27.csv",row.names=F)
##1154
########################################################################################

unique(Combi$Outlet_Year)
















##RF for regression
test<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Test_u94Q5KV.csv")
train<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Train.csv")
str(train)

test$Item_Outlet_Sales<-0
test$Source<-"test"
train$Source<-"train"

Combi<-rbind(train,test)

str(Combi)
##Impute missing value
##Arrnge Levels for Fat

library(plyr)
# let's replace them accordingly


levels(Combi$Item_Fat_Content)

Combi$Item_Fat_Content <- revalue(Combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "low fat" = "Low Fat", "reg" = "Regular"))


#levels(Combi$Outlet_Size)

#Combi$Outlet_Size <- revalue(Combi$Outlet_Size,
#                                 c("small" = "small",""""="small" "Medium" = "Medium","High"="High"))

unique(Combi$Item_Fat_Content)
# weights and standard deviations by item identifier
weightsByItem <- as.data.frame( ddply(na.omit(Combi), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Weight), 
                                      sd=sd(Item_Weight)))

# we can now use these values to fill in the missing weight values:
Combi$Item_Weight <- ifelse(is.na(Combi$Item_Weight), 
                            weightsByItem$mean[
                              match(Combi$Item_Identifier, weightsByItem$Item_Identifier)], Combi$Item_Weight)



Combi$Outlet_Size[Combi$Outlet_Size==""]<-"Small"
class(Combi$Outlet_Size)
levels(Combi$Outlet_Size)


Combi$Item_Visibility[(Combi$Item_Visibility==0)]<-NA

#Visibility and standard deviations by item identifier
VisByItem <- as.data.frame( ddply(na.omit(Combi), 
                                  ~Item_Identifier, 
                                  summarise, 
                                  mean=mean(Item_Visibility), 
                                  sd=sd(Item_Visibility)))


# we can now use these values to fill in the missing vis values:
Combi$Item_Visibility <- ifelse(is.na(Combi$Item_Visibility), 
                                VisByItem$mean[
                                  match(Combi$Item_Identifier, VisByItem$Item_Identifier)], Combi$Item_Visibility)



Combi$Outlet_Establishment_Year<-as.factor(Combi$Outlet_Establishment_Year)
str(Combi)

train_1<-Combi[which(Combi$Source=="train"),]
train_1<-train_1[,c(1:12)]
str(train_1)

library(randomForest)
library(caret)
set.seed(71) 
rf <-randomForest(Item_Outlet_Sales~. ,data=train_1, ntree=1000) 
print(rf)


mtry <- tuneRF(train_1[-1],train_1$Item_Outlet_Sales, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)


model2 <- randomForest(Item_Outlet_Sales~ . ,data=train_1, ntree = 3000, mtry = 9, importance = TRUE)
model2

##test split
##Run a model without log transformation
test_1<-Combi[which(Combi$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:11)]
#test_1<-na.omit(test_1)
str(test_1)
# predict values for test data
pred <- predict(model2, n.trees = 1000, test_1[,c(2:11)])
test_1$Item_Outlet_Sales<-(pred)

sample2<-test_1[,c("Item_Identifier","Outlet_Identifier","Item_Outlet_Sales")]

write.csv(sample,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission9_rf.csv",row.names=F)
#####################################################################################################################
####XGBOOST
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

##With item id =2, with 3 was 4th highest influencer
test<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Test_u94Q5KV.csv")
train<-read.csv("file:///C:/Users/SoniNe02/Desktop/Others/Bigbasket/Train.csv")
str(train)

test$Item_Outlet_Sales<-0
test$Source<-"test"
train$Source<-"train"

Combi<-rbind(train,test)

#write.csv(Combi,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv",row.names = F)

#map<-read_excel("C:/Users/SoniNe02/Desktop/Others/Bigbasket/check.xlsx",sheet=1)

#Combi1<-read.csv("C:/Users/SoniNe02/Desktop/Others/Bigbasket/combi.csv")
library(stringr)
Combi$ItemId<-as.factor(str_sub(Combi$Item_Identifier, start = 1L, end = 2))


str(Combi)

library(plyr)
# let's replace them accordingly


levels(Combi$Item_Fat_Content)

Combi$Item_Fat_Content <- revalue(Combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "low fat" = "Low Fat", "reg" = "Regular"))


# weights and standard deviations by item identifier
weightsByItem <- as.data.frame( ddply(na.omit(Combi), 
                                      ~Item_Identifier, 
                                      summarise, 
                                      mean=mean(Item_Weight), 
                                      sd=sd(Item_Weight)))

# we can now use these values to fill in the missing weight values:
Combi$Item_Weight <- ifelse(is.na(Combi$Item_Weight), 
                            weightsByItem$mean[
                              match(Combi$Item_Identifier, weightsByItem$Item_Identifier)], Combi$Item_Weight)



Combi$Outlet_Size[Combi$Outlet_Size==""]<-"Small"
class(Combi$Outlet_Size)
levels(Combi$Outlet_Size)


Combi$Item_Visibility[(Combi$Item_Visibility==0)]<-NA

#Visibility and standard deviations by item identifier
VisByItem <- as.data.frame( ddply(na.omit(Combi), 
                                  ~Item_Identifier, 
                                  summarise, 
                                  mean=mean(Item_Visibility), 
                                  sd=sd(Item_Visibility)))


# we can now use these values to fill in the missing vis values:
Combi$Item_Visibility <- ifelse(is.na(Combi$Item_Visibility), 
                                VisByItem$mean[
                                  match(Combi$Item_Identifier, VisByItem$Item_Identifier)], Combi$Item_Visibility)

Combi$Outlet_Year<-(2018-Combi$Outlet_Establishment_Year)
summary(Combi)

unique(Combi$Outlet_Year)
table(Combi$Outlet_Year)

Combi$OutltNew<- as.factor(ifelse(Combi$Outlet_Year<15,"New","Old"))
#a<-aggregate(Item_Outlet_Sales ~ Item_Type+Outlet_Identifier,data=Combi,mean)
#names(a)[3]<-"avg"
Combi$Outlet_Establishment_Year<-NULL
Combi$Outlet_Year<-NULL
#Combi2<-merge(Combi,a,by=c("Item_Type","Outlet_Identifier"),all.x=T)

############################Hot encoding
str(Combi)
dummies <- dummyVars( ~ Item_Fat_Content+Item_Type+Outlet_Identifier+Outlet_Size+
                       Outlet_Location_Type+Outlet_Type+ItemId+OutltNew ,data = Combi)


df<- as.data.frame(predict(dummies, newdata = Combi))

combi2<-cbind(df,Combi$Item_Identifier,Combi$Item_Weight,Combi$Item_Visibility,Combi$Item_MRP,Combi$Source,Combi$Item_Outlet_Sales)
str(combi2)
names(combi2)[45:50]<-c("Item_Identifier","Item_weight","Item_Visibility","Item_MRP","Source","Item_Outlet_Sales")
combi2$Source<-as.character(combi2$Source)
str(combi2)

train_1<-combi2[which(combi2$Source=="train"),]
names(train_1)
summary(train_1)
str(train_1)
train_1<-train_1[,c(1:44,46:48,50)]
names(train_1)
y<-train_1$Item_Outlet_Sales

#Step 4: Tune and Run the model
xgb <- xgboost(data = data.matrix(train_1[,c(1:47)]), 
               label = y, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "rmse",
               objective = "reg:linear",
               #num_class = 12,
               nthread = 3
)


##test split
##Run a model without log transformation
test_1<-combi2[which(combi2$Source=="test"),]
test_1$Source<-NULL
test_1$Item_Outlet_Sales<-NULL
test_1<-test_1[,c(1:48)]
#test_1<-na.omit(test_1)
names(test_1)
# predict values for test data
pred <- predict(xgb, data.matrix(test_1[,c(1:44,46:48)]))
test_1$Item_Outlet_Sales<-(pred)

sample2<-data.frame(test_1$Item_Identifier,test$Outlet_Identifier,test_1$Item_Outlet_Sales)
names(sample2)<-c("Item_Identifier","Outlet_Identifie","Item_Outlet_Sales")
sum(sample2$Item_Outlet_Sales)

write.csv(sample2,"C:/Users/SoniNe02/Desktop/Others/Bigbasket/SampleSubmission33_xg.csv",row.names=F)

###########################################################################################################################
# predict values in test set
y_pred <- 













