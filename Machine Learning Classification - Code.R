
  #--------Include Libraries-----------------------#
  #-------------------------------------------------#
  
  library(data.table)
library(dplyr) #to view glimpse of df
library(ggplot2)
library(ggbeeswarm)
library(Amelia)
library(VIM)
library(OneR)
  library(caret)
#-----------------------Import Data Set--------------#
#----------------------------------------------------#
df<-read.delim("closerdata.tab",header=TRUE, sep="\t")
summary(df)



#----------------------Subsetting the data belonging to participants at Age 50--------------------------#
df$ncdsid<-NULL
data<-df[c("bmi50","nd8rlme","nd8socf", "nd8pain","nd8wemwb","nd8phhe","nd8enfa","nd8csp14","nd8emwb","nd8genh","nd8rlmp")]
#-----------------------------Column Renaming is done 
nms <- c("BMI", "Emotion.Role","Social.Function", "Pain","Mental.Wellness","Physical.Function","Energy.Fatigue","Life.Quality","Emotional.Wellness","Gen.Health","Physical.Role.Limit")
setnames(data, nms)

#-------------------------Data Exploration
dim(data)
str(data)
head(data)
summary(data)

#..........................Visualizations Before Pre-Processing .............

mar.default <- c(5,4,4,2) + 0.1
par(mar = mar.default + c(0, 3, 0, 0)) 
par(mfrow=c(1,1))
corrplot::corrplot(cor(data), method = 'color', addCoef.col = "grey")

par(mfrow=c(2,2))
for(i in 1:11) {
  hist(data[,i], main=names(data)[i])
}

par(mfrow=c(1,2))
for(i in 1:11) {
  boxplot(data[,i], main=names(data)[i])
}



#----------------------Data Preprocessing--------------------------#
    # convert data from int to numeric
data$Emotional.Wellness<-as.numeric(data$Emotional.Wellness)
data$Life.Quality<-as.numeric(data$Life.Quality)
data$Emotion.Role<-as.numeric(data$Emotion.Role)
data$Mental.Wellness<-as.numeric(data$Mental.Wellness)
    

#----------------Remove missing values represented by -ve integers in the data.

data<-subset(data,!BMI<=-1)

data<-subset(data,!Social.Function<=-1)
data<-subset(data,!Pain<=-1)
data<-subset(data,!Mental.Wellness<=-1)

data<-subset(data,!Physical.Function<=-1)
data<-subset(data,!Life.Quality<=-1)
data<-subset(data,!Emotional.Wellness<=-1)
data<-subset(data,!Gen.Health<=-1)
data<-subset(data,!Energy.Fatigue<=-1)

#----------------------------Visualizing and removing tMissing Values represented by NA------------------#
# total NA in each column
## Visualise missing values
library(Amelia)
missmap(data, main = "Missing values vs observed")

sapply(data,function(x) sum(is.na(x))) 
missmap(data,main="Missing values vs. Observed")

#Remove missing values from dataset#

#remove NA from data
data<-data[complete.cases(data),]
missmap(data,main="Missing values vs. Observed")

#--------------------------Labeling of data on basis of Physical.Role.Limit


#add labels to the int variables 
b <- c(-Inf, 0,25,50,75, Inf)
names <- c( "Least","Reduced","Medium","Average","Active")
data$Physical.Role.Limit <- cut(data$Physical.Role.Limit, breaks = b,labels=names)
percentage <- prop.table(table(data$Physical.Role.Limit)) * 100
percentage
table(data$Physical.Role.Limit)
plot(data$Physical.Role.Limit)




#----------------------------Scaling and Normalization of data

# One could also use sequence such as df[1:2]
#dfNorm <- as.data.frame(lapply(df[1:2], normalize))

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
ind<-sapply(data, is.numeric)
data[ind] <- lapply(data[ind], normalize)
summary(data)
#------------Z-Score Standardization

data1 <- as.data.frame( scale(data[ind] ))
summary(data1)
#-------------------Visualization after scaling


for(i in 1:10) {
  boxplot(data[,i], main=names(data)[i])
}


#-----------------Visualization after Preprocessing
# Multivariate Plots
# lets look at the interactions between the variables

x<-data[,1:10]
y<-data[,11]
# scatter plot matrix
featurePlot(x=x, y=y, plot="pairs")
# We can see some clear relationships between the input attributes (trends) and between attributes and the class values (ellipses)

# We can also look at box and whisker plots of each input variable  broken down into separate plots for each class. 
# This can help to tease out obvious linear separations between the classes.
# box and whisker plots for each attribute

featurePlot(x=as.matrix(x), y=as.factor(y), plot="box")
# This is useful to see that there are clearly different distributions of the attributes for each class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=as.matrix(x), y=y, plot="density", scales=scales)
plot(data$Emotion.Role,data$Physical.Role.Limit)

#---------------Split Data into Test and Train

library(caret)
set.seed(34)
trainIndex <- createDataPartition(data$Physical.Role.Limit, p = .8, list = FALSE, times = 1) # 80/20% split of  data
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

head(dataTrain)
head(dataTest)
percentage <- prop.table(table(dataTest$Physical.Role.Limit)) * 100
percentage
percentage <- prop.table(table(dataTrain$Physical.Role.Limit)) * 100
percentage

#-----------------------------------Parameters definition for Models ------------------
#dataTrain<-dataTrain[c("bmi50","nd8pain","nd8phhe","nd8enfa","nd8csp14","nd8genh")]
#levels(dataTrain$nd8genh) <- make.names(levels(factor(dataTrain$nd8genh)))
# 10-fold crossvalidation
# The code below will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations of train-test splits. 
# We will also repeat the process 3 times for each algorithm with different splits of the data into 10 groups, in an effort to get a more accurate estimate.
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
# We are using the metric of â???oAccuracyâ??? to evaluate models. 
# This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage


fit.lda <- train(Physical.Role.Limit~., data=dataTrain, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms
# CART

fit.cart <- train(Physical.Role.Limit~., data=dataTrain,  method="rpart", metric=metric, trControl=control)
plot(fit.cart)
# kNN

fit.knn <- train(Physical.Role.Limit~., data=dataTrain, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM

fit.svm <- train(Physical.Role.Limit~., data=dataTrain,  method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Physical.Role.Limit~., data=dataTrain,  method="rf", metric=metric, trControl=control)


#----------Further dig into Decision Tree
tree <- rpart(Physical.Role.Limit ~ .,data = data,method="class",trControl=control)
plot(tree);text(tree, pretty=2)
plotcp(tree)


#### Model Evaluation
# We now have 5 models and accuracy estimations for each. 
# We need to compare the models to each other and select the most accurate.

# We can report on the accuracy of each model by first creating a list of the created models and using the summary function.
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
bwplot(results)
# plot of the model evaluation results and comparison of the spread and the mean accuracy of each model. 
# compare accuracy of models

dotplot(results)
# We can see that the most accurate model in this case was LDA

# The results for models  can be summarized as 
# summarize Best Model
print(fit.lda)# This gives a nice summary of what was used to train the model and the mean and standard deviation (SD) accuracy achieved
print(fit.cart)
print(fit.knn)
print(fit.rf)
print(fit.svm)
# Make Predictions

# The Random forest was the most accurate model 

# Predictions on actual Test Set
# Now we want to get an idea of the accuracy of the model on our validation set  dataTest



# we can run the KNN and other models directly on the validation set and summarize the results in a confusion matrix.
# estimate skill of KNN on the test dataset
predictions <- predict(fit.knn, dataTest)
confusionMatrix(predictions, dataTest$Physical.Role.Limit)
predictions <- predict(fit.svm, dataTest)
confusionMatrix(predictions, dataTest$Physical.Role.Limit)
predictions <- predict(fit.lda, dataTest)
confusionMatrix(predictions, dataTest$Physical.Role.Limit)
predictions <- predict(fit.rf, dataTest)
confusionMatrix(predictions, dataTest$Physical.Role.Limit)
predictions <- predict(fit.cart, dataTest)
confusionMatrix(predictions, dataTest$Physical.Role.Limit)
#----------------------------------------------------------------------------------------------------------
  
#---------------------Top contributing Variables ------------------#

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Physical.Role.Limitation'
predictors<-names(dataTrain)[!names(dataTrain) %in% outcomeName]
Predictor_variables <- rfe(dataTrain[,predictors], dataTrain[,outcomeName],
                           rfeControl = control)


