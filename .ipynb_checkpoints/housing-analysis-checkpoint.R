library(corrplot) # for correlation plot
library(RColorBrewer) # color palette library
library(readxl) # to read excel file
library(randomForest) # for the random forest model
library(caTools) # contains several basic utility functions
library(caret) # general purpose machine learning library
library(e1071) # contains svm function

# Analysis

data = read_excel("housing.xls")
head(data)

str(data) # most columns are numerical variables

summary(data) # here we find a summary of our dataset

colSums(is.na(data)) # we observe that there is no missing values in our dataset



# bigger red dots implies negative correlation while bigger blue dots implies positive correlation
# 
# we first notice that median value of homes is positively correlated with number of rooms per dwelling which is obvious.
# 
# we notice that lower status is positively correlated with crime rates and negatively correlated with home values
# and nitric oxide concentration which is emitted mostly by vehicles and industrial equiment is positively correlated to percentage of lower status.
# 
# from the aforementioned we conclude that the population near the industrial zone tend to be lower status which feeds more into the crime rate.
# 
# moreover student to teacher ratio indicated by ptratio is negatively correlated with home values which makes sense since the less teachers available for students in a specific town the less valuable the homes in that town will be due to lack of education and eventually lower status.
# 
# we will explore these findings further in later visualizations to validate our analysis.


M <-cor(data)
corrplot(M, order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"))


plot(MEDV~RM, data)

plot(MEDV~LSTAT, data)

plot(CRIM~LSTAT, data)

plot(NOX~LSTAT, data)

plot(MEDV~PTRATIO, data)

plot(data[,c(1,7,6,12,13,14)],pch=3)


# These plots confirm our findings which we restate :
# 
# * median value of homes is positively correlated with number of rooms per dwelling.
# * lower status is positively correlated with crime rates and negatively correlated with home values.
# * nitric oxide concentration which is emitted mostly by vehicles and industrial equiment is positively correlated to percentage of lower status.
# * population near the industrial zone tend to be lower status which feeds more into the crime rate.
# * student to teacher ratio indicated by ptratio is negatively correlated with home values which makes sense since the less teachers available for students in a specific town the less valuable the homes in that town will be due to lack of education and eventually lower status.


# Regression

# After visualizing our target variable we see that it's a little bit skewed to the right and not normally distributed
densityplot(data$MEDV)

set.seed(12345)

# we partition our data into train and test datasets so we can benchmark the performance of our model
inTrain <- createDataPartition(y = data$MEDV, p = 0.70, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]

# We use a simple linear regression model at first as a baseline model to compare our improvements to.
fit.lm <- lm(MEDV~.,data = training)
summary(fit.lm)

set.seed(12345)
#predict on test set
pred.lm <- predict(fit.lm, newdata = testing)

# Root-mean squared error
rmse.lm <- sqrt(sum((pred.lm - testing$MEDV)^2)/
                  length(testing$MEDV))
c(RMSE = rmse.lm, R2 = summary(fit.lm)$r.squared)

# Our baseline RMSE (root mean squared error) is 3.919 which we will aim to improve on using feature selection and transformations

plot(pred.lm,testing$MEDV, xlab = "Predicted Price", ylab = "Actual Price", col="blue")

layout(matrix(c(1,2,3,4),2,2))
plot(fit.lm)


# We try a log transform on our target variable as to make it more normally distributed the thing that really did improve our RMSE by a significant amount.
set.seed(12345)
fit.lm1 <- lm(log(MEDV)~.,data = training)

set.seed(12345)
pred.lm1 <- predict(fit.lm1, newdata = testing)

rmse.lm1 <- sqrt(sum((exp(pred.lm1) - testing$MEDV)^2)/
                   length(testing$MEDV))

c(RMSE = rmse.lm1, R2 = summary(fit.lm1)$r.squared)


summary(fit.lm1)

plot(exp(pred.lm1),testing$MEDV, xlab = "Predicted Price", ylab = "Actual Price", col="blue")


layout(matrix(c(1,2,3,4),2,2))
plot(fit.lm1)

set.seed(12345)
#Try simple linear model using selected features
fit.lm2 <- lm(formula = log(MEDV) ~ CRIM + CR01 + CHAS + NOX + RM + DIS + 
                RAD + TAX + PTRATIO + LSTAT, data = training)

set.seed(12345)
#predict on test set
pred.lm2 <- predict(fit.lm2, newdata = testing)

# Root-mean squared error
rmse.lm2 <- sqrt(sum((exp(pred.lm2) - testing$MEDV)^2)/
                   length(testing$MEDV))

c(RMSE = rmse.lm2, R2 = summary(fit.lm2)$r.squared)

# Linear regression models are kinda limited due to them being linear 
# and especially when the data we have is a bit noisy and has some of outliers 
# so we test a random forest model which based on decision trees for our regression, 
# to see if a non-linear regression model can do better

set.seed(12345)
fit.rf <- randomForest(formula = MEDV ~ CRIM + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + LSTAT, data = training)

set.seed(12345)
pred.rf <- predict(fit.rf, newdata = testing)

rmse.rf <- sqrt(sum((pred.rf - testing$MEDV)^2)/
                  length(testing$MEDV))
c(RMSE = rmse.rf, R2 = 1 - (sum((testing$MEDV-pred.rf)^2)/sum((testing$MEDV-mean(testing$MEDV))^2)))

# We see a large improvement of RMSE after using the random forest model confirming that 
# our linear regression models are relatively restricted in how they can fit data 
# due to them being limited to capturing linear relationships and being sensitive to outliers 
# unlike non-linear regression models, however non-linear models can sometimes overfit 
# and they are also relatively harder to interpret compared linear models.

plot(fit.rf)

plot(pred.rf,testing$MEDV, xlab = "Predicted Price", ylab = "Actual Price", col="blue")

# Classification


barplot(table(data$CR01)) # We see that our target variable is balanced.

# We omit the column CRIM because it leaks information about our target variable CR01 
# since they both are indicators of crime rate. 
# (if we include it we get 100% accuracy because our model will everything it needs from that column)

data = subset(data, select = -c(CRIM))

set.seed(12345)

data$CR01 <- as.factor(data$CR01)

# we partition our data into train and test datasets so we can benchmark the performance of our model

inTrain <- createDataPartition(y = data$CR01, p = 0.70, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]


# We start off with an SVM as a classifier and we choose our features from our correlation matrix 
# in the analysis notebook, We then predict on the test dataset and 
# retrieve all the relevant metrics and statistics like accuracy and confusion matrix

set.seed(12345)
fit.svm <- svm(formula = CR01 ~ ZN + NOX + AGE + DIS + RAD + TAX + PTRATIO + LSTAT + MEDV, data = training, type = 'C-classification', kernel = 'radial')

set.seed(12345)
pred.svm <- predict(fit.svm, newdata = testing)

# we get an accuracy of 0.92 which is good but we see from the confusion matrix 
# that we misclassify relatively more areas that have higher crime rate which can be a problem.

print(fit.svm)
confusionMatrix(pred.svm, testing$CR01)
summary(fit.svm)


# We train our model on all the features first, 
# We then predict on the test dataset and 
# retrieve all the relevant metrics and statistics 
# like accuracy, confusion matrix and feature importances
set.seed(12345)
fit.rf <- fit.rf <- randomForest(formula = CR01~., data = training)

set.seed(12345)
pred.rf <- predict(fit.rf, newdata = testing)

print(fit.rf)

confusionMatrix(pred.rf, testing$CR01)

fit.rf$importance

# We get an accuracy of 0.9267 which is already very good, 
# we also find some features that our model didn't really use
# like ZN, CHAS, LSTAT, RM and MEDV


