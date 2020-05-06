library(data.table)
library(reshape2)
library(ggthemes)
library(dplyr)
library(caret)
library(car)
library(ggplot2)
#install.packages("jtools")
library(jtools)
library(MASS)
library(xgboost)
library(rpart)
#install.packages("olsrr")
library(olsrr)



#options(scipen=999)  
#options(digits=5)
options(max.print=999999)
options(warn=-1)

setwd("C:\\Users\\vchan\\Downloads\\Kompella\\6040\\ANYC")

##########################################
# Assignment 1: Data cleaning & profiling
##########################################
# Import data from a .csv file
myData <- read.csv(file="AB_NYC_2019.csv", sep=",", header=TRUE, stringsAsFactors=FALSE)

# Examine data structure
# 48895 observations of 16 variables
str(myData)

# Display data summary
# reviews_per_month: 10052 NAs ()
summary(myData)

# Display number of blank value in each column
# name: 16
# host_name: 21
# last_review: 10052
sapply(myData, function(x) sum(x == '', na.rm = TRUE))

# Display number of NA in each column
# reviews_per_month: 10052
colSums(is.na(myData))

DS <- myData

# Missing values
is.na(DS)
sum(is.na(DS))
mean(is.na(DS))

#converting blanks to NA and finding total NA
DS2=read.csv("AB_NYC_2019.csv", header=T, na.strings=c("","NA"))
sapply(DS2, function(x) sum(is.na(x)))

#checking percentage of missing values of each feature
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(DS2,2,pMiss)


summary(DS2)

# Check if last_review is blank when number_of_reviews is 0 and reviews_per_month is NA
# 10052 rows returned
# i.e. NAs in reviews_per_month and last_review are due to the fact that number_of_reviews is 0
nrow(myData %>% filter(number_of_reviews == 0 & last_review=="" & is.na(reviews_per_month)))

# Convert NAs in reviews_per_month to 0
myData$reviews_per_month[is.na(myData$reviews_per_month)] <- 0

# Convert NAs in last_review to "No Review"
myData$last_review[myData$last_review == ""] <- "No review"

summary(myData)

# Drop the rows with blank values in name and host_name 
# because the amount of data removed is insignicant
# 39 rows dropped. 48858 obs left
myData <- myData[(myData$name!="" & myData$host_name!=""),]

summary(myData)

# Convert varaibles to factor
myData$neighbourhood_group <- factor(myData$neighbourhood_group)
myData$neighbourhood <- factor(myData$neighbourhood)
myData$room_type <- factor(myData$room_type)

# Convert date to "Date" type
# myData$last_review <- as.Date(myData$last_review, format = "%Y-%m-%d")

head(myData$last_review)
str(myData)
summary(myData)

# Visually examine outiers of price
boxplot(price ~ neighbourhood_group, myData)

# Price by neighborhood_group and room_type
myData %>% group_by(neighbourhood_group, room_type) %>% summarise(Mean = mean(price), Median = median(price),
                                                                  Min = min(price),                  
                                                                  Q1 = quantile(price, 0.25), 
                                                                  Q3 = quantile(price, 0.75),
                                                                  Max = max(price),
                                                                  SD = sd(price), count=n())

summary(myData)

# Remove 11 rows where price = 0.  48847 obs left.
myData <- myData[(myData$price!=0),]

# Remove 1 row where price = 5000 on Staten Island 
myData <-myData[!(myData$price==5000 & myData$neighbourhood_group=="Staten Island"),]

# Remove 1 row where price=10000 in Queens
myData <-myData[!(myData$price==10000 & myData$neighbourhood_group=="Queens"),]

# Remove 1 row where price=2500 in Bronx
myData <-myData[!(myData$price==2500 & myData$neighbourhood_group=="Bronx"),]

# Display summary 
str(myData)
summary(myData)


# Price by neighborhood_group
myData %>% group_by(neighbourhood_group) %>% summarise(Mean = mean(price), 
                                                       Min = min(price),                  
                                                       Q1 = quantile(price, 0.25), 
                                                       Q3 = quantile(price, 0.75),
                                                       Max = max(price),
                                                       SD = sd(price), count=n())

# Price by neighborhood_group and room_type
myData %>% group_by(neighbourhood_group, room_type) %>% summarise(Mean = mean(price), Median = median(price),
                                                                  Min = min(price),                  
                                                                  Q1 = quantile(price, 0.25), 
                                                                  Q3 = quantile(price, 0.75),
                                                                  Max = max(price),
                                                                  SD = sd(price), count=n())

# Price by room_type
myData %>% group_by(room_type) %>% summarise(Mean = mean(price), Median = median(price),
                                             Min = min(price),                  
                                             Q1 = quantile(price, 0.25), 
                                             Q3 = quantile(price, 0.75),
                                             Max = max(price),
                                             SD = sd(price), count=n())


# Check duplicate rows: 0 duplicate rows
myData[duplicated(myData)]

# Compare price of all hosts vs top 10 hosts
top10 <- myData %>%
  group_by(calculated_host_listings_count)%>%
  summarize(price = mean(price))%>%
  arrange(desc(calculated_host_listings_count)) %>%
  ungroup %>%
  slice(1:10)
top10
mean(top10$price) 
summary(myData)

# Number of listings per neighborhood group
myData %>% group_by(neighbourhood_group) %>% summarise(n())

# Ignore insignificant variables for correlation analysis:
# including id, name, host_name, last_review
cleanData <- myData[ ,!(colnames(myData) %in% c("id", "name", "host_name", "last_review"))]

# Convert relevant variables to numeric for correlation analysis
cleanData <- cleanData %>% mutate_if(is.factor, as.numeric)
str(cleanData)
summary(cleanData)

# Show correlations
cor(cleanData)

# Create a correlation matrix
corMatrix <- round(cor(cleanData),2)

# Convert the correlation matrix into pairs of variables with their corresponding correlation coefficient


# Helper functions
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
corMatrix <- reorder_cormat(corMatrix)
upper_tri <- get_upper_tri(corMatrix)

# Melt the correlation matrix
meltedCorMatrix <- melt(upper_tri, na.rm = TRUE)

# Create a ggheatmap for correlation matrix
ggheatmap <- ggplot(meltedCorMatrix, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 10, hjust = 1))+
  coord_fixed()

heatmap <- ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 2) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))
heatmap

# Display highly correlated variables
corX = meltedCorMatrix[abs(meltedCorMatrix$value)> 0.1,]
corX[abs(corX$value) != 1.00,]

# Display price distribution
th <- theme_fivethirtyeight() + theme(axis.title = element_text(), axis.title.x = element_text()) # global theme for ggplot2 objects
set.seed(252)
airbnb <- myData
airbnb_nh <- airbnb %>% group_by(neighbourhood_group) %>% summarise(price = round(mean(price), 2))
ggplot(airbnb, aes(price)) +
  geom_histogram(bins = 30, aes(y = ..density..), fill = "purple") + 
  geom_density(alpha = 0.2, fill = "purple") +
  th +
  ggtitle("Transformed distribution of price\n by neighbourhood groups",
          subtitle = expression("With" ~'log'[10] ~ "transformation of x-axis")) +
  geom_vline(data = airbnb_nh, aes(xintercept = price), size = 2, linetype = 3) +
  geom_text(data = airbnb_nh,y = 1.5, aes(x = price + 1400, label = paste("Mean  = ",price)), color = "darkgreen", size = 4) +
  facet_wrap(~neighbourhood_group) +
  scale_x_log10() 

##############################
# Assignment 2: Data Modeling
##############################

# Ignore insignificant variables for correlation analysis:
# including id, name, host_name, last_review
cleanData <- myData[ ,!(colnames(myData) %in% c("id", "name", "host_name", "last_review"))]
#cleanData <- myData[ ,!(colnames(myData) %in% c("name", "host_name", "last_review"))]
str(cleanData)

# Display correlation heatmap
heatmap

# Display highly correlated variables
corX[abs(corX$value) != 1.00,]

# latitude & longtitude by neighborhood_group
cleanData %>% group_by(neighbourhood_group) %>% summarise(Mean_latitude = mean(latitude), 
                                                          Min_latitude = min(latitude),                  
                                                          Max_latitude = max(latitude),
                                                          Mean_longitude = mean(longitude), 
                                                          Min_longitude = min(longitude),                  
                                                          Max_longitude = max(longitude), Count=n())

# Model1: Build a linear regression model with all relevant variables
# Most features seemed significant at a p-value of 0.001 :latitude, longitude, room_type, availability_365
# neighborhood_group, host_id, number of reviews, reviews_per_month, calculated_host_listings_count
# multiple R Square is less than 0.3 meaning independent values have weak effect on variation in size of 
# dependent variable.

# Adj R-sq = 0.1195 (with NAs)
model1 <- lm(price ~ ., data = cleanData)
summary(model1)

# Remove neighbourhood_group 
# Adj R-sq = 0.1195 
model1 <- lm(price ~ . -neighbourhood_group, data=cleanData)
summary(model1)

# Model2: Log Linear Regression Model on price
# Adj R-sq = 0.5453
model2 <- lm(log(price) ~ . -neighbourhood_group, data=cleanData)
summary(model2)

# Remove one variable of the highly-correlated variable pairs
# Adj R-sq = 0.5436
model2 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month -number_of_reviews, data=cleanData)
summary(model2) 
# Adj R-sq = 0.545
model2 <- lm(log(price) ~ . -neighbourhood_group -number_of_reviews, data=cleanData)
summary(model2) 
# Adj R-sq = 0.5451
model2 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month, data=cleanData)
summary(model2) 
# Adj R-sq = 0.119
#model2 <- lm(price ~ . -neighbourhood_group -reviews_per_month, data=cleanData)
#summary(model2) 


# Model3: Interaction terms
# Adj R-sq = 0.5456
model3 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month +availability_365*calculated_host_listings_count, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5521 (with NAs)
model3 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month +availability_365*calculated_host_listings_count +latitude*neighbourhood, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5523 (with NAs)
model3 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month +availability_365*calculated_host_listings_count +latitude*neighbourhood +availability_365*number_of_reviews, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5523 (with NAs)
model3 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month +availability_365*calculated_host_listings_count +latitude*neighbourhood +availability_365*number_of_reviews +availability_365*host_id, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5523 (with NAs)
model3 <- lm(log(price) ~ . -neighbourhood_group -reviews_per_month -number_of_reviews +availability_365*calculated_host_listings_count +latitude*neighbourhood +availability_365*number_of_reviews +availability_365*host_id, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5491 (with NAs)
model3 <- lm(log(price) ~ . -neighbourhood_group  +reviews_per_month*number_of_reviews +longitude*latitude +neighbourhood*longitude, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5459 
model3 <- lm(log(price) ~ . -neighbourhood_group +reviews_per_month*number_of_reviews +availability_365*calculated_host_listings_count, data=cleanData)
summary(model3) 
# Adj R-sq = 0.5463
model3 <- lm(log(price) ~ . -neighbourhood_group +reviews_per_month*number_of_reviews +longitude*latitude +availability_365*calculated_host_listings_count, data=cleanData)
summary(model3) 
# Adj R-sq = 0.1206
#model3 <- lm(price ~ . -neighbourhood_group +reviews_per_month*number_of_reviews +longitude*latitude +availability_365*calculated_host_listings_count, data=cleanData)
#summary(model3) 

# Show cofficient estimates with p-values<0.05
summary(model1)$coef[summary(model1)$coef[,4] < .05, ]
summary(model2)$coef[summary(model2)$coef[,4] < .05, ]
summary(model3)$coef[summary(model3)$coef[,4] < .05, ]

# Show cofficient estimates with p-values>=0.05
summary(model1)$coef[summary(model1)$coef[,4] >= .05, ]
summary(model3)$coef[summary(model3)$coef[,4] >= .05, ]

# Show positive cofficient estimates 
summary(model1)$coef[summary(model1)$coef[,1] > 0, ]
summary(model2)$coef[summary(model2)$coef[,1] > 0, ]
summary(model3)$coef[summary(model3)$coef[,1] > 0, ]

# Show negative cofficient estimates 
summary(model1)$coef[summary(model1)$coef[,1] < 0, ]
summary(model2)$coef[summary(model2)$coef[,1] < 0, ]
summary(model3)$coef[summary(model3)$coef[,1] < 0, ]

# Variable importance in order in model3
vImp <- varImp(model3)
vImp <- data.frame(Variables = rownames(vImp), Overall = vImp$Overall)
vImp[order(vImp$Overall,decreasing = T),]

# Variable inflation factor 
vif(model1)
vif(model2)   
vif(model3)

# Predict price
pred1 <- predict(model1, newdata = cleanData, type = "response")
pred2 <- predict(model2, newdata = cleanData, type = "response")
pred3 <- predict(model3, newdata = cleanData, type = "response")

# RMSE
RMSE1 = sqrt(mean((pred1 - cleanData$price)^2))
RMSE2 = sqrt(mean((pred2 - log(cleanData$price))^2))
RMSE3 = sqrt(mean((pred3 - log(cleanData$price))^2))
cbind(RMSE1, RMSE2, RMSE3)

# Calculate mean absoluate percentage error (MAPE)
mape1 <- mean(abs((pred1 - cleanData$price))/pred1)
mape2 <- mean(abs((pred2 - log(cleanData$price)))/pred2)
mape3 <- mean(abs((pred3 - log(cleanData$price)))/pred3)
cbind(mape1, mape2, mape3)

# Plot predicted price vs actual price
plot(pred3, lty = 1.8, type = "o", col="red", xlab="Sample Number", ylab="Price")
lines(log(cleanData$price), type = "o", col="blue")
legend("bottom", legend=c("actual", "predicted"), col=c("blue", "red"), lty=1:1)

#effect_plot(model3, pred = price, interval = TRUE, plot.points = TRUE)

plot(pred3,log(cleanData$price), col="blue",
     xlab="log(predicted price)",ylab="log(actual price)")
abline(a=0,b=1)

par(mfrow = c(2, 2))
plot(model3)
par(mfrow=c(1,1))

# Remove skewness of price by log transformation
# Distribution of price
ggplot(cleanData, aes(x=price))+
  geom_histogram(bins=20,position="identity", alpha=0.5)+
  labs(title="Price histogram plot",x="Price", y = "Number of Listings")+
  theme_classic()

# Distribution of log(price)
ggplot(cleanData, aes(x=log(price)))+
  geom_histogram(bins=20,position="identity", alpha=0.5)+
  labs(title="Price histogram plot",x="Log(Price)", y = "Number of Listings")+
  theme_classic()

#####################################
# Assignment 3: Model Optimization
#####################################

# Stepwise regression model
model0 <- lm(price ~ 1, data = cleanData)
summary(model0)

model4 <- step(model1, scope=list(lower=model0, upper=model1), direction="both")
# model4 <- ols_step_both_p(model1)

summary(model4)

# Predict price & calculate RMSE & MAPE
pred4 <- predict(model4, newdata = cleanData, type = "response")
RMSE4 = sqrt(mean((pred4 - cleanData$price)^2))
mape4 <- mean(abs((pred4 - cleanData$price))/pred4)

vImp <- varImp(model1)
vImp <- data.frame(Variables = rownames(vImp), Overall = vImp$Overall)
vImp[order(vImp$Overall,decreasing = T),]

vImp <- varImp(model4)
vImp <- data.frame(Variables = rownames(vImp), Overall = vImp$Overall)
vImp[order(vImp$Overall,decreasing = T),]

# Show cofficient estimates with p-values>=0.05
summary(model4)$coef[summary(model4)$coef[,4] >= .05, ]

# Show positive cofficient estimates 
summary(model4)$coef[summary(model4)$coef[,1] > 0, ]

# Show negative cofficient estimates 
summary(model4)$coef[summary(model4)$coef[,1] < 0, ]

# MAPE4(63.4%) is worse than MAPE1(49.3%)

# Build a regression tree 
fit <- rpart(price ~ . - neighbourhood_group, method="anova", data=numericData)

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# create additional plots
par(mfrow=c(1,1)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results  

# plot tree
plot(fit, uniform=TRUE,
     main="Regression Tree for Price ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree
post(fit, file = "tree2.ps",
     title = "Regression Tree for Price ")

# prune the tree
pfit<- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot the pruned tree
plot(pfit, uniform=TRUE,
     main="Pruned Regression Tree for Price")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
post(pfit, file = "ptree2.ps",
     title = "Pruned Regression Tree for Price")

# It turns out that this truned tree is the same as the original tree.

# Make predictions & calculate RMSE & MAPE with the tree model
tree_pred <- predict(fit, newdata = numericData)
tree_RMSE = sqrt(mean((tree_pred - numericData$price)^2))
tree_mape <- mean(abs((tree_pred - numericData$price))/tree_pred)
# tree_mape: 0.4722122

# Build model using gradient boosting
sparse_matrix <- as.matrix(numericData)
output_vector <- numericData$price
xgboost_model <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "reg:squarederror", prediction = TRUE)

importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = xgboost_model)
importance

xgb.plot.importance(importance_matrix = importance)

# Make predictions & calculate RMSE & MAPE with the gradient boosting model
xgboost_pred <- predict(xgboost_model, as.matrix(numericData)) 
xgboost_RMSE = sqrt(mean((xgboost_pred - numericData$price)^2))
xgboost_mape <- mean(abs((xgboost_pred - numericData$price))/xgboost_pred)


# Compare models
cbind(RMSE1, tree_RMSE, xgboost_RMSE)
cbind(mape1, tree_mape, xgboost_mape)




