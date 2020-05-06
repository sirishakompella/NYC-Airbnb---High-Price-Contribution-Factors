library(dplyr)
library(reshape2)
library(ggplot2)
library(caret)
library(InformationValue)


# Import data from a .csv file
setwd("C:\\Users\\vchan\\Downloads\\Kompella\\6040")
#DS <- read.csv(file="titanic.csv", sep=",", header=TRUE, stringsAsFactors=FALSE)
DS <- read.csv("titanic.csv", stringsAsFactors=FALSE)

View(DS)

# Data Exploration
# 887 observations of 8 variables
head(DS)
dim(DS)
colnames(DS)
str(DS)

# No missing value
sapply(DS, typeof)
summary(DS)
colSums(is.na(DS))

# Check duplicate rows: 0 duplicate rows
DS[duplicated(DS)]

# Convert varaibles to factor
DS$Survived <- factor(DS$Survived, levels = c(0,1))
DS$Pclass <- factor(DS$Pclass, levels = c(1,2,3))
DS$Sex <- factor(DS$Sex)

# Convert Fare to integer
#DS$Fare <- as.integer(DS$Fare)

str(DS)

# Find number of rows with Fare=0
# 14 rows with Fare=0
# Keep them because the crew members most likely have free tickets
nrow(DS[(DS$Fare==0),])

# Examine data summary
summary(DS)

# Visually examine outiers of price
boxplot(Fare ~ Pclass, DS)
# Remove 3 rows with the highest fare 
DS <- DS[!(DS$Fare==max(DS$Fare)),]

#Finding Survival Rates

overall_survivalrate = sum(DS$Survived == 1)/length(DS$Survived)
cat("Fraction of people survived = ", format(overall_survivalrate, digits = 3))

male_survivalrate = sum((DS$Survived==1) & (DS$Sex == "male"))/sum((DS$Sex == "male"))
cat("Fraction of men survived = ", format(male_survivalrate, digits = 3))

female_survivalrate = sum((DS$Survived==1) & (DS$Sex == "female"))/sum((DS$Sex == "female"))
cat("Fraction of women survived = ", format(female_survivalrate, digits = 3))

class1_survivalrate = sum((DS$Survived==1) & (DS$Pclass == 1))/sum((DS$Pclass == 1))
cat("Fraction of Class 1 passengers survived = ", format(class1_survivalrate, digits = 3))

class2_survivalrate = sum((DS$Survived==1) & (DS$Pclass == 2))/sum((DS$Pclass == 2))
cat("Fraction of Class 2 passengers survived = ", format(class2_survivalrate, digits = 3))

class3_survivalrate = sum((DS$Survived==1) & (DS$Pclass == 3))/sum((DS$Pclass == 3))
cat("Fraction of Class 3 passengers survived = ", format(class3_survivalrate, digits = 3))

# After data cleaning, examine descriptive statistics

# Average age by survival
DS %>% group_by(Survived) %>% summarise(Count=n(), Percent=n(), Mean_Age = mean(Age), Median_Age = median(Age)) %>% mutate(Percent = round(prop.table(Percent)*100))

# Percentage of Survival and Age by Sex
DS %>% group_by(Survived, Sex) %>% summarise(Count=n(), Percent=n(), Mean = mean(Age), Median = median(Age),
                                                         Min = min(Age),                  
                                                         Q1 = quantile(Age, 0.25), 
                                                         Q3 = quantile(Age, 0.75),
                                                         Max = max(Age),
                                                         SD = sd(Age)) %>% 
                                      group_by(Survived) %>% 
                                      mutate(Percent = round(prop.table(Percent)*100))

# Age distribution of survivors
df <- DS %>% filter(Survived==1)
ggplot(df, aes(x=Age, color=Sex, fill=Sex))+
  geom_histogram(bins=15,position="identity", alpha=0.5)+
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  labs(title="Age histogram plot",x="Age", y = "Number of Survivors")+
  theme_classic()

# Percentage of Survival in each Age group
df <- DS %>% mutate(agegroup = case_when(Age >=0 & Age <= 8 ~'1',
                                             Age >=9 & Age <= 18 ~'2',
                                             Age >=18 & Age <= 27 ~'3',
                                             Age >=27 & Age <= 36 ~'4',
                                             Age >=36 & Age <= 45 ~'5',
                                             Age >=45 & Age <= 54 ~'6',
                                             Age >=54 & Age <= 63 ~'7',
                                             Age >=63 & Age <= 72 ~'8',
                                             Age >=72 & Age <= 80 ~'9'))
df %>% filter(Survived==1) %>% group_by(agegroup) %>% summarise(Count=n(), Percent=n(), Mean = mean(Age), Median = median(Age), Min=min(Age), Max=max(Age)) %>% mutate(Percent = round(prop.table(Percent)*100))

# Percentage of Survival and Fare by Pclass
DS %>% group_by(Survived, Pclass) %>% summarise(Count=n(), Percent=n()) %>% group_by(Pclass) %>% mutate(Percent = round(prop.table(Percent)*100))

# Fare by Pclass
DS %>% group_by(Pclass) %>% summarise(Mean_Fare=mean(Fare), Median_Fare=median(Fare), Min_Fare=min(Fare), Max_Fare=max(Fare)) 

# Percentage of Survival by Siblings.Spouses.Aboard
DS %>% group_by(Survived, Siblings.Spouses.Aboard) %>% summarise(Count=n(), Percent=n(), Age=mean(Age)) %>% mutate(Percent = round(prop.table(Percent)*100)) 

# Percentage of Survival by Parents.Children.Aboard
DS %>% group_by(Survived, Parents.Children.Aboard) %>% summarise(Count=n(), Percent=n(), Age=mean(Age)) %>% mutate(Percent = round(prop.table(Percent)*100)) 

# Percentage of Survival by Parents.Children.Aboard
df <- DS %>% mutate(parents_children_group = case_when(Parents.Children.Aboard == 0 ~'1',
                                                         Parents.Children.Aboard >0  ~'2'))
df %>% filter(Survived==1) %>% group_by(parents_children_group) %>% summarise(Count=n(), Percent=n(), Mean = mean(Parents.Children.Aboard), Median = median(Parents.Children.Aboard), Min=min(Parents.Children.Aboard), Max=max(Parents.Children.Aboard)) %>% mutate(Percent = round(prop.table(Percent)*100))
df %>% filter(Survived==0) %>% group_by(parents_children_group) %>% summarise(Count=n(), Percent=n(), Mean = mean(Parents.Children.Aboard), Median = median(Parents.Children.Aboard), Min=min(Parents.Children.Aboard), Max=max(Parents.Children.Aboard)) %>% mutate(Percent = round(prop.table(Percent)*100))

# Percentage of Survival by Siblings.Spouses.Aboard
df <- DS %>% mutate(siblings_spouses_group = case_when(Siblings.Spouses.Aboard == 0 ~'1',
                                                           Siblings.Spouses.Aboard >0  ~'2'))
df %>% filter(Survived==1) %>% group_by(siblings_spouses_group) %>% summarise(Count=n(), Percent=n(), Mean = mean(Siblings.Spouses.Aboard), Median = median(Siblings.Spouses.Aboard), Min=min(Siblings.Spouses.Aboard), Max=max(Siblings.Spouses.Aboard)) %>% mutate(Percent = round(prop.table(Percent)*100))
df %>% filter(Survived==0) %>% group_by(siblings_spouses_group) %>% summarise(Count=n(), Percent=n(), Mean = mean(Siblings.Spouses.Aboard), Median = median(Siblings.Spouses.Aboard), Min=min(Siblings.Spouses.Aboard), Max=max(Siblings.Spouses.Aboard)) %>% mutate(Percent = round(prop.table(Percent)*100))

# Ignore insignificant variable Name for correlation analysis:
cleanData <- DS[ ,!(colnames(DS) %in% c("Name"))]

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

ggheatmap + 
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

# Display highly correlated variables
corX = meltedCorMatrix[abs(meltedCorMatrix$value)> 0.1,]
corX[abs(corX$value) != 1.00,]

# Build logistic regression model (all variables except Name)
model1 <- glm(Survived ~ .-Name, family=binomial(link="logit"), data=DS)
summary(model1)

# Refit model (only variables with p<0,05)
model2 <- glm(Survived ~ .-Name -Parents.Children.Aboard -Fare, family=binomial(link="logit"), data=DS)
summary(model2)

# Variable importance
varImp(model1)

# Predict probability of Survived=1 from logistic model :
pred1 <- predict(model1, newdata = DS, type = "response")
pred2 <- predict(model2, newdata = DS, type = "response")

# Categorize predicted probabilities as either Survived=1 or Survived=0 based on some cutoff value 
y_pred_num1 <- ifelse(pred1 > 0.5, 1, 0)
y_pred_num2 <- ifelse(pred2 > 0.5, 1, 0)

y_predicted1 <- factor(y_pred_num1, levels=c(0, 1))
y_predicted2 <- factor(y_pred_num2, levels=c(0, 1))

y_observed <- DS$Survived

# Evaluate prediction model by calculating accuracy: Misclassification Error
mean(y_predicted1 == y_observed)
mean(y_predicted2 == y_observed)

# Compare two models
anova(model1, model2 ,test="Chisq")

confint(model1)

# Confusion Matrix
cMatrix <- confusionMatrix(y_observed, pred1, threshold = 0.5)
cMatrix

# Model accuracy = (True Positives + True Negatives)/Total
(cMatrix$`0`[1]+cMatrix$`1`[2])/sum(cMatrix)








