#Analyse statistique des défauts de paiement
library(tidyverse)
library(randomForest)

#working directory
setwd("C:/Users/Luke/ProjetMagistere")

#import database
database<-read_csv("cleaned_UCI_Credit_Card.csv")

#look at missing values
mean(rowMeans(is.na(database)))
missing_data_pattern<-as_tibble(colMeans(is.na(database)))

#eliminate rows with missing values
database<-na.omit(database)

###look at the qualitative and quantitative variables
for(i in colnames(database)){
  print(i)
  lev<-levels(as.factor(database[[i]]))
  if(length(lev)<5){print(lev)}
  else{print(c(mean(database[[i]])))}
}

###try to predict the last variable via random forest
# 
# #define the train and test set
# train_proportion<-0.9
# test_proportion<-1-train_proportion
# 
# #define the train and tests sets
# row_split<-floor(train_proportion*nrow(database))
# database<-as.data.frame(database)
# xtrain<-database[1:row_split,-c(1,25)]
# xtest<-database[row_split:nrow(database),-c(1,25)]
# 
# #convert predicted variable to factor (otherwise randomForest will do a regression
# ytrain<-as.factor(database[1:row_split,c(25)])
# ytest<-as.factor(database[row_split:nrow(database),c(25)])
# 
# #apply the model
# model<-randomForest(x = xtrain,y = ytrain,xtest = xtest,ytest = ytest)
# 
# 
# model_0.8<-model
# 
#we conclude by comparing the OOB error and the test set error that manually setting a test and training set is pointless
#in fact the provided algorithm already does this when calculating OOB error

#calculate using all the data
model<-randomForest(x = database[-c(1,25)],y = as.factor(database[,25]))

#precision and recall

#precision is the rate of true positives to selected elements
#in our case this means the percentage of one's selected that were correct (i.e. percentage of default predictions that materialised into defaults)
#recall is the rate of true positives to total positives, so here the percentage of defaults that were predicted

confusion<-model$confusion
precision<-confusion[2,2]/(confusion[2,1]+confusion[2,2])
recall<-confusion[2,2]/(confusion[1,2]+confusion[2,2])

#try to optimise the threshold of the vote
xtrain = database[-c(1,25)]
ytrain = as.factor(database[[25]])

#threshold to signal alarm
thresholds<-seq(0.02,0.98,0.02)
models<-list()
for(i in seq_along(thresholds)){
  models[[i]]<-randomForest(xtrain,ytrain,cutoff = c(1-thresholds[i],thresholds[i]))
}

recalls<-lapply(models,function(x) x$confusion[2,2]/(x$confusion[2,1]+x$confusion[2,2]))
precisions<-lapply(models,function(x) x$confusion[2,2]/(x$confusion[1,2]+x$confusion[2,2]))
F1_score<-lapply(seq_along(precisions),function(i,precision,recall) 2*(precision[[i]]*recall[[i]])/(precision[[i]]+recall[[i]]), precision=precisions,recall=recalls)
F3_score<-lapply(seq_along(precisions),function(i,precision,recall) 10*(precision[[i]]*recall[[i]])/(9*precision[[i]]+recall[[i]]), precision=precisions,recall=recalls)
F5_score<-lapply(seq_along(precisions),function(i,precision,recall) 26*(precision[[i]]*recall[[i]])/(25*precision[[i]]+recall[[i]]), precision=precisions,recall=recalls)



plot_data<-as_tibble(cbind(thresholds,unlist(precisions),unlist(recalls),unlist(F1_score),unlist(F3_score),unlist(F5_score)))
names(plot_data)<-c("threshold","precision","recall","F1_score","F3_score","F5_score")

ggplot(plot_data, aes(threshold)) + 
  geom_line(aes(y = precision, colour = "Précision")) + 
  geom_line(aes(y = recall, colour = "Rappel")) +
  geom_line(aes(y = F1_score, colour = "F1 score")) +
  geom_line(aes(y = F3_score, colour = "F3 score")) +
  geom_line(aes(y = F5_score, colour = "F5 score")) +
  ggtitle("Évolution de la performance du modèle prédictif obtenu avec les forêts aléatoires en fonction du seuil") +
  labs(x="Seuil de déclenchement d'une alarme",y="Performance du modèle") +
  labs(colour="Légende")

write_csv(plot_data,"évolution de précision et recall.csv")

#calcul du seuil qui maximise le F5 score
optimal_thresh_F5<-thresholds[which.max(F5_score)]
optimal_thresh_F3<-thresholds[which.max(F3_score)]

#on choisit le score F5
thresh<-optimal_thresh_F3

#### On utilise un cutoff c(0.94,0.06)

#define the train and test set
train_proportion<-0.9
test_proportion<-1-train_proportion

#define the train and tests sets
row_split<-floor(train_proportion*nrow(database))
database<-as.data.frame(database)
xtrain<-database[1:row_split,-c(1,25)]
xtest<-database[row_split:nrow(database),-c(1,25)]

#convert predicted variable to factor (otherwise randomForest will do a regression
ytrain<-as.factor(database[1:row_split,c(25)])
ytest<-as.factor(database[row_split:nrow(database),c(25)])

#apply the model
optimal_model<-randomForest(x = xtrain,y = ytrain,xtest = xtest,ytest = ytest,cutoff = c(1-thresh,thresh))
optimal_model

confusion<-optimal_model$test$confusion
recall<-confusion[2,2]/(confusion[2,1]+confusion[2,2])
precision<-confusion[2,2]/(confusion[1,2]+confusion[2,2])
precision
recall

#variable importance
importance<-as.data.frame(optimal_model$importance) %>% rownames_to_column() %>% as_tibble() %>% arrange(MeanDecreaseGini)
View(importance)

#barplot
par(mai=c(1,2,1,1))
barplot(height = importance$MeanDecreaseGini,names.arg = importance$rowname,horiz = T,las=1,main = "Diagramme en barres montrant l'importance des variables dans le modèle de prédiction",xlab = "Perte moyenne de Gini associée à chaque variable")

########### TRY ON FILTERED DATA (FROM ROMAN)

#import clean database
clean_database<-read_csv("cleaned_UCI_Credit_Card.csv")

#filter the columns used in logit
logit_database<-clean_database %>% select(one_of("SEX","EDUCATION","MARRIAGE","AGEA","AGEB","PAY0A","PAY0B","CREDITA","CREDITB","default.payment.next.month"))

#define the train and test set
train_proportion<-0.8
test_proportion<-1-train_proportion

#define the train and tests sets
row_split<-floor(train_proportion*nrow(logit_database))
logit_database<-as.data.frame(logit_database)
xtrain<-logit_database[1:row_split,-c(10)]
xtest<-logit_database[row_split:nrow(logit_database),-c(10)]

#convert predicted variable to factor (otherwise randomForest will do a regression
ytrain<-as.factor(logit_database[1:row_split,c(10)])
ytest<-as.factor(logit_database[row_split:nrow(logit_database),c(10)])

#apply the model
optimal_model<-randomForest(x = xtrain,y = ytrain,xtest = xtest,ytest = ytest,cutoff = c(1-thresh,thresh))
optimal_model

confusion<-optimal_model$test$confusion
recall<-confusion[2,2]/(confusion[2,1]+confusion[2,2])
precision<-confusion[2,2]/(confusion[1,2]+confusion[2,2])
precision
recall
