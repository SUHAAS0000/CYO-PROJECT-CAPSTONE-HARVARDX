## HarvardX Data Science Capstone Project CYO: Credit Card Fraud Detection
## Author: PEDIREDLA SUHAAS
# Note to Reviewers/Graders:  Sorry this dataset is large, and the models take a long time to run.

# Require & Load Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr",  repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales",  repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(geosphere)) install.packages("geosphere", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart",  repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest",  repos = "http://cran.us.r-project.org")


library(tidyverse)
library(data.table)
library(readr)
library(knitr)
library(kableExtra)
library(gridExtra) 
library(scales)
library(lubridate)
library(geosphere)
library(caret)
library(rpart)
library(randomForest)


options(scipen = 999)

# Checking memory limit
memory.limit()

# Increasing memory limit to handle large dataset and models. I'm on a 64-bit Windows.
memory.limit(size=48000)


#################################################################
###         Importing Credit Card Fraud data sets             ###

# Credit card fraud dataset obtained from Kaggle at: 
# https://www.kaggle.com/kartik2112/fraud-detection

# Create temp file and download zip file containing dataset
dl <- tempfile()
download.file("https://github.com/KHHogan06/CYO-Project/releases/download/v1-files/fraud_data.zip", dl)

# unzip
unzip(dl)

# read & import csv files
fraudTest <- read_csv(unzip(dl,"fraudTest.csv"))
fraudTrain <- read_csv(unzip(dl,"fraudTrain.csv"))

# remove tempfile
rm(dl)


#################################################################
###         Exploring Pre-split FraudTrain Data               ###


# Exploring data set variables & structure
glimpse(fraudTrain)

# proportion of fraudulent transactions
prop.table(table(fraudTrain$is_fraud))

# summary of data
summary(fraudTrain)

# date ranges for provided train and test sets 
range(fraudTrain$trans_date_trans_time)
range(fraudTest$trans_date_trans_time)



#################################################################
###         Resampling training & test sets                   ###

# Data was pre-split by date. Merging together to randomly split. Must first remove X1 row number to prevent duplicates.
fraudTest <- fraudTest[-1]
fraudTrain <- fraudTrain[-1]
fraudSet <- rbind(fraudTrain, fraudTest)

# removing pre-split sets
rm(fraudTest, fraudTrain)


## Splitting data set options
# Creating Random Sampling of Training & 20% Final Test set of fraud data. 
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = fraudSet$is_fraud, times = 1, p = 0.2, list = FALSE)
train_set <- fraudSet[-test_index,]
test_set <- fraudSet[test_index,]

# proportion of fraudulent transactions
prop.table(table(train_set$is_fraud)) #.52%
prop.table(table(test_set$is_fraud)) #.52%

# removing FraudSet 
rm(fraudSet)

##################################################################
###                     Exploring Data                         ###


### Fraud ###

# Fraud vs legitimate average transaction amounts & # trans
train_set %>% group_by(is_fraud) %>% summarize(avg_trans = mean(amt), med_trans = median(amt))

train_set %>% group_by(is_fraud) %>% summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100)

# Summary of Bins by Fraud
train_set %>% mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), labels = c("<$100","$100-$999","$1K+"))) %>%
  group_by(bins, is_fraud) %>% summarize(amt = sum(amt), n = n())  %>% mutate(pct_amt = (amt/sum(amt))*100)

# Distributions of Transaction Amounts
train_set %>% filter(is_fraud == 0) %>% mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), labels = c("<$100","$100-$999","$1K+"))) %>%
  ggplot(aes(amt)) + geom_histogram(bins = 40, fill = "#56B4E9") + 
  theme_bw(base_size = 10) + ggtitle("LegitimateTransaction Amounts ($s)") +
  facet_wrap(~ bins, scales = "free")

# Distributions of Fraud Transaction Amounts
train_set %>% filter(is_fraud == 1) %>% mutate(bins = cut(amt, breaks = c(-Inf, 100, 1000, Inf), labels = c("<$100","$100-$999","$1K+"))) %>%
  ggplot(aes(amt)) + geom_histogram(bins = 40, fill = "#56B4E9") + 
  theme_bw(base_size = 10) + ggtitle("Fraudulent Transaction Amounts ($s)") +
  facet_wrap(~ bins, scales = "free")




### Categories ###

# Amounts by Category
cat_amt <- train_set %>% 
  ggplot(aes(amt)) + geom_histogram(bins = 40, fill = "56B4E9") + 
  theme_bw(base_size = 10) + ggtitle("Transaction Amounts ($s)") + facet_wrap(~category)

# Fraud Amounts by Category
fcat_amt <- train_set %>% filter(is_fraud == 1) %>%
  ggplot(aes(amt)) + geom_histogram(bins = 40, fill = "56B4E9") + 
  theme_bw(base_size = 10) + ggtitle("Fraudulent Transaction Amounts ($s)") + facet_wrap(~category)

grid.arrange(cat_amt, fcat_amt, ncol = 2)
rm(cat_amt, fcat_amt)


# Barchart with Fraud & Legit sales by category $s
train_set %>% group_by(category, is_fraud) %>% summarize(amt = sum(amt), n = n()) %>%
  ggplot(aes(x = amt,y = reorder(category,amt), fill = as.factor(is_fraud))) +
  geom_bar(stat = "identity") + labs(x="Transaction Amount", y="") + scale_fill_manual(values=c("grey68","darkorange2")) +
  theme_bw(base_size = 10) + scale_x_continuous(labels = comma) +  theme(legend.position = "bottom") +
  ggtitle("Sales Category Break Down") 

# viewing range of trans amts by category: legit vs fraud (removing extreme outlier values for better scale view)
train_set %>% filter(amt < 1200) %>% group_by(is_fraud, category) %>% 
  ggplot(aes(is_fraud,y=amt, color = as.factor(is_fraud))) +
  geom_boxplot() + theme_bw() + scale_y_continuous(breaks = c(0,250,500,750,1000)) + 
  scale_color_manual(values=c("grey68","darkorange2"))  + xlab("") +
  theme(legend.position = "bottom",panel.grid.minor = element_blank()) + 
  ggtitle("Transaction Amount Ranges by Category: Legit vs Fraud") + facet_wrap(~category, ncol = 4)

# (Grid) Fraud & Legit Breakdown per category by % Amount 
train_set %>% mutate(bins = cut(amt, breaks =  c(-Inf, 100, 250, 800, 1400, Inf), labels = c("<$100","$100-$249","$250-$799","$800-$1399","$1.4K+"))) %>% 
  group_by(is_fraud,category,bins) %>% summarize(amt = sum(amt)) %>% mutate(pct_amt = (amt/sum(amt))) %>%
  ggplot(aes(x = pct_amt,y = reorder(category,pct_amt), fill = as.factor(is_fraud))) +
  geom_col(position = position_dodge2(width = 0.9, preserve = "single")) + labs(x="Transaction Amount", y="") + 
  theme_bw(base_size = 10) + scale_x_continuous(labels = comma) + scale_fill_manual(values=c("grey68","darkorange2")) +
  theme(legend.position = "bottom",panel.grid.minor = element_blank()) + 
  ggtitle("Breakdown of Transaction Amounts by Category") + facet_grid(~bins)



### Transactions by Gender ###  

# fraud by gender
train_set %>% group_by(is_fraud, gender) %>% summarize(amt = sum(amt), n = n()) %>% 
  mutate(pct_amt = amt/sum(amt), pct_n = n/sum(n)) %>% filter(is_fraud ==1)


#### Transaction by Date of Birth ####
range(train_set$dob)

# Transactions legit vs fraud
train_set %>% 
  ggplot(aes(dob)) + geom_histogram(bins = 40, fill = "#56B4E9") + 
  theme_bw(base_size = 10) + theme(plot.title = element_text(size = 9), axis.title=element_blank()) + 
  ggtitle("Legit vs Fraud by Date of Birth: Legitimate Transaction vs Fraud") + facet_wrap(~is_fraud, scales = "free")



### Account Transaction Timeline Example  ### 

# looking at cc_num with fraud trans
train_set %>% filter(cc_num == 60416207185) %>% mutate(trans_date = as_date(trans_date_trans_time)) %>%
  ggplot(aes(trans_date_trans_time, amt, color = as.factor(is_fraud))) + geom_point() +
  labs(x="Transaction Date", y = "Trans Amount") + scale_color_manual(values = c("#CCCCCC", "#990000")) + theme_bw(base_size=10) +
  theme(legend.position = "bottom",  panel.grid.minor = element_blank()) + 
  ggtitle("Example of One CC Transactions 2019-2020: Legimate vs Fraud") 



### Repeat Fraud ### 

# Repeat fraud by CC
repeat_cc <- train_set %>% filter(is_fraud ==1) %>% group_by(cc_num) %>% 
  summarize(total = sum(amt), avg = mean(amt), n = n()) %>% 
  ggplot(aes(n)) + geom_histogram(bins = 20,fill = "56B4E9") + 
  theme_bw(base_size = 9) + theme(axis.title=element_blank()) + ggtitle("Number of Fraud Transactions on Same Credit Card")

# Repeat fraud by Merchant
repeat_merc <- train_set %>% filter(is_fraud ==1) %>% group_by(merchant) %>% 
  summarize(total = sum(amt), avg = mean(amt), n = n()) %>%
  ggplot(aes(n)) + geom_histogram(bins = 40,fill = "56B4E9") + 
  theme_bw(base_size = 9) + theme(axis.title=element_blank()) + ggtitle("Number of Fraud Transactions Same Merchant")

grid.arrange(repeat_cc, repeat_merc, ncol = 2)

rm(repeat_cc, repeat_merc)





### Transactions by Date/Time Elements ### 

## Fraud by Hour

# fraud transaction amt by hour
train_set %>% mutate(trans_hour = hour(trans_date_trans_time)) %>% group_by(trans_hour, is_fraud) %>%
  summarize(amt = sum(amt)) %>% mutate(pct_amt = (amt/sum(amt))*100) %>% filter(is_fraud == 1) %>%
  ggplot(aes(x= trans_hour)) + scale_x_continuous(name = "Transaction Hour", breaks=seq(0,23)) + ylim(0,35) + ylab("Percentage Transaction Amount") +
  geom_bar(aes(y=pct_amt, fill = pct_amt), stat = "identity") + geom_text(aes(y=pct_amt, label = sprintf("%1.1f%%", pct_amt)),size = 3, nudge_y = 1) +
  theme_bw(base_size = 10) +  theme(legend.position = "none",panel.grid.minor = element_blank()) + ggtitle("Breakdown Fraud Amounts by Hour of the Day: 72% of Fraud between 10pm & 3am")

# Legit vs fraud Amts Breakdown transaction hour
train_set %>% mutate(trans_hour = hour(trans_date_trans_time)) %>% group_by(is_fraud, trans_hour) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>%
  ggplot(aes(trans_hour, pct_amt, size = n, color = as.factor(is_fraud))) +
  geom_point(alpha = 1/3) + geom_text(aes(y=pct_amt, label = sprintf("%1.1f%%",pct_amt)),size = 3, nudge_y = .85) + 
  labs(x =  "Transaction Time: Hour", y = "") + ylim(0,35) + theme_bw(base_size = 10) + 
  scale_x_continuous(name = "Transaction Hour", breaks=seq(0,23)) + scale_color_manual(values = c("#999999", "#990000")) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank()) + ggtitle("Breakdown Fraud Amounts by Hour of Day: Legit vs Fraud")

# Legit vs fraud Counts Breakdown transaction hour
train_set %>% mutate(trans_hour = hour(trans_date_trans_time)) %>% group_by(is_fraud, trans_hour) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>%
  ggplot(aes(trans_hour, pct_n, size = n, color = as.factor(is_fraud))) +
  geom_point(alpha = 1/3) + geom_text(aes(y=pct_n, label=sprintf("%1.1f%%",pct_amt)),size = 3, nudge_y = .85) + 
  labs(x =  "Transaction Time: Hour", y = "") + ylim(0,30) +  theme_bw(base_size = 10) + 
  scale_x_continuous(name = "Transaction Hour", breaks=seq(0,23)) + scale_color_manual(values = c("#999999", "#990000")) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank()) + ggtitle("Breakdown Number of Transactions by Hour of Day: Legit vs Fraud")



## Fraud by Month

# Bar & point percent fraud by Month (Amounts / Number of Transactions)
train_set %>% mutate(trans_month = month(trans_date_trans_time, label = TRUE)) %>% group_by(trans_month, is_fraud,) %>%
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>% filter(is_fraud == 1) %>%
  ggplot(aes(x= trans_month)) +labs(x = "", y = "") + geom_bar(aes(y=pct_amt), stat = "identity", fill = "slategrey") + 
  geom_text(aes(y=pct_amt, label=sprintf("%1.1f%%",pct_amt)), size = 3, nudge_y = .5) +
  geom_point(aes(y=pct_n), shape = 18) + geom_text(aes(y=pct_n, label=sprintf("%1.1f%%",pct_n)), color = "grey", size = 3, nudge_y = .5) + 
  theme_bw(base_size = 10) + ylim(0,15) + ggtitle("Each Month's Percentage of Fraud Transactions: $ Amount (& Counts)")

# Breakdown Trans Amt by Month: Legit vs Fraud
train_set %>% mutate(trans_month = month(trans_date_trans_time, label = TRUE)) %>% group_by(is_fraud, trans_month) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100) %>% 
  ggplot(aes(trans_month, pct_amt, group = is_fraud, color = as.factor(is_fraud))) + 
  geom_line(size = .75) + geom_point() + geom_text(aes(label=sprintf("%1.1f%%",pct_amt)), size = 3, nudge_y = .75) + 
  labs(x="",y="Percent by Month") + ylim(0,20) +
  theme_bw(base_size = 10) +   scale_color_manual(values = c("#999999", "#990000")) + 
  theme(legend.position = "bottom", panel.grid.minor = element_blank()) + ggtitle("Breakdown Transactions by Month: Legit vs Fraud")



## Fraud by Day of the Month

# Bar & point percent fraud by Day of Month (Amounts / Number of Transactions)
train_set %>% mutate(day = day(trans_date_trans_time)) %>% group_by(day, is_fraud) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>% filter(is_fraud == 1) %>%
  ggplot(aes(x= day)) +labs(x = "", y = "") + geom_bar(aes(y=pct_amt), stat = "identity", fill = "slategrey") + 
  geom_text(aes(y=pct_amt, label=round(pct_amt, digits = 1)), size = 3, nudge_y = .5) +
  geom_point(aes(y=pct_n), shape = 18) + geom_text(aes(y=pct_n, label=round(pct_n,digits = 1)), color = "grey", size = 2, nudge_y = .5) + 
  scale_x_continuous(name = "Day of the Month", breaks=seq(1,31,1)) +  ylim(0,15) +
  theme_bw(base_size = 10) + theme(panel.grid.minor = element_blank()) + ggtitle("Each Day's Percent of Fraud Transactions: $ Amount (& Counts)")

# Breakdown Trans Amt by Day of Month: Legit vs Fraud 
train_set %>% mutate(day = day(trans_date_trans_time)) %>% group_by(is_fraud, day) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_n = (n/sum(n))*100) %>% 
  ggplot(aes(day, pct_n, color = as.factor(is_fraud), label = sprintf("%0.1f", round(pct_n, digits = 2)))) + 
  geom_line(size = 1) + geom_text(size = 3, nudge_y = .4) + ylab("Transactions: percent by day of Month") +
  theme_bw(base_size = 10) + scale_x_continuous(name = "Day of the Month", breaks=seq(1,31)) + ylim(1,15) +
  scale_color_manual(values = c("#999999", "#990000")) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank()) + ggtitle("Transactions by Day of Month: Legit vs Fraud")


## Fraud by Weekday

# Bar(amt) & point(n) percent fraud by Weekday
train_set %>% mutate(weekday = wday(trans_date_trans_time, label = TRUE)) %>% group_by(weekday, is_fraud) %>% 
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100)  %>% filter(is_fraud == 1) %>%
  ggplot(aes(weekday)) + ylim(0,20) + labs(x="", y="Percent Transactions") + 
  geom_bar(aes(y=pct_amt), stat = "identity", fill = "slategrey") + geom_text(aes(y=pct_amt, label=sprintf("%1.1f%%",pct_amt)), size = 3, nudge_y = .5) +
  geom_point(aes(y=pct_n), shape = 18) + geom_text(aes(y=pct_n, label=sprintf("%1.1f%%",pct_n)), color = "grey", size = 3, nudge_y = .5) + 
  theme_bw(base_size = 10) + ggtitle("Each Weekday's Percent of Fraud Transactions: $ Amount (& Counts)")

# Breakdown Trans Amt by Weekday: Legit vs Fraud
train_set %>% mutate(weekday = wday(trans_date_trans_time, label = TRUE)) %>% group_by(is_fraud, weekday) %>%
  summarize(amt = sum(amt), n = n()) %>% mutate(pct_amt = (amt/sum(amt))*100) %>% 
  ggplot(aes(weekday, pct_amt, group = is_fraud, color = as.factor(is_fraud))) + 
  geom_line(size = .75) + geom_point() + geom_text(aes(label=sprintf("%1.1f%%",pct_amt)), size = 3, nudge_y = .75) + 
  labs(x="",y="Percent by Weekday") + ylim(0,25) +
  theme_bw(base_size = 10) +   scale_color_manual(values = c("#999999", "#990000")) + 
  theme(legend.position = "bottom", panel.grid.minor = element_blank()) + ggtitle("Breakdown Transactions by Weekday: Legit vs Fraud")



### Location Elements ###

## by Distance to Merchant

# Calculate distance between cust & merchant with longitude and latitude . This takes several minutes. 
train_distance <- train_set[c(1:5,12:14,20:22)] %>% rowwise() %>% mutate(trans_dist = distHaversine(c(long, lat),c(merch_long, merch_lat))/ 1609)

range(train_distance$trans_dist)

train_distance %>% 
  ggplot(aes(trans_dist)) + geom_histogram(bins = 40, fill = "56B4E9") + labs(x="Miles between Customer / Merchant ", y="") +
  theme_bw(base_size = 10) + ggtitle("Transaction Distances") + facet_wrap(~is_fraud, scales = "free")


# saving calculated distances for use in .Rmd file
saveRDS(train_distance, file = "train_distance.rds")


#removing train_dist 
rm(train_distance)




### by State ###


# Breakdown/Percent Fraud Sales $s by State. Small pop states have large % of their sales as fraud.  
st_f <- train_set %>% group_by(is_fraud, state) %>% summarize(amt = sum(amt), n = n()) %>% 
  mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>% filter(is_fraud ==1) %>% 
  ggplot(aes(x=pct_amt, y = reorder(state, pct_amt), label = round(pct_amt, digits = 1))) + xlim(0,20) +
  geom_bar(stat = "identity", fill = "slategrey") + geom_text(size = 3, nudge_x = .8, alpha = 1/2) + 
  theme_bw(base_size = 9) + theme(axis.title.x=element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank()) + 
  ggtitle("Breakdown Percent Fraud $s by State")

# Breakdown/Percent Accounts by State
st_a <- train_set %>% group_by(state) %>% summarize(accts = n_distinct(cc_num), amt = sum(amt), n = n()) %>% 
  mutate(pct_accts = (accts/sum(accts))*100) %>%
  ggplot(aes(x=pct_accts, y = reorder(state, pct_accts), label = round(pct_accts, digits = 1))) + xlim(0,20) +
  geom_bar(stat = "identity", fill = "slategrey") + geom_text(size = 3, nudge_x = .8, alpha = 1/2) + 
  theme_bw(base_size = 9) + theme(axis.title.x=element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank()) + 
  ggtitle("Breakdown Accounts by State")

grid.arrange(st_f, st_a, ncol = 2)

rm(st_f, st_a)


# Checking Each State's fraud Percent
train_set %>% group_by(state, is_fraud) %>% summarize(amt = sum(amt), n = n()) %>% 
  mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>% filter(is_fraud ==1) %>% 
  ggplot(aes(pct_amt)) + geom_histogram() + 
  theme_bw(base_size = 10) + ggtitle("Each State's Percent Fraud $s ")  

# Percent Fraud Sales $s by State. Small pop states have large % of their sales as fraud.  
train_set %>% group_by(state, is_fraud) %>% summarize(amt = sum(amt), n = n()) %>% 
  mutate(pct_amt = (amt/sum(amt))*100, pct_n = (n/sum(n))*100) %>% filter(is_fraud ==1 & pct_amt > 5) %>%
  ggplot(aes(x=pct_amt,  y = reorder(state, pct_amt), label = round(pct_amt, digits = 1))) + 
  geom_bar(stat = "identity", fill = "slategrey") + geom_text(size = 3, nudge_x = 2) + labs(x = "", y = "") +
  theme_bw(base_size = 10) + ggtitle("States with Highest Percentage Fraud Amounts")


# Look at Delaware. Only has nine transactions, all fraud.
train_set %>% filter(state == "DE") %>% group_by(is_fraud) %>% summarize(amt = sum(amt), n = n()) 




#################################################################
###                   fraudSet Data Prep                      ###


# Creating prepped training set. Removing columns not used in modeling. Removing fraud from merchant. Added transaction amount bins
train_set <- train_set[c(1:5,22)] 


# Removing columns not used in modeling. Removing fraud from merchant. Added transaction amount bins capped at 100000 as if credit limit.
train_set <- train_set %>% mutate(merchant = str_remove(merchant, "fraud_"), cc_num = as.character(cc_num), is_fraud = as_factor(is_fraud),
                                  bins = cut(amt, breaks=c(0,99,249,799,1399,100000), labels=c("<$100","$100-$249","$250-$799","$800-$1400","$1400+")))

# & separating transaction date & time & converting day and trans_hour to factors. This runs much faster when saved to different object.
train_prep <- train_set %>%  mutate(month = month(trans_date_trans_time, label = TRUE), day = day(trans_date_trans_time), 
                                    trans_hour = hour(trans_date_trans_time), weekday = wday(trans_date_trans_time, label = TRUE)) %>% 
  mutate(day = as_factor(day), trans_hour = as_factor(trans_hour))


# Splitting data again at 90/10: for modeling training & testing. Leaving test_set for final evaluation of models.
set.seed(1, sample.kind="Rounding") 
model_index <- createDataPartition(y = train_prep$is_fraud, times = 1, p = 0.1, list = FALSE)
train2 <- train_prep[-model_index,]
test2 <- train_prep[model_index,]

table(test2$is_fraud)


#Removing initial full data set 
rm(train_set, train_prep)



### Prepping Final Validation test set 
test_set <- test_set[c(1:5,22)] 

test_prep <- test_set %>% mutate(merchant = str_remove(merchant, "fraud_"), cc_num = as.character(cc_num), is_fraud = as_factor(is_fraud),
                                 bins = cut(amt, breaks=c(0,99,249,799,1399,100000), labels=c("<$100","$100-$249","$250-$799","$800-$1400","$1400+"))) 

# removing test_set
rm(test_set)

test_set <- test_prep %>% mutate(month = month(trans_date_trans_time, label = TRUE), day = day(trans_date_trans_time),
                                 trans_hour = hour(trans_date_trans_time), weekday = wday(trans_date_trans_time, label = TRUE)) %>% 
  mutate(day = as_factor(day), trans_hour = as_factor(trans_hour))

# removing test_prep
rm(test_prep, model_index, test_index)




###############################################
##          Saving Prepped Files             ##


# save prepped training & test files to use in .Rmd.
saveRDS(train2, file = "train2.rds")
saveRDS(test2, file = "test2.rds")
saveRDS(test_set, file = "test_set.rds")





##############################################
##          Loading Generated Models        ##

# If you want to run the code, but not spend the time processing the models:
# This will download fit models and load them into your RStudio environment.

# create temp file & url 
dl <- tempfile()
URL <- "https://github.com/KHHogan06/CYO-Project/releases/download/v1-files/CYO_Models.RData"

# download url into temp file
download.file(URL, dl)

# load .RData into environment. This may take a few minutes.
load(dl)

# remove files
rm(dl, URL)




#################################################################
###                        Modeling                           ###



###  No Fraud Predicted  ### 
## Shows revenue loss when no fraud detection used

# Create vector predictions containing 0 for every transfer
predictions <- factor(rep(0, times = nrow(test2)), levels = c(0, 1))

# Compute cost of not detecting fraud
loss <- sum(test2$amt[test2$is_fraud == 1]) 
# $435,007.06

# Compute Accuracy
confusionMatrix(predictions, test2$is_fraud)$overall[["Accuracy"]] #.995
confusionMatrix(predictions, test2$is_fraud)$byClass["Balanced Accuracy"] #.5

# Tabling fraud predictions 
cost_preds <- tibble(amt = test2$amt, results = test2$is_fraud, no_preds = predictions)

# Tabling cost results
cost_results <- tibble(Model = "No Fraud Predicted", AmtSaved = 0, FraudMissed = loss, MisClassified = 0, SavedPct = 0, MisClassPct = 0, Specificity = 0, NPV = 0)

# removing large element
rm(predictions)




#########################################
###     CART Modeling with rpart      ###


## rpart model all predictors ##
# The rpart models take about 1 minute to run.

# all vars: except trans_date_trans_time,trans_date.  Date parts as factors.  
fit_rpart_all <- train2 %>% select(-c(trans_date_trans_time)) %>% 
  rpart(is_fraud ~ ., data = ., method = "class")

# rpart predictions
yhat_rpart_all <- predict(fit_rpart_all, test2, type = "class")

# plot decision tree rules. Merchant and cc_nums makes it un-readable.
plot(fit_rpart_all, margin = 0.1)
text(fit_rpart_all, cex = 0.5, pretty=1)


# Evaluating model performance
confusionMatrix(yhat_rpart_all, test2$is_fraud)$table
rp_all_Sp <- confusionMatrix(yhat_rpart_all, test2$is_fraud)$byClass["Specificity"]
rp_all_NPV <- confusionMatrix(yhat_rpart_all, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rpart_all)
rp_all_saved <- cost_preds %>% filter(results == 1 & yhat_rpart_all == 1) %>% summarize(sum(amt)) %>% pull()  
rp_all_miscl<- cost_preds %>% filter(results == 0 & yhat_rpart_all == 1) %>% summarize(sum(amt)) %>% pull()
rp_all_miss <- cost_preds %>% filter(results == 1 & yhat_rpart_all == 0) %>% summarize(sum(amt)) %>% pull()

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Rpart All Vars", AmtSaved = rp_all_saved, FraudMissed = rp_all_miss, MisClassified = rp_all_miscl, 
                                     SavedPct = rp_all_saved/loss, MisClassPct = rp_all_miscl/loss, Specificity = rp_all_Sp, NPV = rp_all_NPV))
cost_results %>% filter(Model == "Rpart All Vars")


# plot decision tree rules. Not readable.
plot(fit_rpart_all, margin = 0.1)
text(fit_rpart_all, cex = 0.5, pretty=1)

# variable importance. Merchant highest score. Category & bins same.
fit_rpart_all$variable.importance




##  Rpart Model 2  ##

# Basic model with date parts. No merchant, cc_nums, bins, trans_date\time.  
fit_rpart <- train2 %>% select(-c(trans_date_trans_time, cc_num, merchant, bins)) %>% 
  rpart(is_fraud ~ ., data = ., method = "class")

# rpart2 predictions
yhat_rpart <- predict(fit_rpart, test2, type = "class")

# plot decision tree rules. Almost readable.
plot(fit_rpart, margin = 0.25)
text(fit_rpart, cex = 0.5, pretty=1)


# Evaluating model 2 Performance
confusionMatrix(yhat_rpart, test2$is_fraud)$table

# Model 2 Metrics
rp_Sp <- confusionMatrix(yhat_rpart, test2$is_fraud)$byClass["Specificity"]
rp_NPV <- confusionMatrix(yhat_rpart, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rpart)
rp_saved <- cost_preds %>% filter(results == 1 & yhat_rpart == 1) %>% summarize(sum(amt)) %>% pull()  
rp_miscl<- cost_preds %>% filter(results == 0 & yhat_rpart == 1) %>% summarize(sum(amt)) %>% pull()  
rp_miss <- cost_preds %>% filter(results == 1 & yhat_rpart == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Rpart Basic", AmtSaved = rp_saved, FraudMissed = rp_miss, MisClassified = rp_miscl, 
                                     SavedPct = rp_saved/loss, MisClassPct = rp_miscl/loss, Specificity = rp_Sp, NPV = rp_NPV))
cost_results %>% filter(Model == "Rpart Basic")

# Plotting cp vs xerror
plotcp(fit_rpart, cex.lab = 0.8, cex.axis = 0.75)


##  Rpart Model 2: Tuning Complexity Parameter (CP) ##

# Basic rpart model with minsplit & cp = 0  
fit_rpartcp <- train2 %>% select(-c(trans_date_trans_time, cc_num, merchant, bins)) %>% 
  rpart(is_fraud ~ ., data = ., minsplit = 0, cp = 0, method = "class")

# Plotting cp/xerror
plotcp(fit_rpartcp)
printcp(fit_rpartcp, digits = 6)

# Two values. Choosing cp with fewer splits 
rp_cp <- cptable %>% filter(nsplit == 65) %>% pull(CP)

# Pruning basic rpart model   
pfit_rpartcp <- prune.rpart(fit_rpartcp, cp=rp_cp)

# rpart basic cp  predictions
yhat_rpartcp <- predict(pfit_rpartcp, test2, type = "class")

# Evaluating rpart with cp Performance
confusionMatrix(yhat_rpartcp, test2$is_fraud)$table


# Model 2 CP Metrics
rpcp_Sp <- confusionMatrix(yhat_rpartcp, test2$is_fraud)$byClass["Specificity"]
rpcp_NPV <- confusionMatrix(yhat_rpartcp, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rpartcp)
rpcp_saved <- cost_preds %>% filter(results == 1 & yhat_rpartcp == 1) %>% summarize(sum(amt)) %>% pull()  
rpcp_miscl<- cost_preds %>% filter(results == 0 & yhat_rpartcp == 1) %>% summarize(sum(amt)) %>% pull()  
rpcp_miss <- cost_preds %>% filter(results == 1 & yhat_rpartcp == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving Results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Rpart Basic Tuned", AmtSaved = rpcp_saved, FraudMissed = rpcp_miss, MisClassified = rpcp_miscl, 
                                     SavedPct = rpcp_saved/loss, MisClassPct = rpcp_miscl/loss, Specificity = rpcp_Sp, NPV = rpcp_NPV))
cost_results %>% filter(Model == c("Rpart Basic","Rpart Basic Tuned"))





##  Rpart Model 3  ##

# verifying cc_num as predictor. no merchant, bins, trans_date\time
fit_rpart_cc <- train2 %>% select(-c(trans_date_trans_time, merchant, bins)) %>% rpart(is_fraud ~ ., data = ., method = "class")

# model 3 predictions
yhat_rpart_cc <- predict(fit_rpart_cc, test2, type = "class")

# Evaluating model 3 performance
confusionMatrix(yhat_rpart_cc, test2$is_fraud)$table
rp_cc_Sp <- confusionMatrix(yhat_rpart_cc, test2$is_fraud)$byClass["Specificity"]
rp_cc_NPV <- confusionMatrix(yhat_rpart_cc, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rpart3)
rp_cc_saved <- cost_preds %>% filter(results == 1 & yhat_rpart_cc == 1) %>% summarize(sum(amt)) %>% pull()  
rp_cc_miscl<- cost_preds %>% filter(results == 0 &  yhat_rpart_cc == 1) %>% summarize(sum(amt)) %>% pull()  
rp_cc_miss <- cost_preds %>% filter(results == 1 & yhat_rpart_cc == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Rpart CCs", AmtSaved = rp_cc_saved, FraudMissed = rp_cc_miss, MisClassified = rp_cc_miscl, 
                                     SavedPct = rp_cc_saved/loss, MisClassPct = rp_cc_miscl/loss, Specificity = rp_cc_Sp, NPV = rp_cc_NPV))
cost_results %>% filter(Model == "Rpart CCs")

# Plotting cp vs xerror
plotcp(fit_rpart_cc, cex.lab = 0.8, cex.axis = 0.75)


## Rpart Model 3: CP Tuning ##

# verifying cc_num as predictor. no merchant, bins, trans_date\time
fit_rpartcp_cc <- train2 %>% select(-c(trans_date_trans_time, merchant, bins)) %>%
  rpart(is_fraud ~ ., data = ., minsplit =0, cp = 0, method = "class")

# Plotting cp vs xerror
plotcp(fit_rpartcp_cc, cex.lab = 0.8, cex.axis = 0.75)
rpcp_cc <-fit_rpartcp_cc$cptable[which.min(fit_rpartcp_cc$cptable[,"xerror"]),"CP"]

# Pruning basic rpart model   
pfit_rpartcp_cc <- prune.rpart(fit_rpartcp_cc, cp=rpcp_cc)

# rpart basic cp  predictions
yhat_rpartcp_cc <- predict(pfit_rpartcp_cc, test2, type = "class")

# Evaluating rpart with cp Performance
confusionMatrix(yhat_rpartcp_cc, test2$is_fraud)$table


# Model 3 Tuned CP Metrics
rpcp_cc_Sp <- confusionMatrix(yhat_rpartcp_cc, test2$is_fraud)$byClass["Specificity"]
rpcp_cc_NPV <- confusionMatrix(yhat_rpartcp_cc, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rpartcp_cc)
rpcp_cc_saved <- cost_preds %>% filter(results == 1 & yhat_rpartcp_cc == 1) %>% summarize(sum(amt)) %>% pull()  
rpcp_cc_miscl<- cost_preds %>% filter(results == 0 & yhat_rpartcp_cc == 1) %>% summarize(sum(amt)) %>% pull()  
rpcp_cc_miss <- cost_preds %>% filter(results == 1 & yhat_rpartcp_cc == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Rpart CCs Tuned", AmtSaved = rpcp_cc_saved, FraudMissed = rpcp_cc_miss, MisClassified = rpcp_cc_miscl, 
                                     SavedPct = rpcp_cc_saved/loss, MisClassPct = rpcp_cc_miscl/loss, Specificity = rpcp_cc_Sp, NPV = rpcp_cc_NPV))
cost_results %>% filter(Model == c("Rpart CCs","Rpart CCs Tuned"))





#########################################
###     Final Validation Set Up       ###

# Compute cost of not detecting fraud
cost <- sum(test_set$amt[test_set$is_fraud == 1]) 
# $996,490.68

# tabling results & amounts
final_preds <- tibble(amt = test_set$amt, results = test_set$is_fraud)

# Tabulating final cost results
final_results <- tibble(Model = "No Fraud Predicted", AmtSaved = 0, FraudMissed = cost, MisClassified = 0, 
                        SavedPct = 0, MisClassPct = 0, Specificity = 0, NPV = 0)



#########################################
###   Final Validation: Basic rpart   ###


# rpart predictions
yhat_rpartcp_cc <- predict(pfit_rpartcp_cc, test_set, type = "class")

# Evaluating model performance
cm_rpart <- as.data.frame.matrix(confusionMatrix(yhat_rpartcp_cc, test_set$is_fraud)$table)
rp_Sp <- confusionMatrix(yhat_rpartcp_cc, test_set$is_fraud)$byClass["Specificity"]
rp_NPV <- confusionMatrix(yhat_rpartcp_cc, test_set$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
final_preds <- cbind(final_preds, yhat_rpartcp_cc)
rp_saved <- final_preds %>% filter(results == 1 & yhat_rpartcp_cc == 1) %>% summarize(sum(amt)) %>% pull()  
rp_miscl<- final_preds %>% filter(results == 0 & yhat_rpartcp_cc == 1) %>% summarize(sum(amt)) %>% pull()
rp_miss <- final_preds %>% filter(results == 1 & yhat_rpartcp_cc == 0) %>% summarize(sum(amt)) %>% pull()

# Saving results
final_results <- bind_rows(final_results,
                           data_frame(Model = "Tuned Rpart CCs", AmtSaved = rp_saved, FraudMissed = rp_miss, MisClassified = rp_miscl, 
                                      SavedPct = rp_saved/cost, MisClassPct = rp_miscl/cost, Specificity = rp_Sp, NPV = rp_NPV))


# removing rpart variables
rm(rp_all_NPV, rp_all_Sp, rp_all_saved, rp_all_miscl, rp_all_miss, 
   rp_NPV, rp_Sp, rp_saved, rp_miscl, rp_miss, 
   rpcp_NPV, rpcp_Sp, rpcp_saved, rpcp_miscl, rpcp_miss, rpcp_savedpct, rp_cp,
   rp_cc_NPV, rp_cc_Sp, rp_cc_saved, rp_cc_miscl, rp_cc_miss,  
   rpcp_cc_NPV, rpcp_cc_Sp, rpcp_cc_saved, rpcp_cc_miscl, rpcp_cc_miss, rpcp_cc,
   cm_rp, cm_rp, cm_rpcc, cmrps, cmrps2, cmrps3,
   yhat_rpart_all, yhat_rpart, yhat_rpartcp, yhat_rpart_cc, yhat_rpartcp_cc)

# Removing rpart models to clear space
rm(fit_rpart_all, fit_rpart, fit_rpartcp, pfit_rpartcp, fit_rpart_cc, fit_rpartcp_cc, pfit_rpartcp_cc)






#########################################
###      Logistic Regression          ###


## GLM model 1 ##

# category & amt interaction, no bins, date parts as factors. (This took about 5 minutes.)
## Can skip line below if loaded models from github 
fit_glm1 <- train2 %>% glm(is_fraud ~ category * amt + trans_hour + day + month + weekday, data=., family = "binomial")


# glm predictions
phat_glm1 <- predict.glm(fit_glm1, test2, type = "response")
yhat_glm1 <- ifelse(phat_glm1 > .5, 1, 0) %>% factor()

varImp(fit_glm1)


## GLM model 2 ##

# category & amt interaction, no bins, date parts as factors.(~ 5 mins)
## Can skip line below if loaded models from github 
fit_glm2 <- train2 %>% glm(is_fraud ~ category + amt + bins + trans_hour + day + month + weekday, data=., family = "binomial")


# glm predictions
phat_glm2 <- predict.glm(fit_glm2, test2, type = "response")
yhat_glm2 <- ifelse(phat_glm2 > .5, 1, 0) %>% factor()


# Confusion Matrix Model 1 & 2
confusionMatrix(yhat_glm1, test2$is_fraud)$table
confusionMatrix(yhat_glm2, test2$is_fraud)$table


# Evaluating models 1 & 2 performance
glm1_Sp <- confusionMatrix(yhat_glm1, test2$is_fraud)$byClass["Specificity"]
glm1_NPV <- confusionMatrix(yhat_glm1, test2$is_fraud)$byClass["Neg Pred Value"]
glm2_Sp <- confusionMatrix(yhat_glm2, test2$is_fraud)$byClass["Specificity"]
glm2_NPV <- confusionMatrix(yhat_glm2, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model 1 Costs
cost_preds <- cbind(cost_preds, yhat_glm1, yhat_glm2)
glm1_saved <- cost_preds %>% filter(results == 1 & yhat_glm1 == 1) %>% summarize(sum(amt)) %>% pull()  
glm1_miscl<- cost_preds %>% filter(results == 0 & yhat_glm1 == 1) %>% summarize(sum(amt)) %>% pull()  
glm1_miss <- cost_preds %>% filter(results == 1 & yhat_glm1 == 0) %>% summarize(sum(amt)) %>% pull()  
# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Glm cat*amt", AmtSaved = glm1_saved, FraudMissed = glm1_miss, MisClassified = glm1_miscl, 
                                     SavedPct = glm1_saved/loss, MisClassPct = glm1_miscl/loss, Specificity = glm1_Sp, NPV = glm1_NPV))

# Calculating Model2 Costs
glm2_saved <- cost_preds %>% filter(results == 1 & yhat_glm2 == 1) %>% summarize(sum(amt)) %>% pull()  
glm2_miscl<- cost_preds %>% filter(results == 0 & yhat_glm2 == 1) %>% summarize(sum(amt)) %>% pull()  
glm2_miss <- cost_preds %>% filter(results == 1 & yhat_glm2 == 0) %>% summarize(sum(amt)) %>% pull()  
# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Glm bins no int", AmtSaved = glm2_saved, FraudMissed = glm2_miss, MisClassified = glm2_miscl, 
                                     SavedPct = glm2_saved/loss, MisClassPct = glm2_miscl/loss, Specificity = glm2_Sp, NPV = glm2_NPV))

# Print Cost Results Model 1 & 2
cost_results %>% filter(Model == c("GLM cat*amt","GLM bins no int"))



###################################
##    Saving glm models 1& 2     ##

# Can skip this if you have downloaded models from github.

## Glm models can be huge because they store the data set along with a lot of other objects not always needed.
# Prepping glm model to save by nullifying unneeded large objects from model. 
# reference: nzumel, (May 2014). Trimming the Fat from glm Models in R. win-vector.com

fit_glm1$data <- NULL
fit_glm1$qr$qr <- NULL
fit_glm2$data <- NULL
fit_glm2$qr$qr <- NULL

# saving glm models for .Rmd
saveRDS(fit_glm1, file = "fit_glm1.rds")
saveRDS(fit_glm2, file = "fit_glm2.rds")




# Removing stored/large elements
rm(glm1_NPV, glm1_Sp, glm1_saved, glm1_miscl, glm1_miss, 
   glm2_NPV, glm2_Sp, glm2_saved, glm2_miscl, glm2_miss, 
   phat_glm1, yhat_glm1, phat_glm2, yhat_glm2, fit_glm1, fit_glm2)




## GLM model 3 ##

# Model with Category/amt/bins interaction, all date parts. Takes about 25 mins.
## Can skip line below if loaded models from github 
fit_glm_bins <- train2 %>% glm(is_fraud ~ category * amt * bins + trans_hour + day + month + weekday, data=., family = "binomial",
                               model = FALSE, y = FALSE)


# glm3 estimates
phat_glm_bins <- predict.glm(fit_glm_bins, test2, type = "response")

# glm3 predictions at .5
yhat_glm_bins.5 <- ifelse(phat_glm_bins > .5, 1, 0) %>% factor()
# glm3 predictions at .5
yhat_glm_bins.4 <- ifelse(phat_glm_bins > .4, 1, 0) %>% factor()


# Confusion Matrices
cm_glmb.5 <- as.data.frame.matrix(confusionMatrix(yhat_glm_bins.5, test2$is_fraud)$table)
cm_glmb.4 <- as.data.frame.matrix(confusionMatrix(yhat_glm_bins.4, test2$is_fraud)$table)

# combining confusion matrices
cmglms2 <-cbind(cm_glmb.5, cm_glmb.4)
cmglms2

# Evaluating model 3 performance
glm.5_Sp <- confusionMatrix(yhat_glm_bins.5, test2$is_fraud)$byClass["Specificity"]
glm.5_NPV <- confusionMatrix(yhat_glm_bins.5, test2$is_fraud)$byClass["Neg Pred Value"]
glm.4_Sp <- confusionMatrix(yhat_glm_bins.4, test2$is_fraud)$byClass["Specificity"]
glm.4_NPV <- confusionMatrix(yhat_glm_bins.4, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model 3 >.5 Costs
cost_preds <- cbind(cost_preds, yhat_glm_bins.5, yhat_glm_bins.4)
glm.5_saved <- cost_preds %>% filter(results == 1 & yhat_glm_bins.5 == 1) %>% summarize(sum(amt)) %>% pull()  
glm.5_miscl <- cost_preds %>% filter(results == 0 & yhat_glm_bins.5 == 1) %>% summarize(sum(amt)) %>% pull()  
glm.5_miss <- cost_preds %>% filter(results == 1 & yhat_glm_bins.5 == 0) %>% summarize(sum(amt)) %>% pull() 

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Glm bins >.5", AmtSaved = glm.5_saved, FraudMissed = glm.5_miss, MisClassified = glm.5_miscl, 
                                     SavedPct = glm.5_saved/loss, MisClassPct = glm.5_miscl/loss, Specificity = glm.5_Sp, NPV = glm.5_NPV))

# Calculating Model 3 >.4 Costs
glm.4_saved <- cost_preds %>% filter(results == 1 & yhat_glm_bins.4 == 1) %>% summarize(sum(amt)) %>% pull()  
glm.4_miscl <- cost_preds %>% filter(results == 0 & yhat_glm_bins.4 == 1) %>% summarize(sum(amt)) %>% pull()  
glm.4_miss <- cost_preds %>% filter(results == 1 & yhat_glm_bins.4 == 0) %>% summarize(sum(amt)) %>% pull() 

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "Glm bins >.4", AmtSaved = glm.4_saved, FraudMissed = glm.4_miss, MisClassified = glm.4_miscl, 
                                     SavedPct = glm.4_saved/loss, MisClassPct = glm.4_miscl/loss, Specificity = glm.4_Sp, NPV = glm.4_NPV))

# Print Cost Results Model 3
cost_results %>% filter(Model == c("Glm bins >.5","Glm bins >.4"))


# Removing large elements
rm(phat_glm_bins, yhat_glm_at.4, yhat_glm_at.5)



##################################
##      Saving glm model 3      ##

# Can skip this if you have downloaded models from github.

# Prepping glm model for saving by removing unneeded large objects.  
fit_glm_bins$data <- NULL
fit_glm_bins$qr$qr <- NULL

# saving glm models for .Rmd
saveRDS(fit_glm_bins, file = "fit_glm_bins.rds")





#########################################
###   Final Validation: GLM Model 3   ###

# Predictions
phat_glm_bins <- predict.glm(fit_glm_bins, test_set, type = "response")
yhat_glm_bins.45 <- ifelse(phat_glm_bins > .45, 1, 0) %>% factor()

# Evaluating model performance
cm_glm <- as.data.frame.matrix(confusionMatrix(yhat_glm_bins.45, test_set$is_fraud)$table)
glm_bins_Sp <- confusionMatrix(yhat_glm_bins.45, test_set$is_fraud)$byClass["Specificity"]
glm_bins_NPV <- confusionMatrix(yhat_glm_bins.45, test_set$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
final_preds <- cbind(final_preds, yhat_glm_bins.45)
glm_bins_saved <- final_preds %>% filter(results == 1 & yhat_glm_bins.45 == 1) %>% summarize(sum(amt)) %>% pull()  
glm_bins_miscl <- final_preds %>% filter(results == 0 & yhat_glm_bins.45 == 1) %>% summarize(sum(amt)) %>% pull()  
glm_bins_miss <- final_preds %>% filter(results == 1 & yhat_glm_bins.45 == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
final_results <- bind_rows(final_results,
                           data_frame(Model = "Glm bins >.45", AmtSaved = glm_bins_saved, FraudMissed = glm_bins_miss, MisClassified = glm_bins_miscl, 
                                      SavedPct = glm_bins_saved/cost, MisClassPct = glm_bins_miscl/cost, Specificity = glm_bins_Sp, NPV = glm_bins_NPV))


# removing glm saved values
rm(glm.5_NPV, glm.5_Sp, glm.5_saved, glm.5_miscl, glm.5_miss, glm.5_savedpct, glm.5_misclpct,
   glm.4_NPV, glm.4_Sp, glm.4_saved, glm.4_miscl, glm.4_miss, glm.4_savedpct, glm.4_misclpct,
   cm_glmb.4, cm_glmb.5, cmglms2,
   glm_bins_NPV, glm_bins_Sp, glm_bins_saved, glm_bins_miscl, glm_bins_miss, glm_bins_savedpct, glm_bins_misclpct,)

# removing large elements
rm(fit_glm, fit_glm_bins, yhat_glm_bins.45, phat_glm_bins)





#########################################
###          Random Forest            ###


## Random Forest Model 1 ##

# Resetting seed
set.seed(1, sample.kind="Rounding")

# 51 trees, default predictors  (Don't have enough memory for default 500 tress.) Takes about 5 mins.
## Can skip line below if loaded models from github 
fit_rf51 <- train2 %>% 
  randomForest(is_fraud ~., data = .,
               ntree = 51, 
               replacement = TRUE,
               importance = TRUE)


# Making predictions
yhat_rf51 <- predict(fit_rf51, test2, type = "class")

# Evaluating model performance
confusionMatrix(yhat_rf51, test2$is_fraud)$table
rf51_Sp <- confusionMatrix(yhat_rf51, test2$is_fraud)$byClass["Specificity"]
rf51_NPV <- confusionMatrix(yhat_rf51, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rf51)
rf51_saved <- cost_preds %>% filter(results == 1 & yhat_rf51 == 1) %>% summarize(sum(amt)) %>% pull()  
rf51_miscl<- cost_preds %>% filter(results == 0 & yhat_rf51 == 1) %>% summarize(sum(amt)) %>% pull()  
rf51_miss <- cost_preds %>% filter(results == 1 & yhat_rf51 == 0) %>% summarize(sum(amt)) %>% pull()   

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "RandomForest 51", AmtSaved = rf51_saved, FraudMissed = rf51_miss, MisClassified = rf51_miscl, 
                                     SavedPct = rf51_saved/loss, MisClassPct = rf51_miscl/loss, Specificity = rf51_Sp, NPV = rf51_NPV))
cost_results %>% filter(Model == "RandomForest 51")


# plotting trees / error
plot(fit_rf51)

# min OOB & fraud class Error at mtry 3
oobData <- as.data.table(plot(fit_rf51))
min(oobData$OOB)
min(oobData$'1')


# variable importance
fit_rf51$importance


# Removing large element
rm(yhat_rf51)

# saving rf model for .Rmd
saveRDS(fit_rf51, file = "fit_rf51.rds")


## Tuning RF Model for best mtry value ##

# Tuning rf for mtry 4:6, 75 trees.  Takes about 18 mins.  
## Can skip line below if loaded models from github 
rf_tune <-  tuneRF(train2[-6], train2$is_fraud, mtryStart = 5, ntreeTry = 75, stepFactor = .9,
                   trace=TRUE, plot=TRUE)

options(digits = 6)
rf_tune

# saving rf_tune for .Rmd 
# Uploaded to repo, can download & use instead)
saveRDS(rf_tune, file = "rf_tune.rds")





## Random Forest Model 2 ##

# 251 trees, 4/10 predictors. This takes about 25 mins & a lot of memory.
## Can skip line below if loaded models from github.
## Note: RandomForest results will differ slightly if you run your own model.
fit_rf251 <- train2 %>% 
  randomForest(is_fraud ~., data = .,
               ntree = 251, mtry = 4,
               replacement = TRUE,
               importance = TRUE)

# predictions
yhat_rf251_ <- predict(fit_rf251, test2, type = "class")

# Evaluating model performance
confusionMatrix(yhat_rf251, test2$is_fraud)$table
rf251_Sp <- confusionMatrix(yhat_rf251 , test2$is_fraud)$byClass["Specificity"]
rf251_NPV <- confusionMatrix(yhat_rf251, test2$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
cost_preds <- cbind(cost_preds, yhat_rf251)
rf251_saved <- cost_preds %>% filter(results == 1 & yhat_rf251 == 1) %>% summarize(sum(amt)) %>% pull()  
rf251_miscl<- cost_preds %>% filter(results == 0 & yhat_rf251 == 1) %>% summarize(sum(amt)) %>% pull()  
rf251_miss <- cost_preds %>% filter(results == 1 & yhat_rf251 == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
cost_results <- bind_rows(cost_results,
                          data_frame(Model = "RandomForest 251", AmtSaved = rrf251_saved, FraudMissed = rf251_miss, MisClassified = rf251_miscl, 
                                     SavedPct = rf251_saved/loss, MisClassPct = rf251_miscl/loss, Specificity = rf251_Sp, NPV = rf251_NPV))
cost_results %>% filter(Model == "RandomForest 251")

# plot the model
plot(fit_rf251)


# saving rf models for .Rmd
saveRDS(fit_rf251, file = "fit_rf251.rds")


confusionMatrix(yhat_rf251, test_set$is_fraud)$table


#########################################
###   Final Validation: RF Model  2   ###


# predictions
yhat_rf251 <- predict(fit_rf251, test_set, type = "class")

# Evaluating model performance
cm_rf <- as.data.frame.matrix(confusionMatrix(yhat_rf251, test_set$is_fraud)$table)
rf251_Sp <- confusionMatrix(yhat_rf251 , test_set$is_fraud)$byClass["Specificity"]
rf251_NPV <- confusionMatrix(yhat_rf251, test_set$is_fraud)$byClass["Neg Pred Value"]

# Calculating Model Costs
final_preds <- cbind(final_preds, yhat_rf251)
rf251_saved <- final_preds %>% filter(results == 1 & yhat_rf251 == 1) %>% summarize(sum(amt)) %>% pull()  
rf251_miscl<- final_preds %>% filter(results == 0 & yhat_rf251 == 1) %>% summarize(sum(amt)) %>% pull()  
rf251_miss <- final_preds %>% filter(results == 1 & yhat_rf251 == 0) %>% summarize(sum(amt)) %>% pull()  

# Saving results
final_results <- bind_rows(final_results,
                           data_frame(Model = "RandomForest 251", AmtSaved = rf251_saved, FraudMissed = rf251_miss, MisClassified = rf251_miscl, 
                                      SavedPct = rf251_saved/cost, MisClassPct = rf251_miscl/cost, Specificity = rf251_Sp, NPV = rf251_NPV))


# combining confusion matrices
cms <- cbind(cm_rpart, cm_glm, cm_rf)


# Final Confusion Matrices
cms

# Final Cost Results Compared
final_results



