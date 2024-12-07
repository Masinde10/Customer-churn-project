# Predicting Customer Churn For SyriaTel Company

**Author**: [Masinde Victor](https://github.c om/Masinde10)

![Telecommunication](http://localhost:8888/view/pict--telecommunication.png)

## Project Overview
This project deals with a company, SyriaTel, that wants to know more about their customer churn. Customers leave a company due to different reasons and my project aims to uncover the reasons and predict customer churn. The company can then use the information gained from this project to work on retaining their customers. Retaining customers is cheaper and easier as compared to gaining a new customer. The project utilizes the SyriaTel Telecoms dataset to create a classification model that predicts if a customer will churn or not. This is a problem of binary classification. 

## Business understanding 
Communication is key in our daily lives as it is what keeps the world running. We have several comminucation network providers that help make it easy for us to interact with people. With innovation and technology that keeps improving by day, just like any other sector, the telecommunication industry is affected by this. Competition gets tight as different companies try to intergrate the latest models into their systems for better customer satistfaction and better Return on Investments(ROI) on their end. One effect of these improvements is churning(customers leaving) as they propagate towards where they can get better services. I aim at creating a predictive business model which would enable Syriatel detect churn and identify reasons for the same. It will also help the company to  adopt strategies that would reduce churn, and maintain and grow its customer base, in a bid to sustain overall growth and profitability.
**Stakeholders**
1. The Syriatel Telecommunications board
2. The customers of the company 
3. The employees of the company since company making profits can also impact them.

**Research objectives**
1. To identify the key features that determine if a customer is likely to churn.
2. To determine the most suitable model to predict Customer Churn. 
3. To establish customer retention strategy to reduce churn.

## Data understanding
This dataset was sourced from kaggle and it has 3333 rows and 21 columns. The dataset has data recorded in different data type including float, intergers and objects. The columns are properly named showing what happens in the communication sector. This dataset includes details about a telecom company's customers, such as their state, phone number, area code, account length, and whether or not they have voice mail and international plans. Out of the 21 columns, 4 of them are in object form which means that they are categorical columns. Since we are dealing with models, they will later be transformed to numerics using encoding methods.

## Data Preparation
In this section we combine both `data cleaning` and `Exploratory Data Analysis` since they all do the same work, prepare data to be modelled. We dont want missing values in our final dataset sos we start with checking for that. we found that there were no missing values. We also check for duplicates and we found no duplicates. Dropping of unnecessary columns is also done at this stage. one of the columns was recorded irregularly and we correct that gere. In the phone numbercolumn, the values were recorded with a hyphen and they were of the type `object`. We transform this to numeric form and make the colimn to be our index as it a unique identifier to the customer. 

After we solve the few issues that were present,we inspect how our data looks like before we fit to a model. We do this by visualizing the the different excisting features. These can guide us in knowing what model to choose or the features that are going to be of huge importance to our models.After a proper visualization of the data in our dataset, we can now do modelling. 

## Modelling
For my project, I am using three different models namely, logistic regression, Decision tree classifier and Random Forest classifier. For logistic regression, I have baseline model and one that is tuned.

#### 1. Baseline Logistic Model
Logistic regression is among the models that can be used when dealing with classiication problems. It requires that our target variable bne in form of classes. For our case, we have a `target` variable with `two classes` commonly reffered to as `Binary`. This model if fitted with the data before any manipulation apart from splitting the data for train and test is done. We fit the model to scaled data of target(y) and feature variables(X). We can make predictions using the model and compare it with actual values. This gives us the perfomance metrics of the model.

#### 2. Tuned Logistic Model
As seen in our Exploratory data analysis, there is an imbalance in our target class. The class `not churned`(0) is bigger than the `churned` class(1). This can lead to overfitting of the models and as a result predict one class very well but fail to predict the other. To adress this we intorduce SMOTE to help us in balancing the target classes. It does that by duplicating the minority class until they are even with the majority class. We then pass the balanced X and ys to our logistic model and make predictions which will in tunr help us calculate the perfomance metrics of the model.

![Before SMOTE](http://localhost:8888/view/Before_SMOTE.png)
![After SMOTE](http://localhost:8888/view/After_SMOTE.png)

#### 3.Decision Tree Classifier
Decision tree classifiers are sophisticated machine learning models that use tree-like structures to categorise data. They work by recursively splitting the feature space into smaller subsets based on the most informative attributes. Starting from the root node, decision trees assess conditions at each internal node before proceeding along the appropriate branch. This process continues until it reaches a leaf node, which gives the final prediction or label. Decision trees can handle category and numerical variables, making them useful tools for a variety of classification applications. For this model, we will make use of the data that have their classes balanced by SMOTE. We can also get the metrics of the model as we did above.

#### 4.Random Forest Classifier
Random forests Classifier models can handle high-dimensional datasets and are especially good at dealing with noise and outliers. The randomisation introduced during tree construction helps to reduce overfitting, so random forests are less likely to memorise training data patterns. Despite their complexity, random forests are interpretable due to the intrinsic simplicity of individual decision trees in the ensemble. For this model, we will make use of the data that have their classes balanced by SMOTE. We can also get the metrics of the model as we did in the other models above.

![Tree Diagram](http://localhost:8888/view/first_tree.png)

## Evaluation of Models
Now that we have four models, we need to settle on one that will be used by the SyriaTel communications company to predict customer churn with confidence. All the models have their metrics calculated and we can use them to compare the models. The metrics include; **Accuracy, F1 score, recall, precission, AUC and ROC curves**

In terms of `accuracy`, the model that is perfoming the best is `Decision Tree Classifier` then closely followed by Random Forest Classifier. The worst perfoming is the tuned logistic regression model.

![Accuracy](http://localhost:8888/view/Accuracy.png)

In terms of `F1 score`, the model that is perfoming the best is `Random Forest Classifier` then closely followed by Decision Tree Classifier.The worst perfoming model is the logistic regression baseline model.

![F1 Score](http://localhost:8888/view/F1_scores.png)

Evaluating the visualizations made, we see that the `AUC` of the `Random Forest Classifier` is higher than all the other models. This an indicator of a good model. We also check at the positions of our graph, in that the curve that is closer to the `top left corner` is the one with the `best perfomance`. The curve closer to the top left corner is that of the `Random Forest Classifier.`

![ROC Curve](http://localhost:8888/view/Roc_curves.png)

## Conclusion, 
the project above shows that we can accurately predict customer churn on a significant level using a machine learning model.The comparisons indicate that the Random Forest Classifier Model is the most appropriate in our case being the one with good perfomance metrics overally. Random Forest Classifier is the best performing model with an ROC curve that is near the upper left corner of the graph, hence giving us the largest AUC (Area Under the curve).

## Recommendations
 
I would recommend that Syriatel make use of the Random Forest Classifier as the primary model for predicting customer churn. This model has a higher ROC curve and strong overall performance in terms of accuracy, F1-score, recall, and precision on the test set, making it well-suited for accurately classifying customers as likely or unlikely to churn.

I would recommend that the company focuses on tuning the call minutes and charges to best fit the customers. These efforts could include personalized offers or discounts on day charges.Customers could be given more talk time with lower charges. By implementing cost-effective strategies that address the key factors driving customer churn, SyriaTel can retain customers and minimize revenue loss.

I would recommend, that Syriatel comes up with strategies to reduce on Customer Service calls. The more the customer service calls, the higher the likelihood of churn as this can easily irritate the customer. If the calls are necessary, then the company should come up with a way of educating thneir customers on the importance of the same.


