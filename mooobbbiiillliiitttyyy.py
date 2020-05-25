# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:01:09 2020

@author: HP
"""


"""
Created on Sun May 17 00:59:07 2020

@author: HP
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import itertools

warnings.filterwarnings("ignore")

os.chdir(r"D:\python\decision tree")

def plotConfusionMatrix(cm, n,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if n == 2:
        classes = [0,1]
    elif n == 3:
        classes = [0,1,2]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [0,1.2])
    plt.yticks(tick_marks, [0,1,2])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore")

os.chdir(r"D:\janataaatat\New folder")


test_data = pd.read_csv('test_VsU9xXK.csv')
train_data = pd.read_csv('train_Wc8LBpr.csv')


train_data1=train_data.copy() # backup
test_data1=test_data.copy() # backup

train_data1.isnull().sum()


data = pd.concat([train_data1, test_data1], axis = 0)

"droping irrelevent variable"
data.drop(['Trip_ID','Var1'],1,inplace=True)

####trying to plot the target variable
train_data1['Surge_Pricing_Type'].value_counts().plot.bar()

#trying to deal with missing values
#feature imputed with mode only if categorical data present

data['Type_of_Cab']=data['Type_of_Cab'].fillna('B')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Type_of_Cab']=le.fit_transform(data['Type_of_Cab'])

data['Customer_Since_Months']=data['Customer_Since_Months'].fillna('10').astype('int64')

#since Life_Style_Index is numerical we are imputing it with mean
data['Life_Style_Index'].fillna((data['Life_Style_Index'].mean()), inplace=True)

#since Confidence_Life_Style_Index is categorical and na is large so treating them as another category
data['Confidence_Life_Style_Index']=data['Confidence_Life_Style_Index'].fillna('D')

#since Var1 is numerical and imputing them with mode will be biased 
#so we are imputing it with mean value
# data['Var1']=data['Var1'].fillna(data['Var1'].fillna('64')).astype('int64')

data['Gender']=data['Gender'].map({'Male':1,'Female':0})

# from sklearn.preprocessing import minmax_scale
# train_data1[['Trip_Distance','Life_Style_Index','Customer_Rating','Var1','Var2','Var3']] = minmax_scale(train_data1[['Trip_Distance','Life_Style_Index','Customer_Rating','Var1','Var2','Var3']])

#####label encoding some ordinal categorical data
data['Confidence_Life_Style_Index']=le.fit_transform(data['Confidence_Life_Style_Index'])
data['Destination_Type']=le.fit_transform(data['Destination_Type'])


data['Trip_Distance_bin'] = pd.cut(data['Trip_Distance'], 4)
data['Trip_Distance_bin'].value_counts()
data.loc[ data['Trip_Distance'] <= 27.54, 'Trip_Distance'] = 0
data.loc[(data['Trip_Distance'] > 27.54) & (data['Trip_Distance'] <= 54.77), 'Trip_Distance'] = 1
data.loc[(data['Trip_Distance'] > 54.77) & (data['Trip_Distance'] <= 82), 'Trip_Distance'] = 2
data.loc[data['Trip_Distance'] > 82, 'Trip_Distance'] = 3

data['Life_Style_Index_bin'] = pd.cut(data['Life_Style_Index'], 4)
data['Life_Style_Index_bin'].value_counts()
data.loc[ data['Life_Style_Index'] <= 2.207, 'Life_Style_Index'] = 0
data.loc[(data['Life_Style_Index'] > 2.207) & (data['Life_Style_Index'] <= 3.096), 'Life_Style_Index'] = 1
data.loc[(data['Life_Style_Index'] > 3.096) & (data['Life_Style_Index'] <= 3.986), 'Life_Style_Index'] = 2
data.loc[data['Life_Style_Index'] > 3.986, 'Life_Style_Index'] = 3

data['Customer_Rating_bin'] = pd.cut(data['Customer_Rating'], 4)
data['Customer_Rating_bin'].value_counts()
data.loc[ data['Customer_Rating'] <= 1.251, 'Customer_Rating'] = 0
data.loc[(data['Customer_Rating'] > 1.251) & (data['Customer_Rating'] <= 2.501), 'Customer_Rating'] = 1
data.loc[(data['Customer_Rating'] > 2.501) & (data['Customer_Rating'] <= 3.75), 'Customer_Rating'] = 2
data.loc[data['Customer_Rating'] > 3.75, 'Customer_Rating'] = 3


# data['Var1_bin'] = pd.cut(data['Var1'], 4)
# data['Var1_bin'].value_counts()
# data.loc[ data['Var1'] <= 75, 'Var1'] = 0
# data.loc[(data['Var1'] > 75) & (data['Var1'] <= 120), 'Var1'] = 1
# data.loc[(data['Var1'] > 120) & (data['Var1'] <= 165), 'Var1'] = 2
# data.loc[data['Var1'] > 165, 'Var1'] = 3

data['Var2_bin'] = pd.cut(data['Var2'], 4)
data['Var2_bin'].value_counts()
data.loc[ data['Var2'] <= 61, 'Var2'] = 0
data.loc[(data['Var2'] > 61) & (data['Var2'] <= 82), 'Var2'] = 1
data.loc[(data['Var2'] > 82) & (data['Var2'] <= 103), 'Var2'] = 2
data.loc[data['Var2'] > 103, 'Var2'] = 3

data['Var3_bin'] = pd.cut(data['Var3'], 4)
data['Var3_bin'].value_counts()
data.loc[ data['Var3'] <= 90.5, 'Var3'] = 0
data.loc[(data['Var3'] > 90.5) & (data['Var3'] <= 129), 'Var3'] = 1
data.loc[(data['Var3'] > 129) & (data['Var3'] <= 167.5), 'Var3'] = 2
data.loc[data['Var3'] > 167.5, 'Trip_Distance'] = 3


listRemove = data.columns.tolist()[-5:]
data = data.drop(listRemove, axis = 1)

####seperating the train test
data11_test = data [data['Surge_Pricing_Type'].isnull()]
data11_train = data[data['Surge_Pricing_Type'].notna()]
# data11_train['Surge_Pricing_Type']=le.fit_transform(data11_train['Surge_Pricing_Type'])

data11_test.drop(['Surge_Pricing_Type'],1,inplace=True)

####################### machine learning ##############################################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# import sklearn.metrics as metrics
# from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
# from sklearn.metrics import confusion_matrix, classification_report , f1_score
# from confusionMatrix import plotConfusionMatrix
# from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier



#set seed for same results everytime
seed=0
import sklearn.ensemble as ensemble

X=data11_train.drop('Surge_Pricing_Type',1)
y=data11_train['Surge_Pricing_Type']
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state =1)

##############DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf1 = round(clf.score(X_train, y_train) * 100, 2)
clf2 = round(clf.score(X_val, y_val) * 100, 2)
y_prediction1 = clf.predict(data11_test)

submission=test_data['Trip_ID']
y_pprreedd1= pd.DataFrame(y_prediction1)
final_submissions1 = pd.concat([submission,y_pprreedd1,], axis = 1)
final_submissions1.to_csv("D:/janataaatat/New folder/dt2.csv",index=False)


##############LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log1 = round(logreg.score(X_train, y_train) * 100, 2)
acc_log2 = round(logreg.score(X_val, y_val) * 100, 2)
acc_log1
acc_log2
y_prediction2 = logreg.predict(data11_test)

submission=test_data['Trip_ID']
y_pprreedd2= pd.DataFrame(y_prediction2)
final_submissions2 = pd.concat([submission,y_pprreedd2,], axis = 1)
final_submissions2.to_csv("D:/janataaatat/New folder/logistic2.csv",index=False)

##############SVC
# svc = SVC()
# svc.fit(X_train, y_train)
# acc_svc1 = round(svc.score(X_train, y_train) * 100, 2)
# acc_svc2 = round(svc.score(X_val, y_val) * 100, 2)
# acc_svc1
# acc_svc2

##############GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian1 = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian2 = round(gaussian.score(X_val, y_val) * 100, 2)
acc_gaussian1
acc_gaussian2

y_prediction3 = gaussian.predict(data11_test)
submission=test_data['Trip_ID']
y_pprreedd3= pd.DataFrame(y_prediction3)
final_submissions3 = pd.concat([submission,y_pprreedd3,], axis = 1)
final_submissions3.to_csv("D:/janataaatat/New folder/nb2.csv",index=False)


###############KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_knn1 = round(knn.score(X_train, y_train) * 100, 2)
acc_knn2 = round(knn.score(X_val, y_val) * 100, 2)
acc_knn1
acc_knn2

y_prediction4 = knn.predict(data11_test)
submission=test_data['Trip_ID']
y_pprreedd4= pd.DataFrame(y_prediction4)
final_submissions4 = pd.concat([submission,y_pprreedd4], axis = 1)
final_submissions4.to_csv("D:/janataaatat/New folder/knn2.csv",index=False)


###############RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest1 = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest2 = round(random_forest.score(X_val, y_val) * 100, 2)
acc_random_forest1
acc_random_forest2

y_prediction5 = random_forest.predict(data11_test)
submission=test_data['Trip_ID']
y_pprreedd5= pd.DataFrame(y_prediction5)
final_submissions5 = pd.concat([submission,y_pprreedd5], axis = 1)
final_submissions5.to_csv("D:/janataaatat/New folder/rf2.csv",index=False)


###############XGBClassifier
# model = XGBClassifier()
# model.fit(X_train, y_train)
# accuracy1 = accuracy_score(X_train, y_train)
# accuracy2 = accuracy_score(X_val, y_val)

#################### model evaluation is done only for validation data
models = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression', 'GaussianNB', 'KNN', 'RandomForest'],
    'train_Score': [clf1, acc_log1, acc_gaussian1, acc_knn1, acc_random_forest1],
    'val_Score': [clf2, acc_log2, acc_gaussian2, acc_knn2, acc_random_forest2]})

models.sort_values(by='val_Score', ascending=False)



from sklearn.model_selection import GridSearchCV

parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
# parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train, y_train)
preds1 = tree.predict(data11_test)
accu1 = tree.score(X_train, y_train)

submission=test_data['Trip_ID']
y_pprreedd8= pd.DataFrame(preds1)
final_submissions8 = pd.concat([submission,y_pprreedd8], axis = 1)
final_submissions8.to_csv("D:/janataaatat/New folder/decisiontgrid2.csv",index=False)

parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
# parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(random_forest,parameters)
tree.fit(X_train, y_train)
preds2 = tree.predict(data11_test).astype('int64')
accu2 = tree.score(X_train, y_train)

submission=test_data['Trip_ID']
y_pprreedd9= pd.DataFrame(preds2)
final_submissions9 = pd.concat([submission,y_pprreedd9], axis = 1)
final_submissions9.to_csv("D:/janataaatat/New folder/rfgrid2.csv",index=False)


# import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=1000, max_depth=10)
model_G.fit(X_train, y_train)

# make predictions for test set
y_pred_xg = model_G.predict(data11_test)
preds = [round(value) for value in y_pred_xg]

accxg = model_G.score(X_val,y_val)

submission=test_data['Trip_ID']
y_pprreedd10= pd.DataFrame(y_pred_xg)
final_submissions9 = pd.concat([submission,y_pprreedd10], axis = 1)
final_submissions9.to_csv("D:/janataaatat/New folder/xgboost7.csv",index=False)

from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)

from numpy import sort
from sklearn.feature_selection import SelectFromModel

thresholds = sort(model_G.feature_importances_)

feature_num = []
acc = []

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model_G, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train, sample_weight=sample_weights_data)
    # eval model
    select_X_val = selection.transform(data11_test)
    y_predlast = selection_model.predict(select_X_val)
    predictions = [round(value) for value in y_predlast]
    accuracy = accuracy_score(y_val, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    feature_num.append(select_X_train.shape[1])
    acc.append(accuracy*100.0)

submission=test_data['Trip_ID']
y_pprreeddlast= pd.DataFrame(y_predlast)
final_submissionslast = pd.concat([submission,y_pprreeddlast], axis = 1)
final_submissionslast.to_csv("D:/janataaatat/New folder/xgboostlast.csv",index=False)



