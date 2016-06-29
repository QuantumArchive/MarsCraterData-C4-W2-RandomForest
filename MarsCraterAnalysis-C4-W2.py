# coding: utf-8

"""
Created on Tue June 29 13:48:12 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.pylab as pll
import sklearn.metrics
import scipy.stats
import pydotplus
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from io import BytesIO
from IPython.display import Image

#from IPython.display import display
get_ipython().magic(u'matplotlib inline')

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with 'No Morphology'
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ','No Morphology')

#Remove any data with NaN values
data2 = data.dropna()
data2.describe()

#Again, our original hypothesis was to see if there was a relationship between a craters given latitude and its diameter. Because
#we want to predict the diameter, and to be able to use a random forest, we need a binary categorical variable. From the summary
#statistics table above we see that the 50% of all crater diameters is 1.53 km, so we'll use this as a natural boundary to
#split the data on. Because random forest is good at ranking different explanatory variables, we will be using the entire dataset.
#Additional, we will code craters with an EJECTA_MORPHOLOGY_1 equal to 0 if no morphology, and 1 if there is morphology.
    
def cratersize(x):
    if x <= 1.53:
        return 0
    else:
        return 1
    
def cratermorph(x):
    if x == 'No Morphology':
        return 0
    else:
        return 1
    
data2['CRATER_DIAM_BIN'] = data2['DIAM_CIRCLE_IMAGE'].apply(cratersize)
data2['CRATER_DIAM_BIN'] = data2['CRATER_DIAM_BIN'].astype('category')
data2['CRATER_MORPHOLOGY_BIN'] = data2['MORPHOLOGY_EJECTA_1'].apply(cratermorph)
data2['CRATER_MORPHOLOGY_BIN'] = data2['CRATER_MORPHOLOGY_BIN'].astype('category')

#So now we'll set up our predictors and target
predictors = data2[['LATITUDE_CIRCLE_IMAGE','LONGITUDE_CIRCLE_IMAGE','DEPTH_RIMFLOOR_TOPOG',
                    'CRATER_MORPHOLOGY_BIN','NUMBER_LAYERS']]

target = data2['CRATER_DIAM_BIN']

#set up our training and test data for our predictors and target
pred_train, pred_test, target_train, target_test = train_test_split(predictors, target, test_size=0.4)

#inspect the number of observations for each split
pred_train.shape
pred_test.shape
target_train.shape
target_test.shape

#Build model on training data
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,target_train)

#Now we produce our predictions
predictions = classifier.predict(pred_test)

print(sklearn.metrics.confusion_matrix(target_test,predictions))
print(sklearn.metrics.accuracy_score(target_test,predictions))

#fir an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,target_train)
#see relative importance of each attribute
print(model.feature_importances_)

#Running different number of trees and see how overall accuracy of the model fitting is

trees=range(25)
accuracy=numpy.zeros(25)

for idx in range(len(trees)):
    classifier=RandomForestClassifier(n_estimators=idx + 1)
    classifier=classifier.fit(pred_train,target_train)
    predictions=classifier.predict(pred_test)
    accuracy[idx]=sklearn.metrics.accuracy_score(target_test, predictions)
    
plt.cla()
plt.plot(trees,accuracy)