# coding: utf-8
## Show impact factors

import numpy as np
import pylab as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from flask import send_file

def currentDirectory():
	Xlabel = np.load('./app/model/Fentanyl_Xlabel.npy')
	return Xlabel[0]


def impactBarGraph(featureModifier = []):

	X = np.load('./app/model/Fentanyl_Feature_X.npy')
	y = np.load('./app/model/Fentanyl_Output_y.npy')
	yLinear = np.load('./app/model/Fentanyl_Output_yLinear.npy')
	Xlabel = np.load('./app/model/Fentanyl_Xlabel.npy')

	# separate train/test data
	np.random.seed(123)
	is_trainData = np.random.uniform(0, 1, len(y)) <= .75
	X_train, X_test = X[is_trainData==True], X[is_trainData==False]
	y_train, y_test = y[is_trainData==True], y[is_trainData==False]

	# Create Random Forest object
	model= RandomForestClassifier(n_estimators=10)
	# Train the model using the training sets and check score
	model.fit(X_train,y_train)
	# Predict Output
	y_predicted = model.predict(X_test)

	print('Training/Test Ratio: {}'.format(X_train.shape[0] / X_test.shape[0]))
	pd.crosstab(y_test, y_predicted, rownames=['actual'], colnames=['preds'])

	# Get Feature Importance from the classifier
	numBars = 5
	feature_importance = model.feature_importances_
	# Normalize The Features
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	fig = plt.figure(figsize=(3, 4))
	plt.barh(pos[len(pos)-numBars-1:-1], feature_importance[sorted_idx][len(pos)-numBars-1:-1], align='center', color='#30C0E0')
	plt.yticks(pos[len(pos)-numBars-1:-1], np.asanyarray(Xlabel)[sorted_idx][len(pos)-numBars-1:-1],fontsize = 15)
	plt.xlabel('Relative Importance',fontsize = 20)
	plt.title('Impact factors on Fentanyl overdose reports',fontsize = 15)
	img = io.BytesIO()
	fig.savefig(img,bbox_inches='tight')
	img.seek(0)
	return img

def googleNewsTrend():
	yLinear = np.load('./app/model/Fentanyl_Output_yLinear.npy')
	dateVector = np.load('./app/model/Fentanyl_dateVector.npy')
	dateVector = dateVector.astype(datetime)
	avgWnd = 20 # data was smoothed witha moving average window of 20 width
	plt.close('all')
	fig, ax = plt.subplots(1,figsize=(4, 2.5))
	ax.plot(dateVector, yLinear/avgWnd, color='blue', linewidth=2.0)
	plt.ylabel('News reports')
	# rotate and align the tick labels so they look better
	fig.autofmt_xdate()
	# use a more precise date string for the x axis locations in the toolbar
	ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
	xLim = ax.get_xlim()
	ax.set_xlim(xLim[0], xLim[1]+50) # add some space on the right
	#plt.title("""Google News reports on "Fentanyl overdose" """)
	img = io.BytesIO()
	fig.savefig(img,bbox_inches='tight')
	img.seek(0)
	return img

def googleNewsTrendDiff():
	yClassified = np.load('./app/model/Fentanyl_Output_y.npy')
	yDiff = np.load('./app/model/Fentanyl_Output_yDiff.npy')
	dateVector = np.load('./app/model/Fentanyl_dateVector.npy')
	dateVector = dateVector.astype(datetime)

	yC = yClassified*np.nanstd(yDiff)*2.5
	plt.close('all')
	fig, ax = plt.subplots(1,figsize=(4, 2.5))
	ax.plot(dateVector, yDiff, color='black', linewidth=1.0)
	ax.plot(dateVector,yC, color='red', linewidth=2.0)
	plt.ylabel('Change rate / week')
	# rotate and align the tick labels so they look better
	fig.autofmt_xdate()
	# use a more precise date string for the x axis locations in the toolbar
	ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
	yLim = ax.get_ylim()
	ax.set_ylim(-yLim[1]*1.1, yLim[1]*1.1) # add some space on the right
	#plt.title("""Google News reports on "Fentanyl overdose" """)
	img = io.BytesIO()
	fig.savefig(img,bbox_inches='tight')
	img.seek(0)
	return img


def betterPlace(featureModifier):
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pylab as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import cross_validation
    import io
  
    X = np.load('./app/model/Fentanyl_Feature_X.npy')
    y = np.load('./app/model/Fentanyl_Output_y.npy')
    yLinear = np.load('./app/model/Fentanyl_Output_yLinear.npy')
    Xlabel = np.load('./app/model/Fentanyl_Xlabel.npy')
    dateVector = np.load('./app/model/Fentanyl_dateVector.npy')
    dateVector = dateVector.astype(datetime)

    # separate train/test data
    np.random.seed(123)
    is_trainData = np.random.uniform(0, 1, len(y)) <= .75
    X_train, X_test = X[is_trainData==True], X[is_trainData==False]
    y_train, y_test = y[is_trainData==True], y[is_trainData==False]

    # Create Random Forest object
    model= RandomForestClassifier(n_estimators=10)
    # Train the model using the training sets and check score
    model.fit(X_train,y_train)
    # Predict Output

#     print Xlabel[5]
#     print Xlabel[28]
#     print Xlabel[30]

    XX = X
    XX[:,5] *= featureModifier
    XX[:,28] *= featureModifier
    XX[:,30] *= featureModifier
    y_predicted = model.predict(X)
    y_predictedScaled = model.predict(XX)
#     print dateVector[350]
#     print y_predicted[350]
    from scipy import misc
    arrowUp = misc.imread('./app/static/images/arrowUp.png')
    arrowFlat = misc.imread('./app/static/images/arrowFlat.png')
    arrowDown = misc.imread('./app/static/images/arrowDown.png')
    
    import matplotlib.pyplot as plt
#    print y_predictedScaled[350]

    if y_predictedScaled[350]==1:
        img = arrowUp
    if y_predictedScaled[350]==0:
        img = arrowFlat
    if y_predictedScaled[350]==-1:
        img = arrowDown
    
    with open("./app/static/images/arrowUp.png", "rb") as imageFile:
        f = imageFile.read()
        b = bytearray(f)
    return b


if __name__ == '__main__':
    featureModifier = 0.25
    outcome = betterPlace(featureModifier)
    print outcome



