from flask import render_template
from flask import request
from flask import send_file
from app import app
import pymysql as mdb
import pandas as pd
#import pylab as plt
import io
import pickle
import base64
# my python scripts
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import plotTimeSeries
import plotFTimeSeries
import ShowImpactFactors
import predict

massCutOff = 100000e-3;

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/test')
def test():
	return render_template('test.html')

@app.route('/presentation')
def GDrivePresentation():
	return render_template('presentation.html')

#print debug 
@app.route('/debug')
def cDir():
	messages = []
	messages.append( ShowImpactFactors.currentDirectory() )
	return render_template('debug.html', messages = messages)


# Listings from SQL db
@app.route('/listings')
def listings():
	con = None
	#con = mdb.connect(user="userX", host="localhost", passwd="qwerty", db="evolution_db",  charset='utf8')	
	con = mdb.connect(user="root", host="localhost", passwd="qwerty", db="evolution_db",  charset='utf8')

	searchStr = request.args.get('products')
	query = "SELECT * FROM listings WHERE product LIKE '%%%s%%' LIMIT 10;" % (searchStr)
	query_results=pd.read_sql_query(query,con)
	products = []
	for i in range(0,query_results.shape[0]):
        	products.append(dict(date=query_results.iloc[i]['date'], product=query_results.iloc[i]['product'], price=query_results.iloc[i]['price']))
	return render_template("listings.html", products = products)

# Time series analysis
@app.route('/timeseries')
def timeseries():
	return render_template("timeseries.html")

@app.route('/tsFenListing')
def timeseriesListing():
    drugname = request.args.get('drugname')
    if drugname==None: drugname='Fentanyl HCl'
    print drugname
    img_price, img_listings = plotFTimeSeries.getFentanylTimeSeries(drugname)
    return base64.b64encode(img_listings.getvalue())

@app.route('/tsFenAvgPrice')
def timeseriesAvgPrice():
    drugname = request.args.get('drugname')
    if drugname==None: drugname='Fentanyl HCl'
    print drugname
    img_price, img_listings = plotFTimeSeries.getFentanylTimeSeries(drugname)
    return base64.b64encode(img_price.getvalue())

# Impact factor analysis

@app.route('/showImpactFactors')
def showImpactFactorsPlot():
	img = ShowImpactFactors.impactBarGraph(0)
	return send_file(img, mimetype='image/png')

@app.route('/googleNewsTrend')
def showGoogleNewsTrend():
	img = ShowImpactFactors.googleNewsTrend()
	return send_file(img, mimetype='image/png')

@app.route('/googleNewsTrendDiff')
def showGoogleNewsTrendDiff():
	img = ShowImpactFactors.googleNewsTrendDiff()
	return send_file(img, mimetype='image/png')

# Predict decrease
@app.route('/impact')
def predictSomething():
	pF_str = request.args.get('pF')
	pCW_str = request.args.get('pCW')
	numF_str = request.args.get('numF')
	numCW_str = request.args.get('numCW')
	numCW_str = request.args.get('numCW')
	button = request.args.get('button')
	if button=="predict":
		print "predicting"
		pF=float(pF_str)
		pCW=float(pCW_str)
		numF=float(numF_str)
		numCW=float(numCW_str)
	elif button=="findParams":
		print "find params"
		modPrice, modNumList = predict.findTurnaroundParams()
		pF=int(modPrice*100)
		pCW=int(modPrice*100)
		numF=int(modNumList*100)
		numCW=int(modNumList*100)
		pF_str=str(pF)
		pCW_str=str(pCW)
		numF_str=str(numF)
		numCW_str=str(numCW)
	else:
		print "reset"
		pF=100
		pF_str="100"
		pCW=100
		pCW_str="100"
		numF=100
		numF_str="100"
		numCW=100
		numCW_str="100"

	probUp,probDown = predict.predictDecrease(pF/100,pCW/100,numF/100,numCW/100)
	imgPath = predict.getArrow(probDown)
	return render_template("impact2.html",probDown=probDown,pF_str=pF_str,pCW_str=pCW_str,numF_str=numF_str,numCW_str=numCW_str,imgPath=imgPath)




