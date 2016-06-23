from flask import render_template
from flask import request
from flask import send_file
from app import app
from predict import pricePredict
import pymysql as mdb
import pandas as pd
#import pylab as plt
import io
import pickle
import base64
# my python scripts
import matplotlib
#matplotlib.use('PDF')
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

massCutOff = 100000e-3;

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/listings')
def products_page_fancy():
	con = None
	con = mdb.connect(user="userX", host="localhost", passwd="qwerty", db="evolution_db",  charset='utf8')

	#pull 'products' from input field and store it
	searchStr = request.args.get('products')
	query = "SELECT * FROM listings WHERE product LIKE '%%%s%%' LIMIT 100;" % (searchStr)
	query_results=pd.read_sql_query(query,con)
	products = []
	for i in range(0,query_results.shape[0]):
        	products.append(dict(date=query_results.iloc[i]['date'], product=query_results.iloc[i]['product'], price=query_results.iloc[i]['price']))
		the_result = pricePredict(products[0]['price'])
	return render_template("listings_ajax.html", products = products, the_result = the_result)

@app.route('/search')
def search():
	con = None
	con = mdb.connect(user="userX", host="localhost", passwd="qwerty", db="evolution_db",  charset='utf8')

	#pull 'products' from input field and store it
	searchStr = request.args.get('product')
	query = "SELECT * FROM listings WHERE product LIKE '%%%s%%' LIMIT 100;" % (searchStr)
	query_results=pd.read_sql_query(query,con)
	products = []
	for i in range(0,query_results.shape[0]):
		products.append(dict(date=query_results.iloc[i]['date'], product=query_results.iloc[i]['product'], price=query_results.iloc[i]['price']))
		the_result=products[0]['price']
	return render_template("listings_ajax.html", products = products, the_result=the_result)


@app.route('/showHist')
def get_image():
	# Plot histogram
	import pandas as pd
	import matplotlib.pyplot as plt
	matplotlib.use('PDF')
	import io

	m = pd.read_csv('aPVP_massesInGrams.csv')
	masses = m['masses'].values

	fig = plt.figure()
	n, bins, patches = plt.hist(masses, bins=(1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6),normed=0, facecolor='green', alpha=0.5, label="Product mass")

	fig.suptitle('Alpha-PVP darknet market distribution', fontsize=15)
	plt.xlabel("Product mass (g)",fontsize=15)
	plt.ylabel("Number of listings",fontsize=15)
	plt.grid(True)
	plt.legend(shadow=True, fancybox=True)
	plt.xscale('log')
#	plt.show()
	img = io.BytesIO()
	fig.savefig(img)
	img.seek(0)
	return send_file(img, mimetype='image/png')

@app.route('/aPVAnalysis')
def aPVAnalysis():
	global massCutOff
	massCutOffStr = request.args.get('massCutoff')
	try:
		massCutOff = float(massCutOffStr)
	except:
		massCutOff = 1e9
	return render_template("show.html")

@app.route('/ts_plot')
def getplot():
	img = plotTimeSeries.getplot(1)
	return base64.b64encode(img.getvalue())

# Time series plots
@app.route('/timeseries')
def timeseries():
	global drugname
	drugname = request.args.get('drugname')
	return render_template("timeseries.html")

@app.route('/tsFenListing')
def timeseriesListing():
    img_price, img_listings = plotFTimeSeries.getFentanylTimeSeries(drugname)
    return base64.b64encode(img_listings.getvalue())

@app.route('/tsFenAvgPrice')
def timeseriesAvgPrice():
    img_price, img_listings = plotFTimeSeries.getFentanylTimeSeries(drugname)
    return base64.b64encode(img_price.getvalue())

#Impact factor analysis
@app.route('/impact')
def impact():
	return render_template("impact.html")

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

@app.route('/betterPlace')
def betterPlace():
	massCutOffStr = request.args.get('featureModifier')
	try:
		featureMod = float(featureModifier)
	except:
		featureMod = 1.0
	img = ShowImpactFactors.betterPlace(featureMod)
	return send_file(img, mimetype='image/png')






