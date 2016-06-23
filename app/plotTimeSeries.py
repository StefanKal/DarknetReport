# coding: utf-8
## plotTimeSeries

def getplot(massCutOff):
	import pandas as pd
	import matplotlib.pyplot as plt
	import io

	dates = pd.read_csv('aPVP_dates.csv')
	prices = pd.read_csv('aPVP_prices.csv')
	masses = pd.read_csv('aPVP_massesInGrams.csv')
	BTC_USD_rate = pd.read_csv('bitcoin_dollar_rate.csv') # BTC course for every single day

	# match to dates in dataset
	r = []
	for d in dates['dates']:
	    idx = BTC_USD_rate['Date'].values == d
	    r.append( BTC_USD_rate['Close'].values[idx] )
	rate = pd.DataFrame(r, columns=['BTC_USD_rate'])

	# combine in pandas array
	df = pd.concat([dates, prices, rate, masses], axis=1)

	print massCutOff

	# select only masses within desired range
	df_cutOff = df[ (df['masses']>1e-9) & (df['masses'] <= massCutOff) ]

	# Plot time series plot
	from datetime import datetime

	x = []
	for date in df_cutOff['dates'].values:
		x.append(datetime.strptime(date, "%Y-%m-%d"))
    
	y = []
	for ii, price in enumerate(df_cutOff['prices'].values):
		pBTC = float(price.replace("BTC ",""))
		pUSD = pBTC*df_cutOff['BTC_USD_rate'].values[ii]
		m = df_cutOff['masses'].values[ii]
		pricePerGram = pUSD/m
		y.append( pricePerGram )

	# average prices on the same date
	yAvg = []
	unique_dates = df_cutOff['dates'].unique()
	for ud in unique_dates:
		idx = df_cutOff['dates'].values == ud
		allPricesOfThatDay = pd.Series(y)[idx]
		yAvg.append( allPricesOfThatDay.mean(axis=0) )

	xAvg = []
	for date in unique_dates:
		xAvg.append(datetime.strptime(date, "%Y-%m-%d"))

	fig = plt.figure()
	ts = pd.Series(yAvg,xAvg)
	ax = ts.plot()
	ax.set_ylabel("Average price ($)")
	img = io.BytesIO()
	fig.savefig(img)
	img.seek(0)
	return img





if __name__ == '__main__':
	featureModifier = 0
	impactBarGraph(featureModifier)

