@app.route('/predict')
def getprediction():
	import os
	cdir = os.getcwd()	
	deployed_model = pickle.load( open( cdir+"/DNetModel.p", "rb" ) )
	print 'nothing yet'
#	features = request.args.get('features')
#	prediction = deployed_model.predict(features)
	return render_template('home.html')
