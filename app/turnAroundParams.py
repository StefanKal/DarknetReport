# coding: utf-8
## plotFTimeSeries

def findTurnaroundParams():
    import cPickle
    import numpy as np
    from datetime import datetime
    import matplotlib.dates as mdates
    import pylab as plt
    from random import randint
    %matplotlib inline


    X = np.load('./app/model/Fentanyl_Feature_X.npy')
    y = np.load('./app/model/Fentanyl_Output_y.npy')
    yLinear = np.load('./app/model/Fentanyl_Output_yLinear.npy')
    Xlabel = np.load('./app/model/Fentanyl_Xlabel.npy')
    dateVector = np.load('./app/model/Fentanyl_dateVector.npy')
    dateVector = dateVector.astype(datetime)
    autoSel = np.load('./app/model/AutoSelect.npy')

    with open(r"./app/model/DNetModel.b", "rb") as input_file:
        deployed_model = cPickle.load(input_file)


    idx = randint(0,len(autoSel))
    dollarMod = autoSel[idx][0]
    numberMod = autoSel[idx][1]

    XX = np.copy(X)
    # Fentanyl HCl price
    XX[:,0] *= dollarMod
    XX[:,2] *= dollarMod
    XX[:,4] *= dollarMod
    XX[:,18] *= dollarMod
    XX[:,20] *= dollarMod
    XX[:,22] *= dollarMod
    # China White price
    XX[:,28] *= dollarMod
    XX[:,30] *= dollarMod
    # Fentanyl number of listings
    XX[:,1] *= numberMod
    XX[:,3] *= numberMod
    XX[:,5] *= numberMod
    XX[:,19] *= numberMod
    XX[:,21] *= numberMod
    XX[:,23] *= numberMod
    # China White number of listings
    XX[:,29] *= numberMod
    XX[:,31] *= numberMod

    # probability for 1 year after data start
    probMod = deployed_model.predict_proba(XX)
    pDecreasing = np.mean(probMod[355:375,0])
    pStable = np.mean(probMod[355:375,1])
    pIncreasing = np.mean(probMod[355:375,2])
    
    return dollarMod, numberMod, pDecreasing

if __name__ == '__main__':
    dollarMod, numberMod, pDecreasing = findTurnaroundParams()
    print dollarMod, numberMod, pDecreasing
