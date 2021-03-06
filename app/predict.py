# coding: utf-8
## Show impact factors

def findTurnaroundParams():
    import cPickle
    import numpy as np
    from random import randint
    
    XX = np.load('./app/model/Fentanyl_Feature_X.npy')
    autoSel = np.load('./app/model/AutoSelect.npy')

    with open(r"./app/model/DNetModel.b", "rb") as input_file:
        deployed_model = cPickle.load(input_file)


    idx = randint(0,len(autoSel)-1)
    dollarMod = autoSel[idx][0]
    numberMod = autoSel[idx][1]
   
    return dollarMod, numberMod

def predictDecrease(dollarModF,dollarModCW,numberModF,numberModCW):
    import pickle
    import numpy as np

    XX = np.load('./app/model/Fentanyl_Feature_X.npy')

    with open(r"./app/model/DNetModel.b", "rb") as input_file:
        deployed_model = pickle.load(input_file)

    # Fentanyl HCl price
    XX[:,0] *= dollarModF
    XX[:,2] *= dollarModF
    XX[:,4] *= dollarModF
    XX[:,18] *= dollarModF
    XX[:,20] *= dollarModF
    XX[:,22] *= dollarModF
    # China White price
    XX[:,28] *= dollarModCW
    XX[:,30] *= dollarModCW
    # Fentanyl number of listings
    XX[:,1] *= numberModF
    XX[:,3] *= numberModF
    XX[:,5] *= numberModF
    XX[:,19] *= numberModF
    XX[:,21] *= numberModF
    XX[:,23] *= numberModF
    # China White number of listings
    XX[:,29] *= numberModCW
    XX[:,31] *= numberModCW

    # probability for 1 year after data start
    probMod = deployed_model.predict_proba(XX)
    pDecreasing = np.mean(probMod[355:375,0])
    pStable = np.mean(probMod[355:375,1])
    pIncreasing = np.mean(probMod[355:375,2])

    pUp = int(100*(pIncreasing+0.5*pStable) + 1)	#let's round up
    pDown = int(100*(pDecreasing+0.5*pStable) + 1)

    return pUp,pDown

def getArrow(p):
    from scipy import misc
    if p<48:
        imgPath = "./static/images/arrowUp.png"
    elif p>=50:
        imgPath = "./static/images/arrowDown.png"
    return imgPath

if __name__ == '__main__':
    dollarMod, numberMod, pDecreasing = findTurnaroundParams()
    print dollarMod, numberMod, pDecreasing


