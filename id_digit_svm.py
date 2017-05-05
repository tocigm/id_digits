from sklearn import svm, linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import numpy as np
import gzip
import pickle

from PIL import Image, ImageFilter
from id_gray_digit_data import DENOISE_THRESHOLD

NORMALISED = 2 # 0: no nomalise, 1: 0:1, 2: -1:1
PCA = 0 # TODO LATER

def run():
    data_path = "./gray_data.pkl.gz"
    with gzip.open(data_path,'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data_dict = data_dict = u.load()
        train_dat  = data_dict["X_train"]
        train_lab  = data_dict["y_train"]
        valid_dat  = data_dict["X_valid"]
        valid_lab  = data_dict["y_valid"]

    train_dat = np.reshape(train_dat,[-1,train_dat.shape[1]*train_dat.shape[2]])
    valid_dat = np.reshape(valid_dat,[-1,valid_dat.shape[1]*valid_dat.shape[2]])

    
    accs = []
    models = []
    # Normalize data
    max_=min_=0
    if NORMALISED>0:
        max_ = np.amax(train_dat,axis=0)
        min_ = np.amin(train_dat,axis=0)
        train_dat =  (train_dat-min_)/(max_-min_) 
        valid_dat =  (valid_dat-min_)/(max_-min_)
        if NORMALISED==2:
            train_dat = train_dat*2-1
            valid_dat = valid_dat*2-1
        
    ##########################################################################
    """ SVM """
    try:
        #print("normalise data to : %.3f - %.3f"%(np.amin(train_dat),np.amax(train_dat)))
        clf = svm.SVC(decision_function_shape="ovo")
        clf.fit(train_dat,train_lab)
        clf.decision_function_shape="ovr"
        pred = np.argmax(clf.decision_function(valid_dat),axis=1)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("SVM")

    # The best model here, save it
    save_model(clf,"SVM",max_,min_)
    ###########################################################################
    """ Naive Bayes """
    try:
        clf = MultinomialNB()
        clf.fit(train_dat,train_lab)
        pred = clf.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("NB")    
    ##########################################################################
    """ Logistic regression """
    try:
        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(train_dat, train_lab)
        pred = logreg.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("LG")
        #######################################################################
    try:
        lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        lda.fit(train_dat, train_lab)
        pred = lda.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("LDA")
    #######################################################################
    try:
        qda = QuadraticDiscriminantAnalysis(store_covariances=True)
        qda.fit(train_dat, train_lab)
        pred = qda.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("QDA")
    #####################################################################
    try:
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_dat, train_lab)
        pred = clf.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("DCT")
    ###################################################################
    try:
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(train_dat, train_lab)
        pred = clf.predict(valid_dat)
        pred = clf.predict(valid_dat)
        acc = np.mean(np.equal(pred,valid_lab).astype(float))
        accs.append(acc)
    except ValueError:
        accs.append(-1)
    models.append("RF")

    print("All results:")
    inds = np.argsort(accs)
    for inx in inds:
        print(models[inx] + " " + str(accs[inx]))

def save_model(model,modelname,max_,min_):
    with gzip.open("./model.pkl.gz","wb") as f:
        pickle.dump({"model":model,
                     "name":modelname,
                     "max_":max_,
                     "min_":min_},f)

def test():
    with gzip.open("./model.pkl.gz",'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data_dict = data_dict = u.load()
        clf  = data_dict["model"]
        max_ = data_dict["max_"]
        min_ = data_dict["min_"]
    clf.decision_function_shape="ovr"
    
    dat = Image.open("103_6_220601447+0.jpg")
    
    dat = np.asarray(dat)
    dat = np.where(dat<DENOISE_THRESHOLD,dat,255)[np.newaxis,:,:]
    dat = np.reshape(dat,[-1,dat.shape[1]*dat.shape[2]])
    if NORMALISED>0:
        dat = (dat-min_)/(max_-min_)
        if NORMALISED>1:
            dat = dat*2-1
            
    pred = np.argmax(clf.decision_function(dat),axis=1)
    print(pred)    
if __name__=="__main__":
    run()
    test()
