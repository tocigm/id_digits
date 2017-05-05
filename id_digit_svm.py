from sklearn import svm, linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import numpy as np
import gzip
import pickle

def run():
    data_path = "/home/tra161/WORK/Data/bagiks/ID_DIGITS/gray_labelled/gray_data.pkl.gz"
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
    max_ = np.amax(train_dat,axis=0)
    min_ = np.amin(train_dat,axis=0)
    train_dat =  2*(train_dat-min_)/(max_-min_)-1 # to -1:1
    valid_dat =  2*(valid_dat-min_)/(max_-min_)-1 # to -1:1
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
if __name__=="__main__":
    run()
