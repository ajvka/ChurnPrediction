import dataPreprocessing as dpp
import statsmodels.api as sma
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def createRandomForestModel(X, Y):
    # rfc = RandomForestClassifier()
    # train logreg model
    logreg = RandomForestClassifier(max_depth=8)
    logreg.fit(data_train[X], data_train[Y].values.ravel())
    predictedY = logreg.predict(data_test[X])
    print('\nRandom Forest Classifier Test Accuracy: {:.3f}\n'.format(logreg.score(data_test[X], data_test[Y])))

    # k-fold cross validation
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = RandomForestClassifier()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, data_train[X], data_train[Y].values.ravel(), cv=kfold,
                                              scoring=scoring)
    print("\nK-Fold Cross Validation Accuracy: %.3f\n" % (results.mean()))

    # confusion matrix
    conf_matrix = confusion_matrix(data_test[Y], predictedY)
    print(conf_matrix)
    print(classification_report(data_test[Y], predictedY))

    # plot ROC curve
    logit_roc_auc = roc_auc_score(data_test[Y], logreg.predict(data_test[X]))
    fpr, tpr, thresholds = roc_curve(data_test[Y], logreg.predict_proba(data_test[X])[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="upper right")
    plt.savefig('RandomForest_ROC')
    plt.show()


data_train = dpp.data_train
data_test = dpp.data_test

# get features from RFE results
X = dpp.getRFEFeatures()
Y = ['churn']

createRandomForestModel(X, Y)
