import preprocess as pp
import correlation as correlation 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier


def formatData(data):
    # preprocess raw data
    # convert booleans to 0-1
    pp.boolnumCols(data, ['intplan', 'voice', 'label'])
    # relabel columns
    data.rename(columns={'tecahr': 'techar', 'tn cal': 'tncal', 'label': 'churn'}, inplace=True)


# specify path for training data
path_train = r'./dataset/churn_train.txt'

# read raw csv data into pandas dataframe
raw_train = pp.readData(path_train)
formatData(raw_train)

# calculate correlations
correlation.calcCorr(raw_train, ['st', 'phnum'])

# remove the correlated vars from final training data
data_train=correlation.removeCorr(raw_train, ['st', 'nummailmes', 'phnum', 'tdmin', 'temin', 'tnmin', 'timin'])

# specify dependent and independant variables
X, Y = pp.label(raw_train)

# recursive feature elimination
lr = LogisticRegression()
# specifying target number of features = 8
rfe = RFE(lr, 8)
rfe = rfe.fit(data_train[X], data_train[Y].values.ravel())
print("\n\nRFE RESULTS: \n")
print(rfe.support_)
print(rfe.ranking_)

# feature importance calculation
etc_model = ExtraTreesClassifier()
etc_model.fit(data_train[X], data_train[Y].values.ravel())
print("\n\nFEATURE IMPORTANCE: \n")
print(etc_model.feature_importances_)


# get features based on RFE results
def getRFEFeatures():
    features = []
    for index, i in enumerate(rfe.ranking_):
        if i == 1:
            features.append(X[index])
    return features


# get features based on ETC results
def getETCFeatures(threshold):
    features = []
    for index, i in enumerate(etc_model.feature_importances_):
        if i >= threshold:
            features.append(X[index])
    return features


# read test data
path_test = r'./dataset/churn_test.txt'
data_test = pp.readData(path_test)
formatData(data_test)
