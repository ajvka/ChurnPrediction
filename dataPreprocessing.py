import pandas as pd
import preprocess as pp
import correlation as correlation 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def readData(path_train):
	raw_train = pd.read_csv(path_train, delimiter=', ', index_col=False, engine='python')
	raw_train['label'] = raw_train.apply(lambda row: pp.removeTrailingDot(row['label']), axis=1)
	print("\nRAW DATA: \n")
	print(raw_train)
	raw_train.info()
	return raw_train


	
def label(raw_train):
	# construct list of column labels of dataframe
	data_vars = raw_train.columns.values.tolist()
	# specify dependent var as Y
	Y = ['churn']
	# specify list of independent vars as X
	X = [i for i in data_vars if i not in Y]
	return X,Y	




# specify path for training data
path_train = r'./dataset/churn_train.txt'

# read raw csv data into pandas dataframe
raw_train=readData(path_train)

# preprocess raw data
# convert booleans to 0-1
pp.boolnumCols(raw_train, ['intplan', 'voice', 'label'])

# relabel columns
raw_train.rename(columns={'tecahr':'techar', 'tn cal':'tncal', 'label':'churn'}, inplace=True)

# calculate correlations
correlation.calcCorr(raw_train)

# remove the correlated vars from final training data
data_train=correlation.removeCorr(raw_train)

#specify dependent and independant variables
X,Y = label(raw_train)

# recursive feature elimination
lr = LogisticRegression()

# specifying target number of features = 5
rfe = RFE(lr, 5)
rfe = rfe.fit(data_train[X], data_train[Y])
print("\n\nRFE RESULTS: \n")
print(rfe.support_)
print(rfe.ranking_)

