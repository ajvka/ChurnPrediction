import preprocess as pp


def calcCorr(raw_train):
	print("\n\nPEARSON CORRELATION: \n")
	print(raw_train.drop(['st', 'phnum'], axis=1).corr(method='pearson'))
	print("\n\nSPEARMAN CORRELATION: \n")
	print(raw_train.drop(['st', 'phnum'], axis=1).corr(method='spearman'))


def removeCorr(raw_train):
	data_train = raw_train
	pp.removeCols(data_train, ['st', 'nummailmes', 'phnum', 'tdmin', 'temin', 'tnmin', 'timin'])
	print("\n\nPREPROCESSED DATA: \n")
	print(data_train)
	data_train.info()
	return data_train


