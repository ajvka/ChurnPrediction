import pandas as pd

def readData(path_train):
    data = pd.read_csv(path_train, delimiter=', ', index_col=False, engine='python')
    data['label'] = data.apply(lambda row: removeTrailingDot(row['label']), axis=1)
    print("\nREAD DATA: \n")
    print(data)
    data.info()
    return data


def label(data):
    # construct list of column labels of dataframe
    data_vars = data.columns.values.tolist()
    # specify dependent var as Y
    Y = ['churn']
    # specify list of independent vars as X
    X = [i for i in data_vars if i not in Y]
    return X, Y


def toBoolnum(ele):
    if ele == 'False' or ele == 'no':
        return 0
    else:
        return 1


def boolnumCols(dataframe, col_list):
    for i in col_list:
        dataframe[i] = dataframe.apply(lambda row: toBoolnum(row[i]), axis=1)


def removeTrailingDot(ele):
    return ele[:-1]


def removeCols(dataframe, col_list):
    for i in col_list:
        del dataframe[i]
