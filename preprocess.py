
def toBoolnum(ele):
    if ele=='False' or ele=='no':
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
