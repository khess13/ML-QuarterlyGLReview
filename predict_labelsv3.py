import pandas as pd
import pickle
import datetime as dt
import os
#run IT and RE together?

'''targets'''
root = os.getcwd()+'\\'
quarter = input('Which quarter? 1,2,3,4')
fileloc = root + 'PreprocessFY22Q'+quarter+'.csv'
exportloc = root
pickleloc = root + 'pickleJar\\'


datestamp = dt.datetime.now().strftime('%m-%d-%Y')

target = ['IT?', 'RE?']
ITRE_lab = ['IT', 'RE']

#read file, dtype specs for faster read
print(f'Processing {ITRE_lab}')
acct_trans = pd.read_csv(fileloc,
                    header = 0,
                    encoding = 'ISO-8859-1', #b/c excel update
                    usecols = ['mDocNo',
                               'PONo',
                               'Order - Key',
                               'BAKey',
                               'BATxt',
                               'VenKey',
                               'VenTxt',
                               'GLKey',
                               'Amt',
                               'GLTxt',
                               'Ven_LD_Header',
                               'Long Description',
                               'IT?',
                               'RE?',
                               'ITSuspect',
                               'RESuspect',
                               'ITBlocked',
                               'REBlocked'],
                      dtype = {'mDocNo': str,
                               'PONo' : str,
                               'Order - Key': str,
                               'BAKey':str,
                               'BATxt':str,
                               'VenKey':str,
                               'VenTxt':str,
                               'GLKey':str,
                               'Amt':str,
                               'GLTxt':str,
                               'Ven_LD_Header':str,
                               'Long Description':str,
                               'IT?':str,
                               'RE?':str,
                               'ITSuspect':str,
                               'RESuspect':str,
                               'ITBlocked':str,
                               'REBlocked':str})



#drop rows with agency code
acct_trans.dropna(subset = ['BAKey'], inplace = True)
#inplace = mod df
#convert label back to int
#acct_trans[ITRE] = acct_trans[ITRE].apply(lambda x: int(x))
#acct_trans[ITRE_lab+'Suspect'] = acct_trans[ITRE_lab+'Suspect'].apply(lambda x: int(x))
acct_trans['IT?'] = acct_trans['IT?'].apply(lambda x: int(x))
acct_trans['RE?'] = acct_trans['RE?'].apply(lambda x: int(x))
acct_trans['ITSuspect'] = acct_trans['ITSuspect'].apply(lambda x: int(x))
acct_trans['RESuspect'] = acct_trans['RESuspect'].apply(lambda x: int(x))
#blocked accts list
acct_trans['ITBlocked'] = acct_trans['ITBlocked'].apply(lambda x: "Blocked" if x == "1" else "")
acct_trans['REBlocked'] = acct_trans['REBlocked'].apply(lambda x: "Blocked" if x == "1" else "")

acct_trans['Ven_LD_HeaderITResult'] = ''
acct_trans['Ven_LD_HeaderREResult'] = ''
acct_trans['VenTxtITResult'] = ''
acct_trans['VenTxtREResult'] = ''

acct_trans['FYQtr'] = fileloc[-10:][:6] #FY18Q2 #fileloc[-6:][:2] #Q2

pickles = ['Ven_LD_HeaderIT', 'Ven_LD_HeaderRE', 'VenTxtIT', 'VenTxtRE']
#max loop
loop_len = len(pickles)
loop_ct = 0

for picklez in pickles:
    #create label options
    if picklez[-2:] == 'IT':
        IT_label = ['NIT', 'IT']
    else:
        IT_label = ['NRE', 'RE']
    #open pickled classifiers
    print(f'Loading {picklez[:-2]} classifier')
    text_clf = pickle.load(open(pickleloc+picklez+'.pickle', 'rb'))
    new_data = acct_trans[picklez[:-2]].copy()
    #total_rows = len(new_data)

    print(f'Predicting labels for {picklez[-2:]}')
    predicted = text_clf.predict(new_data)

    res_data = []
    res_lis = []
    print('Building list of results')
    for data, label in zip(new_data, predicted):
        res_data.append(data)
        res_lis.append(IT_label[label])
    #rowcount = 0

    print('Rebuilding dataframe with predicted labels')
    #append to df
    #res_df = pd.DataFrame({picklez[:-2] : res_data, picklez+'Result' : res_lis})
    res_df = pd.DataFrame({picklez+'Result' : res_lis})
    acct_trans.update(res_df)#, raise_conflict=True)


acct_fin = acct_trans

print('Sorting positive and negative results.')
#ventxt pos or ITResult is pos, IT? is neg
posIT = acct_fin.loc[((acct_fin['Ven_LD_HeaderITResult'] == 'IT') & (acct_fin['IT?'] == 0) & (acct_fin['RE?'] == 0)) |\
                    ((acct_trans['VenTxtITResult'] == 'IT') & (acct_trans['ITSuspect'] == 1))].copy()
posIT['Type'] = 'Information Technology'
posIT['Result'] = 'Possible IT item'

#ventxt neg or ITResult is neg, IT? is pos
negIT = acct_fin.loc[(acct_fin['Ven_LD_HeaderITResult'] == 'NIT') & (acct_fin['IT?'] == 1) & (acct_fin['RE?'] == 0)].copy()
negIT['Type'] = 'Information Technology'
negIT['Result'] = 'Possible non-IT item'

#ventxt pos or REResult is pos, RE? is neg
posRE = acct_fin.loc[((acct_fin['Ven_LD_HeaderREResult'] == 'RE') & (acct_fin['RE?'] == 0) & (acct_fin['IT?'] == 0)) |\
                    ((acct_trans['VenTxtREResult'] == 'RE') & (acct_trans['RESuspect'] == 1))].copy()
posRE['Type'] = 'Real Estate'
posRE['Result'] = 'Possible RE Item'

#ventxt neg or REResult is neg, RE? is pos
negRE = acct_fin.loc[(acct_fin['Ven_LD_HeaderREResult'] == 'NRE') & (acct_fin['RE?'] == 1) & (acct_fin['IT?'] == 0)].copy()
negRE['Type'] = 'Real Estate'
negRE['Result'] = 'Possible non-RE item'

#ITsus = acct_fin.loc[((acct_fin['ITSuspect'] == 1) & (acct_fin['VenTxtITResult'] == 'IT')) | (acct_fin['ITBlocked'] == "Blocked")].copy()
#REsus = acct_fin.loc[((acct_fin['RESuspect'] == 1) & (acct_fin['VenTxtREResult'] == 'RE')) | (acct_fin['REBlocked'] == "Blocked")].copy()

#blocked entries
ITblocked = acct_fin.loc[(acct_fin['ITBlocked'] == "Blocked")].copy()
ITblocked['Type'] = 'Information Technology'
ITblocked['Result'] = 'Blocked GL Acct'
REblocked = acct_fin.loc[(acct_fin['REBlocked'] == "Blocked")].copy()
REblocked['Type'] = 'Real Estate'
REblocked['Result'] = 'Blocked GL Acct'


all = pd.concat([posIT, posRE, negIT, negRE, ITblocked, REblocked], ignore_index = True, sort = False).copy()
#all2 = acct_fin

#added blocked to suspect list
#sus = pd.concat([ITsus, REsus, all], sort = False) #, ITblocked, REblocked
all['Amt'] = all['Amt'].apply(lambda x: float(x))
#sus['Amt'] = sus['Amt'].apply(lambda x: float(x))

with pd.ExcelWriter(exportloc+datestamp+'.xlsx') as writer:
    #write dfs to excel file
    posIT.to_excel(writer, sheet_name = 'positiveIT', index = False)
    negIT.to_excel(writer, sheet_name = 'negativeIT', index = False)
    posRE.to_excel(writer, sheet_name = 'positiveRE', index = False)
    negRE.to_excel(writer, sheet_name = 'negativeRE', index = False)
    ITblocked.to_excel(writer, sheet_name = 'ITblocked', index = False)
    REblocked.to_excel(writer, sheet_name = 'REblocked', index = False)
    #all pos results
    all.to_excel(writer, sheet_name = 'all-posRE-IT', index = False)
    #all2.to_excel(writer, sheet_name =  'allresults', index = False)
print('Excel exported!')
