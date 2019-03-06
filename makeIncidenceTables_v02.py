#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:44:23 2018

@author: hanxinzhang

Log
 - 2018
 - Sep 30: to py36
 - Sep 30: Make age/sex stratifications
 - Oct 26: Adapt to Swedish data
 - Oct 30: to py27
 - Oct 31: Handle some bad samples
 - Nov 2: Count good/bad ICDs and total ICDs for each sex/age group.
 - Nov 6: Revise data structure, using defaultdict(Counter)
 - Nov 7: Multiprocessing
 - Nov 8: Adapt to MarketScan data
"""

import pickle
import re
import datetime
import numpy as np
import os
import multiprocessing
from collections import Counter
from collections import defaultdict

# Constants and Look-ups ------------------------------------------------------

INPATH = '/Volumes/Mac HDD/Hanxin/Data/allpatients/DX_OtherThanAz/'
OUTPATH = '/Volumes/Mac HDD/Hanxin/Data/allpatients/icdTables_strats_NoAz/'

MAX_LEN = 6000
START_DATE = datetime.date(2003, 1, 1)

YEARS = []
for d in range(MAX_LEN):
    nextDate = START_DATE + datetime.timedelta(days=d)
    YEARS.append(nextDate.year)
YEARS = np.array(YEARS)

DATES_TO_DAYS = {}
for d in range(MAX_LEN):
    nextDate = START_DATE + datetime.timedelta(days=d)
    nextDateStr = nextDate.strftime('%d%b%Y').upper()
    DATES_TO_DAYS[nextDateStr] = d

SEX_GROUPS = {'1': 'M', '2': 'F'}

AGE_GROUPS = {'0-20': list(range(21)),
              '21-40': list(range(21, 41)),
              '41-60': list(range(41, 61)),
              '61-': list(range(61, 130))}

AGE_GROUPS_INV = {}
for k, v in AGE_GROUPS.items():
    for ag in v:
        AGE_GROUPS_INV[ag] = k

PAT = re.compile('[|^]')

# -----------------------------------------------------------------------------

def findEnrlTotal(strt, end, sex, birth):

    enrl_dict = defaultdict(Counter)
    ageArray = YEARS - birth

    for d in range(strt, end):

        age_gp = AGE_GROUPS_INV[ageArray[d]]
        label = age_gp + ', ' + sex
        enrl_dict[label][d] = 1

    return enrl_dict


def parseRecord(recordList):
    
    sex = recordList[8]
    birth = int(recordList[4])
    diagList = recordList[11:]
    
    dayin = DATES_TO_DAYS[recordList[2]]
    dayout = DATES_TO_DAYS[recordList[3]]

    icd_dict = defaultdict(Counter)
    tot_dict = Counter()
    enrl_dict  = findEnrlTotal(dayin, dayout, sex, birth)
    
    for diagnosis in diagList:
        
        try:
            if diagnosis:
    
                icd9, ageStr, dayStr = diagnosis.split(':')                  
                g = AGE_GROUPS_INV[int(ageStr)]        
                lab = ', '.join([icd9, g, sex])
                icd_dict[lab][int(dayStr)] = 1
                tot_dict[sex + ', ' + g] += 1
                    
        except:
            print('Bad diagnosis!')
            print(diagnosis)

    return enrl_dict, tot_dict, icd_dict


def summarize(fpath):
    
    '''
    totalPatientNum: total number of patient entries.
    totalDiagNumDict: total number of diagnoses groupped by sex/age.
    icdDict: Day counts of all occuring ICDs groupped by sex/age.
    enrollDict: Day counts of enrollees groupped by sex/age
    '''

    totalDiagNumDict = Counter()
    totalPatientNum = 0
    icdDict = defaultdict(Counter)
    enrollDict = defaultdict(Counter)
    
    inFileName = fpath.split('/')[-1]
    outName = (OUTPATH + 'ICD_summ_' +  inFileName + '.pickle36')

    with open(fpath) as data:
        for i, recordLine in enumerate(data):
                
            recordLine = recordLine.strip()
            recordList = PAT.split(recordLine)
            
            try:
                enrollAdd, totalAdd, icdAdd = parseRecord(recordList)
            except:
                pid = recordList[0]
                print('Bad patient or unexpected situation happened! PID = ' + pid)
                print(recordLine)
                continue 
                    
            for k, v in icdAdd.items():
                icdDict[k] += v
            
            for k, v in enrollAdd.items():
                enrollDict[k] += v
                    
            totalDiagNumDict += totalAdd
            totalPatientNum += 1

            if i % 100000 == 0:
                print(str(i) + ' lines finished. Part: ' + inFileName)

    with open(outName, 'wb') as pic:
        pickle.dump({'totalPatientNum': totalPatientNum,
                     'totalDiagNumDict': totalDiagNumDict,
                     'icdDict': icdDict,
                     'enrollDict': enrollDict}, pic)
        

if __name__ == '__main__':
    
    inputFilesNames = filter(lambda fname: fname[0] == 'O', os.listdir(INPATH))
    inputFiles = list(map(lambda f: INPATH + f, inputFilesNames))
     
    poolOf_16_workers = multiprocessing.Pool(15)
    poolOf_16_workers.map(summarize, inputFiles)
    poolOf_16_workers.close()
    poolOf_16_workers.join()
            


