#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:15:49 2018

@author: hanxinzhang
"""


import pickle
import pandas as pd
import os
from collections import Counter
from collections import defaultdict


# -----------------------------------------------------------------------------

INPATH = '/Volumes/Mac HDD/Hanxin/Data/inpatients/icdTables_strats_NoAz/'
    
DAY_ENROLL_BY_GROUP = defaultdict(Counter)
COND_SUMM = defaultdict(Counter)
TOTAL_DIAG_SUMM = Counter()
TOTAL_PATIENT_NUM = 0
    
US_ICD9_TO_CONDITIONS = pd.read_csv('usICD9toConds.csv', 
                                    #usecols=[1, 2, 3],
                                    index_col=0)

US_ICD9_TO_CONDITIONS_DICT = defaultdict(set)

for _, row in US_ICD9_TO_CONDITIONS.iterrows():
    i9 = row['US ICD9']
    sys = row['System']
    des = row['Description']
    lab = des + '|' + sys
    US_ICD9_TO_CONDITIONS_DICT[i9].add(lab)

multimap = []
for k, v in US_ICD9_TO_CONDITIONS_DICT.items():
    if len(v) != 1:
        multimap.append(k)

# -----------------------------------------------------------------------------
        

def toCondTable(part):
    
    enrollDictPart = part['enrollDict']
    icdDictPart = part['icdDict']
    totalDiagNumDictPart = part['totalDiagNumDict']
    totalPatientNumPart = part['totalPatientNum']
    
    condDictPart = defaultdict(Counter)
    badICD = Counter()
    
    for code, counter in icdDictPart.items():
        icd, agegrp, sexgrp = code.split(',')
        icdf = icd.strip().replace('.', '')
        
        try:
            icdh, icdt = icd.split('.')
        except ValueError:
            icdh = icd[:3]
            icdt = icd[3:]
      
        icd_trial0 = icdf
        icd_trial1 = icdh + icdt[0]
        icd_trial2 = icdh
        
        if icd_trial0 in US_ICD9_TO_CONDITIONS_DICT:         
            conditions = US_ICD9_TO_CONDITIONS_DICT[icd_trial0]       
        elif icd_trial1 in US_ICD9_TO_CONDITIONS_DICT:         
            conditions = US_ICD9_TO_CONDITIONS_DICT[icd_trial1]
        elif icd_trial2 in US_ICD9_TO_CONDITIONS_DICT:         
            conditions = US_ICD9_TO_CONDITIONS_DICT[icd_trial2]  
        else:      
            conditions = []
            badICD[icd] += sum(counter.values())
              
        for cond in conditions:
            label = cond.capitalize() + '$' + sexgrp.strip() + '$'+  agegrp.strip()
            condDictPart[label] += counter
    
    return (condDictPart, enrollDictPart, badICD,
            totalDiagNumDictPart, totalPatientNumPart)
    


if __name__ == '__main__':  
    
    inparList = os.listdir(INPATH)
    
    COND_DICT = defaultdict(Counter)
    ENROLL_DICT = defaultdict(Counter)
    BAD_ICD = Counter()
    TOTAL_DIAG_NUM_DICT = Counter()
    TOTAL_PATIENT_NUM = 0
    
    for inpar in inparList:
        
        print('Finishing ' + inpar)
        
        with open(INPATH + inpar, 'rb') as f:
            part = pickle.load(f)
            
            (condDictPart, enrollDictPart, badICD,
             totalDiagNumDictPart, totalPatientNumPart) = toCondTable(part)
            
            for cond, count in condDictPart.items():
                COND_DICT[cond] += count
                
            for grp, count in enrollDictPart.items():
                ENROLL_DICT[grp] += count
            
            BAD_ICD += badICD     
            TOTAL_DIAG_NUM_DICT += totalDiagNumDictPart     
            TOTAL_PATIENT_NUM += totalPatientNumPart
    
    with open('OTHER_STATISTICS.bpickle3', 'wb') as f:
        pickle.dump({'BAD_ICD': BAD_ICD,
                     'TOTAL_DIAG_NUM_DICT': TOTAL_DIAG_NUM_DICT,
                     'TOTAL_PATIENT_NUM': TOTAL_PATIENT_NUM}, f)
            
    with open('INP_COND_ENROLL_SUMMARY.bpickle3', 'wb') as f:
        pickle.dump({'COND_DICT': COND_DICT,
                     'ENROLL_DICT': ENROLL_DICT}, f)
            
            