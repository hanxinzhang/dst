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

INPATH = '/Volumes/Mac HDD/Hanxin/Data/swedish/icdTables/'
    
    
SE_UNIICD_TO_CONDITIONS = pd.read_csv('seUniICDToConds.csv', 
                                      index_col=0)

SE_UNIICD_TO_CONDITIONS_DICT = defaultdict(set)

for _, row in SE_UNIICD_TO_CONDITIONS.iterrows():
    uniicd = row['SE UniICD']
    sys = row['System']
    des = row['Description']
    lab = des + '|' + sys
    SE_UNIICD_TO_CONDITIONS_DICT[uniicd].add(lab)

multimap = []
for k, v in SE_UNIICD_TO_CONDITIONS_DICT.items():
    if len(v) != 1:
        multimap.append(k)

# -----------------------------------------------------------------------------
        

def toCondTable(part):
    
    icdDictPart = part['diseaseDict']
    totalDiagNumDictPart = part['totalNumDict']
    totalPatientNumPart = part['totalEnrollNum']
    
    condDictPart = defaultdict(Counter)
    badICD = Counter()
    
    for code, counter in icdDictPart.iteritems():
        icd, agegrp, sexgrp = code.split(', ')
        
        try:
            iver, icdc = icd.strip().split(':')
        except ValueError:
            conditions = []
            badICD[icd] += sum(counter.values())
            
      
        icd_trial0 = iver + ':' + icdc[:6]
        icd_trial1 = iver + ':' + icdc[:5]
        icd_trial2 = iver + ':' + icdc[:4]
        icd_trial3 = iver + ':' + icdc[:3]
        
        if icd_trial0 in SE_UNIICD_TO_CONDITIONS_DICT:         
            conditions = SE_UNIICD_TO_CONDITIONS_DICT[icd_trial0]       
        elif icd_trial1 in SE_UNIICD_TO_CONDITIONS_DICT:         
            conditions = SE_UNIICD_TO_CONDITIONS_DICT[icd_trial1]
        elif icd_trial2 in SE_UNIICD_TO_CONDITIONS_DICT:         
            conditions = SE_UNIICD_TO_CONDITIONS_DICT[icd_trial2]  
        elif icd_trial3 in SE_UNIICD_TO_CONDITIONS_DICT:         
            conditions = SE_UNIICD_TO_CONDITIONS_DICT[icd_trial3]           
        else:      
            conditions = []
            badICD[icd] += sum(counter.values())
              
        for cond in conditions:
            label = cond.capitalize() + '$' + sexgrp.strip() + '$'+  agegrp.strip()
            condDictPart[label] += counter
    
    return (condDictPart, badICD,
            totalDiagNumDictPart, totalPatientNumPart)
    


if __name__ == '__main__':  
    
    inparList = os.listdir(INPATH)
    
    COND_DICT = defaultdict(Counter)
    BAD_ICD = Counter()
    TOTAL_DIAG_NUM_DICT = Counter()
    TOTAL_PATIENT_NUM = []
    
    for inpar in inparList:
        
        print('Finishing ' + inpar)
        
        with open(INPATH + inpar, 'rb') as f:
            part = pickle.load(f)
            
            (condDictPart, badICD,
             totalDiagNumDictPart, totalPatientNumPart) = toCondTable(part)
            
            for cond, count in condDictPart.items():
                COND_DICT[cond] += count
            
            BAD_ICD += badICD     
            TOTAL_DIAG_NUM_DICT += totalDiagNumDictPart     
            TOTAL_PATIENT_NUM.append(totalPatientNumPart)
            
            
    with open('SE_COND_ENROLL_SUMMARY.bpickle3', 'wb') as f:
        pickle.dump({'COND_DICT': COND_DICT}, f)
        
    with open('SE_OTHER_STATISTICS.bpickle3', 'wb') as f:
        pickle.dump({'BAD_ICD': BAD_ICD,
                     'TOTAL_DIAG_NUM_DICT': TOTAL_DIAG_NUM_DICT,
                     'TOTAL_PATIENT_NUM': TOTAL_PATIENT_NUM}, f)
            
            
