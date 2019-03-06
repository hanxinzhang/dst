#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:57:43 2018

@author: hanxinzhang
"""

import pickle
import numpy as np
import pymc3 as pm
import pandas as pd
import os

if not os.path.exists('results'):
    os.makedirs('results')

with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1/'
          'samplingOut/incidences.bpickle3', 'rb') as f:
    inciDict = pickle.load(f)
    incidencesSprAf1980 = inciDict['incidencesSpr']
    incidencesAutAf1980 = inciDict['incidencesAut']
    
    diseasesSprAf1980 = np.array(sorted(list(incidencesSprAf1980.keys())))
    diseasesAutAf1980 = np.array(sorted(list(incidencesAutAf1980.keys())))
    
    
with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1 before 1980/'
          'samplingOut/incidences.bpickle3', 'rb') as f:
    inciDict = pickle.load(f)
    incidencesSprBf1980 = inciDict['incidencesSpr']
    incidencesAutBf1980 = inciDict['incidencesAut']
    
    diseasesSprBf1980 = np.array(sorted(list(incidencesSprBf1980.keys())))
    diseasesAutBf1980 = np.array(sorted(list(incidencesAutBf1980.keys())))

with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1/'
          'models/mcmcModel_weekSpring.pickle', 'rb') as f:
    modelWeekSprAf1980 = pickle.load(f)
    
with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1/'
          'models/mcmcModel_weekAutumn.pickle', 'rb') as f:
    modelWeekAutAf1980 = pickle.load(f) 
    
with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1 before 1980/'
          'models/mcmcModel_weekSpring.pickle', 'rb') as f:
    modelWeekSprBf1980 = pickle.load(f)
    
with open('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
          'swedish analyses/v1 before 1980/'
          'models/mcmcModel_weekAutumn.pickle', 'rb') as f:
    modelWeekAutBf1980 = pickle.load(f) 

traceWeekSprAf1980 = pm.load_trace('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
                                   'swedish analyses/v1/'
                                   'samplingOut/weekTraceSpr', model=modelWeekSprAf1980)

traceWeekAutAf1980 = pm.load_trace('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
                                   'swedish analyses/v1/'
                                   'samplingOut/weekTraceAut', model=modelWeekAutAf1980)

traceWeekSprBf1980 = pm.load_trace('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
                                   'swedish analyses/v1 before 1980/'
                                   'samplingOut/weekTraceSpr', model=modelWeekSprBf1980)

traceWeekAutBf1980 = pm.load_trace('/Users/hanxinzhang/Documents/Rzhetsky lab/Trends-project/DST/'
                                   'swedish analyses/v1 before 1980/'
                                   'samplingOut/weekTraceAut', model=modelWeekAutBf1980)




SAMPLE_SIZE = 5000
np.random.seed(2018)
# CI_ALPHA = 1e-2

def isSig(interv):

    return (min(interv) > 1.) or (0. < max(interv) < 1.)

def findRRR(disAf, disBf, traceAf, traceBf, season,
            ci_alpha):
    
    rrrName = 'Relative Risk Ratio' + ' (' + season + ')'
    hpdName = str(100. * (1. - ci_alpha)) + '% C.I.' + ' (' + season + ')'
    sigName = season + ' Significant?'
    columnNames = ['Condition', 'System', rrrName, hpdName, sigName]
    rrrDF = pd.DataFrame(columns=columnNames)
    index = 0
    for iAf, dis in enumerate(disAf):
        if dis in disBf:
            iBf = np.where(dis == disBf)[0][0]
            
            postDisAf = traceAf[:, iAf]
            postDisBf = traceBf[:, iBf]
            
            sampled_rr_Af = np.random.choice(postDisAf, 
                                             size=SAMPLE_SIZE, 
                                             replace=True)
            
            sampled_rr_Bf = np.random.choice(postDisBf, 
                                             size=SAMPLE_SIZE, 
                                             replace=True)
            
            rrr = sampled_rr_Af / sampled_rr_Bf
            ci = pm.stats.hpd(rrr, alpha=ci_alpha)
            sig = 'Yes' if isSig(ci) else 'No'
            sys = dis.split('|')[1].split('$')[0]
            
            rrrDF.loc[index] = [dis, sys, rrr.mean(), ci, sig]
            
            index += 1
            
    return rrrDF

# -----------------------------------------------------------------------------
# 99% C.I.
    
rrrWeekSprDF = findRRR(diseasesSprAf1980, diseasesSprBf1980, 
                       traceWeekSprAf1980['rr'], traceWeekSprBf1980['rr'],
                       season='Spring', ci_alpha=0.01)

rrrWeekAutDF = findRRR(diseasesAutAf1980, diseasesAutBf1980, 
                       traceWeekAutAf1980['rr'], traceWeekAutBf1980['rr'],
                       season='Autumn', ci_alpha=0.01)

df_dst_day_bayes = pd.merge(rrrWeekSprDF,
                            rrrWeekAutDF,
                            on=['Condition', 'System']).dropna().reset_index(drop=True)
df_dst_day_sig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                        (df_dst_day_bayes['Autumn Significant?'] == 'No')]
df_dst_day_bothsig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                            (df_dst_day_bayes['Autumn Significant?'] == 'Yes')]


df_dst_day_bayes = df_dst_day_bayes.sort_values(by=['System', 'Condition'])
df_dst_day_bayes.reset_index(drop=True, inplace=True)

df_dst_day_sig_bayes = df_dst_day_sig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_sig_bayes.reset_index(drop=True, inplace=True)

df_dst_day_bothsig_bayes = df_dst_day_bothsig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_bothsig_bayes.reset_index(drop=True, inplace=True)
    
df_dst_day_bayes.to_csv('./results/allResults_Bayesian99.csv')
df_dst_day_sig_bayes.to_csv('./results/interestingResults_Bayesian99.csv')
df_dst_day_bothsig_bayes.to_csv('./results/interestingResultsBothSig_Bayesian99.csv')

# -----------------------------------------------------------------------------
# 95% C.I.
    
rrrWeekSprDF = findRRR(diseasesSprAf1980, diseasesSprBf1980, 
                       traceWeekSprAf1980['rr'], traceWeekSprBf1980['rr'],
                       season='Spring', ci_alpha=0.05)

rrrWeekAutDF = findRRR(diseasesAutAf1980, diseasesAutBf1980, 
                       traceWeekAutAf1980['rr'], traceWeekAutBf1980['rr'],
                       season='Autumn', ci_alpha=0.05)

df_dst_day_bayes = pd.merge(rrrWeekSprDF,
                            rrrWeekAutDF,
                            on=['Condition', 'System']).dropna().reset_index(drop=True)
df_dst_day_sig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                        (df_dst_day_bayes['Autumn Significant?'] == 'No')]
df_dst_day_bothsig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                            (df_dst_day_bayes['Autumn Significant?'] == 'Yes')]


df_dst_day_bayes = df_dst_day_bayes.sort_values(by=['System', 'Condition'])
df_dst_day_bayes.reset_index(drop=True, inplace=True)

df_dst_day_sig_bayes = df_dst_day_sig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_sig_bayes.reset_index(drop=True, inplace=True)

df_dst_day_bothsig_bayes = df_dst_day_bothsig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_bothsig_bayes.reset_index(drop=True, inplace=True)
    
df_dst_day_bayes.to_csv('./results/allResults_Bayesian95.csv')
df_dst_day_sig_bayes.to_csv('./results/interestingResults_Bayesian95.csv')
df_dst_day_bothsig_bayes.to_csv('./results/interestingResultsBothSig_Bayesian95.csv')

# -----------------------------------------------------------------------------
# 90% C.I.

rrrWeekSprDF = findRRR(diseasesSprAf1980, diseasesSprBf1980, 
                       traceWeekSprAf1980['rr'], traceWeekSprBf1980['rr'],
                       season='Spring', ci_alpha=0.1)

rrrWeekAutDF = findRRR(diseasesAutAf1980, diseasesAutBf1980, 
                       traceWeekAutAf1980['rr'], traceWeekAutBf1980['rr'],
                       season='Autumn', ci_alpha=0.1)

df_dst_day_bayes = pd.merge(rrrWeekSprDF,
                            rrrWeekAutDF,
                            on=['Condition', 'System']).dropna().reset_index(drop=True)
df_dst_day_sig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                        (df_dst_day_bayes['Autumn Significant?'] == 'No')]
df_dst_day_bothsig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                            (df_dst_day_bayes['Autumn Significant?'] == 'Yes')]

df_dst_day_bayes = df_dst_day_bayes.sort_values(by=['System', 'Condition'])
df_dst_day_bayes.reset_index(drop=True, inplace=True)

df_dst_day_sig_bayes = df_dst_day_sig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_sig_bayes.reset_index(drop=True, inplace=True)

df_dst_day_bothsig_bayes = df_dst_day_bothsig_bayes.sort_values(
    by=['System', 'Condition'])
df_dst_day_bothsig_bayes.reset_index(drop=True, inplace=True)

df_dst_day_bayes.to_csv('./results/allResults_Bayesian90.csv')
df_dst_day_sig_bayes.to_csv('./results/interestingResults_Bayesian90.csv')
df_dst_day_bothsig_bayes.to_csv('./results/interestingResultsBothSig_Bayesian90.csv')