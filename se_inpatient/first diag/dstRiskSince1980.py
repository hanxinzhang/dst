#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:10:03 2018

@author: hanxinzhang
"""

import numpy as np
import pandas as pd
import datetime
import pytz
import holidays
import pickle
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import os
from textwrap import wrap
from statsmodels.stats.proportion import proportion_confint
from collections import defaultdict, Counter
# import sys

OUTPATH = './results_SwedenSince1980'

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

os.chdir(OUTPATH)

for method in ['Bayesian', 'Frequentist']:

    if not os.path.exists('plots%sMerged' % method):
        os.makedirs('plots%sMerged' % method)

if not os.path.exists('models'):
    os.makedirs('models')
    
if not os.path.exists('samplingOut'):
    os.makedirs('samplingOut')
    
os.chdir('..')

# Data and constants ----------------------------------------------------------

cond_df = pd.read_csv('toandrey.txt.incidences.csv', index_col=0)
COND_DICT = defaultdict(Counter)

for i in range(1, cond_df.shape[1]):
    col = cond_df.iloc[:, i]
    for d, c in enumerate(col):
        if c != 0:
            COND_DICT[col.name][d] = c

# with open('SE_COND_ENROLL_SUMMARY.bpickle3', 'rb') as f:
#     summDict = pickle.load(f)
#     COND_DICT = summDict['COND_DICT']

START_DATE = datetime.datetime(1968, 1, 1)
START_DATE_DST = datetime.datetime(1980, 1, 1)
END_DATE = datetime.datetime(2017, 12, 31)

SWEDEN_TZ = pytz.timezone('Europe/Stockholm')

ALL_DST_DATES = list(filter(lambda dt: START_DATE_DST < dt < END_DATE,
                            SWEDEN_TZ._utc_transition_times))

DST_START = ALL_DST_DATES[::2]
DST_END = ALL_DST_DATES[1::2]

assert np.all(np.array([d.weekday() for d in DST_START]) == 6)
assert np.all(np.array([d.weekday() for d in DST_END]) == 6)

DST_START_DAYS_FROM = np.array([(dt - START_DATE).days for dt in DST_START]) + 1
DST_END_DAYS_FROM = np.array([(dt - START_DATE).days for dt in DST_END]) + 1

ALL_SE_HOLIDAYS = holidays.Sweden(years=range(1980,2018))

EASTER_DATES = []
for dt, hl in ALL_SE_HOLIDAYS.items():
    if hl == 'Påskdagen, Söndag':
        EASTER_DATES.append(dt)
        
assert np.all(np.array([d.weekday() for d in EASTER_DATES]) == 6)

EASTER_DAYS_FROM = np.array([(dt - START_DATE.date()).days 
                             for dt in EASTER_DATES]) + 1
    
# Data and constants before 1980 DST effected----------------------------------
    
ALL_SE_HOLIDAYS_BEFORE_1980 = holidays.Sweden(years=range(1968, 1980))
    
EASTER_DATES_BEFORE_1980 = []
for dt, hl in ALL_SE_HOLIDAYS_BEFORE_1980.items():
    if hl == 'Påskdagen, Söndag':
        EASTER_DATES_BEFORE_1980.append(dt)

assert np.all(np.array([d.weekday() for d in EASTER_DATES_BEFORE_1980]) == 6)

EASTER_DAYS_FROM_BEFORE_1980 = np.array([(dt - START_DATE.date()).days 
                                         for dt in EASTER_DATES_BEFORE_1980]) + 1
    
PSEUDO_DST_START_DAYS_FROM = DST_START_DAYS_FROM[:11] - 365 * 12 - 2
PSEUDO_DST_END_DAYS_FROM = DST_END_DAYS_FROM[:11] - 365 * 12 - 2

# -----------------------------------------------------------------------------
    
NUM_ITERATIONS = 2000  # int(sys.argv[1])
EPS = np.finfo(float).eps
PRIOR_FLATNESS = 5
CI_ALPHA = 1e-2
OBS_BOUND = 10


def isSig(interv):

    return (min(interv) > 1.) or (0. < max(interv) < 1.)


def findCI(x1, x2, alpha):
    
    n = x1 + x2
    theta_lo, theta_up = proportion_confint(x1, n, 
                                            alpha=alpha, 
                                            method='wilson')
    
    rr_lo = theta_lo / (1. - theta_lo)
    rr_up = theta_up / (1. - theta_up)
    
    ci = np.sort([2. * rr_lo, 2. * rr_up])
       
    return {'CI': ci}


def relativeRiskFrequentist(xbefore, x0, xafter, n_tests, n_selection,
                            ci_alpha=CI_ALPHA, correction=None, days=7, adj=True):
    
    selected = 0
    
    # 1: adjustment factor (Sunday) 2: stay the same (Monday to Saturday)
    factor1s = (days * 24) / (days * 24 - 1)
    factor2s = 1.
    factor1a = (days * 24) / (days * 24 + 1)
    factor2a = 1.
    
    if correction == 'Spring':
        x0 = (x0 * factor1s) if adj else (x0 * factor2s)
    
    if correction == 'Autumn':
        x0 = (x0 * factor1a) if adj else (x0 * factor2a)
        
    x_obs = x0
    x_double_exp = xbefore + xafter
    
    ci = findCI(x1=x_obs, x2=x_double_exp, alpha=ci_alpha)
    
    ci_bonf = findCI(x1=x_obs, x2=x_double_exp, alpha=ci_alpha/n_tests)
    
    if isSig(ci['CI']):
        selected += 1
        ci_fcr = findCI(x1=x_obs, x2=x_double_exp, 
                        alpha=n_selection * ci_alpha/n_tests)
    else:
        ci_fcr = ci
    
    rr = x_obs / (0.5 * x_double_exp)
    
    return rr, ci, ci_bonf, ci_fcr, selected


def relativeRiskTraceBayesian(xbefore, x0, xafter,
                              correction=None, days=7):

    # Functions handling machine precision issues
    def clip0to1(rv):

        return tt.clip(rv, EPS, 1. - EPS)

    def clip0toInf(rv):

        return tt.clip(rv, EPS, rv)

    if isinstance(x0, float):
        modelShape = 1
    else:
        modelShape = x0.shape

    # 1: collapsed factor, 2: uncollapsed factor
    factor1s = (days * 24) / (days * 24 - 1)
    factor2s = np.array([24. / 23.] + [1. for _ in range(days - 1)])
    factor1a = (days * 24) / (days * 24 + 1)
    factor2a = np.array([24. / 25.] + [1. for _ in range(days - 1)])

    collapsed = isinstance(x0, float) or (len(modelShape) == 1)
    if correction == 'Spring':
        x0 = (x0 * factor1s) if collapsed else (x0 * factor2s)

    if correction == 'Autumn':
        x0 = (x0 * factor1a) if collapsed else (x0 * factor2a)

    modelName = (OUTPATH + '/models/mcmcModel_' + ('week' if collapsed else 'day')
                 + correction + '.bpkl3')

    try:
        '''
        Try to load an existing model.
        This ensures reproducible results of RR estimates.
        Because some initialization issues of theano variables, 
        PyMC3 models with identical observed inputs but initialized independently 
        cannot produce reproducible traces even with the same random seed.
        '''

        with open(modelName, 'rb') as buff:
            basic_model = pickle.load(buff)

        print('\x1b[6;30;42m' + 'Load a pickled model' + '\x1b[0m')

    except:
        '''
        If there is no existing model,
        make a new model and pickle it for future reference.
        '''

        print('\x1b[6;30;42m' + 'Making a new model...' + '\x1b[0m')

        basic_model = pm.Model()

        with basic_model:

            # Hierarchical prior for cc to pool information across all conditions.
            # This is a bayesian way to deal with multiple comparisons.
            # cc is a the curvature of trend.
            # rr = cc / E[cc] is the redefined relative risk.
            # E[rr] = 1, according with our prior belief.

            cc_mu = pm.HalfCauchy('cc_mu', PRIOR_FLATNESS)
            cc_sd = pm.HalfCauchy('cc_sd', PRIOR_FLATNESS)

            lbefore = pm.HalfCauchy('lbefore', 
                                    PRIOR_FLATNESS,  
                                    shape=modelShape) * 100.
            lafter = pm.HalfCauchy('lafter',
                                   PRIOR_FLATNESS,  
                                   shape=modelShape) * 100.
            cc = pm.Gamma('cc',
                          mu=cc_mu,
                          sd=clip0toInf(cc_sd),
                          shape=modelShape)

            l0 = pm.Deterministic('p0', 0.5 * cc * (lbefore + lafter))

            x_obs_b = pm.Poisson('x_obs_b',
                                  mu=clip0toInf(lbefore),
                                  observed=xbefore,
                                  shape=modelShape)
            x_obs_a = pm.Poisson('x_obs_a',
                                 mu=clip0toInf(lafter),
                                 observed=xafter,
                                 shape=modelShape)
            x_obs_0 = pm.Poisson('x_obs_0',
                                 mu=clip0toInf(l0),
                                 observed=x0,
                                 shape=modelShape)

            rr = pm.Deterministic('rr', cc / cc_mu)

        for RV in basic_model.basic_RVs:
            # Print initial loss
            print(RV.name, RV.logp(basic_model.test_point))

        with open(modelName, 'wb') as buff:
            pickle.dump(basic_model, buff)

    '''
    Sampling ...
    '''
    with basic_model:
        trace = pm.sample(draws=NUM_ITERATIONS,
                          tune=NUM_ITERATIONS,
                          init='advi',
                          cores=4,
                          chains=4,
                          n_init=200000,
                          random_seed=2018,
                          progressbar=True)

    return trace


def relativeRiskBayesian(trace, ci_alpha=CI_ALPHA):

    gelman_rubin = pm.diagnostics.gelman_rubin(trace)['rr']

    relativeRisk = trace['rr'].mean(axis=0)
    ci = pm.stats.hpd(trace['rr'], alpha=ci_alpha)

    return (relativeRisk,
            {'CI': ci, 'Gelman-Rubin': gelman_rubin})
    
    
def doFrequentist(dis,
                  system,
                  incidences_dst_before,
                  incidences_dst_after,
                  incidences_dst_0,
                  days,
                  n_diseases,
                  n_selection_day,
                  n_selection_week,
                  correction,
                  methodFunc):
    
    day_selected = 0
    week_selected = 0
    
    dfEntry, plotDataDictEntry = np.nan, np.nan
    
    rr_array = np.zeros(days)
    errorBars = np.zeros((2, days))
    ymax = -np.inf
    ymin = np.inf
    
    try:             
        for dayOfWeek in range(0, days):
            
            xbefore = incidences_dst_before[dayOfWeek]
            xafter = incidences_dst_after[dayOfWeek]
            x0 = incidences_dst_0[dayOfWeek]
            
            rr, conf, bonf_conf, fcr_conf, dsel = methodFunc(xbefore, 
                                                             x0,
                                                             xafter,
                                                             n_tests=n_diseases * days,
                                                             n_selection=n_selection_day,
                                                             correction=correction, 
                                                             days=1,
                                                             adj=True if dayOfWeek==0 else False)
            
            day_selected += dsel
            
            rr_array[dayOfWeek] = rr
            errorBars[:, dayOfWeek] = np.abs(fcr_conf['CI'] - rr)
            
            ymax = max(fcr_conf['CI'][1], ymax)
            ymin = min(fcr_conf['CI'][0], ymin)
            
            
        (rrCollapse, confCollapse, bonf_confCollapse, 
         fcr_confCollapse, wsel) = methodFunc(incidences_dst_before.sum(), 
                                              incidences_dst_0.sum(), 
                                              incidences_dst_after.sum(), 
                                              n_tests=n_diseases,
                                              n_selection=n_selection_week,
                                              correction=correction, 
                                              days=days)
        
        week_selected += wsel
        
    except:
        print('\x1b[0;37;41m' + 
              'Something happened when doing Frequentist or half-Bayesian modeling!' + 
              '\x1b[0m')
    else:
      
        errorBarsClps = np.abs(fcr_confCollapse['CI'] - rrCollapse)
        ymaxClps = fcr_confCollapse['CI'][1]
        yminClps = fcr_confCollapse['CI'][0]
        
        sig = 'Yes' if isSig(fcr_confCollapse['CI']) else 'No'
    
        dfEntry = [dis, system, rrCollapse, confCollapse, fcr_confCollapse, sig]
        
        plotDataDictEntry = (correction,
                             np.append(rr_array, [rrCollapse]),
                             np.append(errorBars, errorBarsClps[:,np.newaxis], axis=1),
                             min(ymin, yminClps),
                             max(ymax, ymaxClps))
    
    return dfEntry, plotDataDictEntry, day_selected, week_selected



def doBayesianTensor(incidencesDict, days, correction):

    diseases = np.array(sorted(list(incidencesDict.keys())))

    x0_tensor = np.array([incidencesDict[k][2] for k in diseases])
    xbefore_tensor = np.array([incidencesDict[k][3] for k in diseases])
    xafter_tensor = np.array([incidencesDict[k][4] for k in diseases])

    dayTrace = relativeRiskTraceBayesian(xbefore_tensor,
                                         x0_tensor,
                                         xafter_tensor,
                                         correction=correction,
                                         days=days)

    weekTrace = relativeRiskTraceBayesian(xbefore_tensor.sum(axis=1),                                     
                                          x0_tensor.sum(axis=1),
                                          xafter_tensor.sum(axis=1),
                                          correction=correction,
                                          days=days)

    return dayTrace, weekTrace


def tracesToResults(incidencesDict, correction,
                    dayTrace, weekTrace, ci_alpha=CI_ALPHA):

    diseases = np.array(sorted(list(incidencesDict.keys())))
    n_diseases = len(diseases)
    dis_indices = np.array([incidencesDict[k][0] for k in diseases])
    systems = np.array([incidencesDict[k][1] for k in diseases])

    try:
        rr, cred = relativeRiskBayesian(dayTrace, ci_alpha=ci_alpha)
        rrClps, credClps = relativeRiskBayesian(weekTrace, ci_alpha=ci_alpha)

        errorBars = np.abs(cred['CI'] - rr[..., np.newaxis])
        ymax = cred['CI'][..., 1].max(axis=1)
        ymin = cred['CI'][..., 0].min(axis=1)

        errorBarsClps = np.abs(credClps['CI'] - rrClps[..., np.newaxis])
        ymaxClps = credClps['CI'][..., 1]
        yminClps = credClps['CI'][..., 0]

        plotDataDict = {dis: (correction,
                              np.append(rr[i], [rrClps[i]]),
                              np.append(errorBars[i, ...].transpose(),
                                        errorBarsClps[i, ...][..., np.newaxis],
                                        axis=1),
                              min(ymin[i], yminClps[i]),
                              max(ymax[i], ymaxClps[i])) for i, dis in enumerate(diseases)}

        sig = ['Yes' if isSig(credClps['CI'][i])
               else 'No' for i in range(n_diseases)]
        df_dict = {dis_indices[i]: [diseases[i],
                                    systems[i],
                                    rrClps[i],
                                    {'CI': credClps['CI'][i],
                                     'Gelman-Rubin': credClps['Gelman-Rubin'][i]},
                                    sig[i]] for i in range(n_diseases)}

    except:
        df_dict = np.nan
        plotDataDict = np.nan

    return df_dict, plotDataDict


def cleanIncidences(incidDict, bound=OBS_BOUND):

    cleaned = {}
    for k, v in incidDict.items():
        midGreaterThan = np.all(v[2] >= bound)
        befGreaterThan = np.all(v[3] >= bound)
        aftGreaterThan = np.all(v[4] >= bound)
        if all([midGreaterThan, befGreaterThan, aftGreaterThan]):
            cleaned[k] = v

    return cleaned


def plotBarsMerged(plotDataDict1, plotDataDict2, days, method, ci_lab):

    disGroup1 = list(plotDataDict1.keys())
    disGroup2 = list(plotDataDict2.keys())
    overlapGroup = [dis for dis in disGroup1 if dis in disGroup2]

    days += 1
    colors = np.append(np.repeat('lightsteelblue', 7), 'teal')
    
    plt.rcParams.update({'font.size': 10})
    
    for dis in overlapGroup:

        notNaN = isinstance(plotDataDict1[dis], tuple) and isinstance(
            plotDataDict2[dis], tuple)
        condSys, sex, age = dis.split('$')
        sex = {'M': 'Male', 'F': 'Female'}[sex]
        cond, sys = condSys.split('|')
        
        saveName = dis.replace(' ', '_').replace('/', 'or')

        if notNaN:
            
            plt.tight_layout()
            fig = plt.figure(figsize=(6, 8))
            ax = fig.add_subplot(111)   
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            
            ax1.bar(x=range(0, days),
                    height=plotDataDict1[dis][1],
                    color=colors,
                    yerr=plotDataDict1[dis][2],
                    capsize=7)
            ax1.axhline(1., linestyle='--', color='gray')
            ax1.set_ylim(plotDataDict1[dis][3] - 0.1, plotDataDict1[dis][4] + 0.1)
            ax1.set_xticks(range(0, days))
            ax1.set_xticklabels([])
            ax1.set_title(plotDataDict1[dis][0], size=10, y=0.9)
            
            ax2.bar(x=range(0, days),
                    height=plotDataDict2[dis][1],
                    color=colors,
                    yerr=plotDataDict2[dis][2],
                    capsize=7)
            ax2.axhline(1., linestyle='--', color='gray')
            ax2.set_ylim(plotDataDict2[dis][3] - 0.1, plotDataDict2[dis][4] + 0.1)
            ax2.set_xticks(range(0, days))
            ax2.set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Week'])
            ax2.set_title(plotDataDict2[dis][0], size=10, y=0.9)
            
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, 
                           bottom=False, left=False, right=False)
           
            ax.set_ylabel('Relative Risk ' + '(%s)' % ci_lab)
            ax.yaxis.set_label_coords(-0.12, 0.5)
            fig.suptitle('\n'.join(wrap(cond + ', ' + sys)))
            fig.text(0.14, 0.9, sex + ', ' + age, size=10)

            figName = OUTPATH + '/plots' + method + 'Merged/' + saveName + '.pdf'
            fig.savefig(figName, format='pdf')
            plt.close(fig)
            # break


def sampleTraces(day_of_interest, 
                 day_of_correction=None,
                 diseasesDict=None,
                 correction=None,
                 days=7,
                 method=None,
                 df=None,
                 plotDataDict=None):

    all_groups = sorted(list(diseasesDict.keys()))

    if method != 'Frequentist':

        print('Apply the Bayesian Method...')

    incidencesDict = {}

    for studyGrp_index, studyGrp in enumerate(all_groups):

        try:
            condSysName, sex, age = studyGrp.split('$')
            #print('Stratified!')
        except:
            condSysName = studyGrp
            #print('Pooled!')
                
                

        observations = diseasesDict[studyGrp]
        
        condName, sysName = condSysName.split('|')
            
        incidences_dst_0 = np.zeros(days)
        incidences_dst_before = np.zeros(days)
        incidences_dst_after = np.zeros(days)

        for i, di in enumerate(day_of_interest):
            
            try:
                dc = day_of_correction[i]
            except:
                dc = np.inf
            
            if dc == di:
                continue
                
            if correction == 'Spring':
                 
                if abs(dc - di) != 14:
                    # Condition day index starts from 0
                    index_0 = np.array(range(di, di + days)) - 1 
                    index_before = np.array(range(di - 14, di - 14 + days)) - 1
                    index_after = np.array(range(di + 14, di + 14 + days)) - 1
                else:
                    index_0 = np.array(range(di, di + days)) - 1
                    index_before = np.array(range(di - 21, di - 21 + days)) - 1
                    index_after = np.array(range(di + 21, di + 21 + days)) - 1

            elif correction == 'Autumn':
                
                if abs(dc - di) >= 21:
                    index_0 = np.array(range(di, di + days)) - 1
                    index_before = np.array(range(di - 14, di - 14 + days)) - 1
                    index_after = np.array(range(di + 14, di + 14 + days)) - 1
                else:
                    index_0 = np.array(range(di, di + days)) - 1
                    index_before = np.array(range(di - 7, di - 7 + days)) - 1
                    index_after = np.array(range(di + 7, di + 7 + days)) - 1

            else:

                index_0 = np.array(range(di, di + days)) - 1
                index_before = np.array(range(di - 14, di - 14 + days)) - 1
                index_after = np.array(range(di + 14, di + 14 + days)) - 1

            incidences_dst_0 += np.array([observations[d] for d in index_0])
            incidences_dst_before += np.array([observations[d] for d in index_before])
            incidences_dst_after += np.array([observations[d] for d in index_after])
            
        midGreaterThan = np.all(incidences_dst_before >= OBS_BOUND)
        befGreaterThan = np.all(incidences_dst_after >= OBS_BOUND)
        aftGreaterThan = np.all(incidences_dst_0 >= OBS_BOUND)
        qc = all([midGreaterThan, befGreaterThan, aftGreaterThan])
        
        if qc:

            incidencesDict[studyGrp] = (studyGrp_index,
                                        sysName,
                                        incidences_dst_0,
                                        incidences_dst_before,
                                        incidences_dst_after)
                
    if method != 'Bayesian':
        
        n_diseases = len(incidencesDict)
        day_selected = 0
        week_selected = 0
        
        for studyGrp, v in incidencesDict.items():
            
            (studyGrp_index,
             sysName,
             incidences_dst_0,
             incidences_dst_before,
             incidences_dst_after) = v

            if method == 'Frequentist':
                
                (_, 
                 _, 
                 dsel, 
                 wsel) = doFrequentist(studyGrp,
                                       sysName,
                                       incidences_dst_before,
                                       incidences_dst_after,
                                       incidences_dst_0,
                                       days,
                                       n_diseases,
                                       n_selection_day=1,
                                       n_selection_week=1,
                                       correction=correction,
                                       methodFunc=relativeRiskFrequentist)
                
                day_selected += dsel
                week_selected += wsel
        
        print('Day selection = ', day_selected)
        print('Week selection = ', week_selected)
        
        for studyGrp, v in incidencesDict.items():
            
            (studyGrp_index,
             sysName,
             incidences_dst_0,
             incidences_dst_before,
             incidences_dst_after) = v
             
            if method == 'Frequentist':
                
                (dfEntry, 
                 plotDataDictEntry, 
                 _, 
                 _) = doFrequentist(studyGrp,
                                    sysName,
                                    incidences_dst_before,
                                    incidences_dst_after,
                                    incidences_dst_0,
                                    days,
                                    n_diseases,
                                    n_selection_day=day_selected,
                                    n_selection_week=week_selected,
                                    correction=correction,
                                    methodFunc=relativeRiskFrequentist)
                
                df.loc[studyGrp_index] = dfEntry
                plotDataDict[studyGrp] = plotDataDictEntry
           
        return df, plotDataDict    
    
    if method != 'Frequentist':
        dayTrace, weekTrace = doBayesianTensor(incidencesDict,
                                               days, correction)
    
    return incidencesDict, dayTrace, weekTrace


def wrapResults(incidencesDict, correction,
                dayTrace, weekTrace, ci_alpha=CI_ALPHA):

    ci_name = str(100. * (1. - ci_alpha)) + '% C.I.'
    columnNames = ['Condition', 'System', 'Relative Risk',
                   ci_name, 'Significant?']

    if correction == 'Spring':
        columnNames = ['Condition', 'System', 'Relative Risk (Spring)',
                       ci_name + ' (Spring)', 'Spring Significant?']

    if correction == 'Autumn':
        columnNames = ['Condition', 'System', 'Relative Risk (Autumn)',
                       ci_name + ' (Autumn)', 'Autumn Significant?']

    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}

    df_dict, plotDataDict = tracesToResults(
        incidencesDict, correction, dayTrace, weekTrace, ci_alpha=ci_alpha)

    for index in list(df_dict.keys()):
        df.loc[index] = df_dict[index]

    return df, plotDataDict


# -----------------------------------------------------------------------------
    
if __name__ == '__main__':
    
# Frequentist's Method (all diseases ...) -------------------------------------
    
    ci_name = str(100. * (1. - CI_ALPHA)) + '% C.I.'   
    
    columnNames = ['Condition', 'System', 
                   'Relative Risk (Spring)', 
                   ci_name + ' (Spring)', 
                   'False Coverage Rate Corrected ' + ci_name + ' (Spring)', 
                   'Spring Significant?']
    
        
    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}
    
    df_dst_start_frequ, plotDataDict_start_frequ = sampleTraces(day_of_interest=DST_START_DAYS_FROM,
                                                                day_of_correction=EASTER_DAYS_FROM,
                                                                diseasesDict=COND_DICT,
                                                                correction='Spring',
                                                                days=7,
                                                                method='Frequentist',
                                                                df=df,
                                                                plotDataDict=plotDataDict)
    
    
    columnNames = ['Condition', 'System', 
                   'Relative Risk (Autumn)', 
                   ci_name + ' (Autumn)', 
                   'False Coverage Rate Corrected ' + ci_name + ' (Autumn)', 
                   'Autumn Significant?']
    
    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}
    
    df_dst_end_frequ, plotDataDict_end_frequ = sampleTraces(day_of_interest=DST_END_DAYS_FROM,
                                                            diseasesDict=COND_DICT,
                                                            correction='Autumn',
                                                            days=7,
                                                            method='Frequentist',
                                                            df=df,
                                                            plotDataDict=plotDataDict)
    
    df_dst_day_frequ = pd.merge(df_dst_start_frequ,
                                df_dst_end_frequ,
                                on=['Condition', 'System']).dropna().reset_index(drop=True)
    df_dst_day_sig_frequ = df_dst_day_frequ[(df_dst_day_frequ['Spring Significant?'] == 'Yes') &
                                            (df_dst_day_frequ['Autumn Significant?'] == 'No')]
    df_dst_day_bothsig_frequ = df_dst_day_frequ[(df_dst_day_frequ['Spring Significant?'] == 'Yes') &
                                                (df_dst_day_frequ['Autumn Significant?'] == 'Yes')]

    df_dst_day_frequ = df_dst_day_frequ.sort_values(by=['System', 'Condition'])
    df_dst_day_frequ.reset_index(drop=True, inplace=True)

    df_dst_day_sig_frequ = df_dst_day_sig_frequ.sort_values(by=['System', 'Condition'])
    df_dst_day_sig_frequ.reset_index(drop=True, inplace=True)

    df_dst_day_bothsig_frequ = df_dst_day_bothsig_frequ.sort_values(by=['System', 'Condition'])
    df_dst_day_bothsig_frequ.reset_index(drop=True, inplace=True)
    
    plotBarsMerged(plotDataDict_start_frequ, plotDataDict_end_frequ, days=7, 
                   method='Frequentist', ci_lab='FCR Corrected ' + ci_name)
    
    df_dst_day_frequ.to_csv(OUTPATH + '/allResults_Frequentist.csv')
    df_dst_day_sig_frequ.to_csv(OUTPATH + '/interestingResults_Frequentist.csv')
    df_dst_day_bothsig_frequ.to_csv(OUTPATH + '/interestingResultsBothSig_Frequentist.csv')


# Bayesian Method (all diseases ...) ------------------------------------------

    incidencesSpr, dayTraceSpr, weekTraceSpr = sampleTraces(day_of_interest=DST_START_DAYS_FROM,
                                                            day_of_correction=EASTER_DAYS_FROM,
                                                            diseasesDict=COND_DICT,
                                                            correction='Spring',
                                                            days=7,
                                                            method='Bayesian')

    incidencesAut, dayTraceAut, weekTraceAut = sampleTraces(day_of_interest=DST_END_DAYS_FROM,
                                                            diseasesDict=COND_DICT,
                                                            correction='Autumn',
                                                            days=7,
                                                            method='Bayesian')

    CI_ALPHA = 1e-2
    df_dst_strt_bayes, plotDataDict_strt_bayes = wrapResults(incidencesSpr,
                                                             'Spring',
                                                             dayTraceSpr,
                                                             weekTraceSpr,
                                                             ci_alpha=CI_ALPHA)

    df_dst_end_bayes, plotDataDict_end_bayes = wrapResults(incidencesAut,
                                                           'Autumn',
                                                           dayTraceAut,
                                                           weekTraceAut,
                                                           ci_alpha=CI_ALPHA)

    df_dst_day_bayes = pd.merge(df_dst_strt_bayes,
                                df_dst_end_bayes,
                                on=['Condition', 'System']).dropna().reset_index(drop=True)
    df_dst_day_sig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                            (df_dst_day_bayes['Autumn Significant?'] == 'No')]
    df_dst_day_bothsig_bayes = df_dst_day_bayes[(df_dst_day_bayes['Spring Significant?'] == 'Yes') &
                                                (df_dst_day_bayes['Autumn Significant?'] == 'Yes')]

    df_dst_day_bayes = df_dst_day_bayes.sort_values(by=['System', 'Condition'])
    df_dst_day_bayes.reset_index(drop=True, inplace=True)

    df_dst_day_sig_bayes = df_dst_day_sig_bayes.sort_values(by=['System', 'Condition'])
    df_dst_day_sig_bayes.reset_index(drop=True, inplace=True)

    df_dst_day_bothsig_bayes = df_dst_day_bothsig_bayes.sort_values(by=['System', 'Condition'])
    df_dst_day_bothsig_bayes.reset_index(drop=True, inplace=True)

    df_dst_day_bayes.to_csv(OUTPATH + '/allResults_Bayesian.csv')
    df_dst_day_sig_bayes.to_csv(OUTPATH + '/interestingResults_Bayesian.csv')
    df_dst_day_bothsig_bayes.to_csv(OUTPATH + '/interestingResultsBothSig_Bayesian.csv')
    
    pm.save_trace(dayTraceSpr, OUTPATH + '/samplingOut/dayTraceSpr')
    pm.save_trace(weekTraceSpr, OUTPATH + '/samplingOut/weekTraceSpr')
    pm.save_trace(dayTraceAut, OUTPATH + '/samplingOut/dayTraceAut')
    pm.save_trace(weekTraceAut, OUTPATH + '/samplingOut/weekTraceAut')
    
    with open(OUTPATH + '/samplingOut/incidences.bpickle3', 'wb') as f:
        pickle.dump({'incidencesSpr': incidencesSpr,
                     'incidencesAut': incidencesAut}, f)
    
    with open(OUTPATH + '/plotSource.bpickle3', 'wb') as f:
        pickle.dump({'plotDataDict_strt_bayes': plotDataDict_strt_bayes,
                     'plotDataDict_end_bayes': plotDataDict_end_bayes}, f)
    
    plotBarsMerged(plotDataDict_strt_bayes,
                   plotDataDict_end_bayes, days=7, 
                   method='Bayesian',
                   ci_lab='Highest Posterior Density ' + ci_name)
