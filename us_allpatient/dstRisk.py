#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 13:50:17 2018

@author: hanxinzhang
Log
 - Aug 28, 2018: to py36
 - Aug 28: add week-level bars to plots.
 - Sep 26: hierarchical prior for rr.
 - Sep 26: results sorted by system and disease names.
 - Oct 3: Stratify by age and sex.
 - Oct 9: Set some theano flags. Make use of AMD math lib.
 - Oct 18: curvature, and redefine rr.
 - Dec 12: Add year 2015.
"""

import datetime
import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import theano.tensor as tt
import pymc3 as pm
import os
from textwrap import wrap
import warnings
import sys

# -----------------------------------------------------------------------------

PART = {'1': 'AllStatesWithDst',
        '2': 'NoDstStates',
        '3': 'NorthernStatesWithDst',
        '4': 'SouthernStatesWithDst',
        '5': 'EasternStatesWithDst',
        '6': 'WesternStatesWithDst'}[sys.argv[1]]


INPUT = './data/ALLP_COND_ENROLL_SUMMARY_%s.bpkl3' % PART
OUTPATH = './results_%s' % PART

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

os.chdir(OUTPATH)

for method in ['Bayesian', 'Frequentist', 'HalfBayesian']:

    if not os.path.exists('plots%sMerged' % method):
        os.makedirs('plots%sMerged' % method)

if not os.path.exists('models'):
    os.makedirs('models')
    
if not os.path.exists('samplingOut'):
    os.makedirs('samplingOut')
    
os.chdir('..')

# os.environ['THEANO_FLAGS'] = 'lib.amdlibm=True, gcc.cxxflags="-L /home/hanxin/amdlibm-3.2/lib/dynamic"'

NUM_ITERATIONS = 2000
EPS = np.finfo(float).eps
PRIOR_FLATNESS = 5
CI_ALPHA = 1e-3

OBS_BOUND = 10 # filter out incidence < 10 at any week day

# -----------------------------------------------------------------------------

START_DATE = datetime.date(2002, 12, 31)

DST_SPRING = [datetime.date(2003, 4, 6),
              datetime.date(2004, 4, 4),
              datetime.date(2005, 4, 3),
              datetime.date(2006, 4, 2),
              datetime.date(2007, 3, 11),
              datetime.date(2008, 3, 9),
              datetime.date(2009, 3, 8),
              datetime.date(2010, 3, 14),
              datetime.date(2011, 3, 13),
              datetime.date(2012, 3, 11),
              datetime.date(2013, 3, 10),
              datetime.date(2014, 3, 9),
              datetime.date(2015, 3, 8)]

DST_AUTUMN = [datetime.date(2003, 10, 26),
              datetime.date(2004, 10, 31),
              datetime.date(2005, 10, 30),
              datetime.date(2006, 10, 29),
              datetime.date(2007, 11, 4),
              datetime.date(2008, 11, 2),
              datetime.date(2009, 11, 1),
              datetime.date(2010, 11, 7),
              datetime.date(2011, 11, 6),
              datetime.date(2012, 11, 4),
              datetime.date(2013, 11, 3),
              datetime.date(2014, 11, 2),
              datetime.date(2015, 11, 1)]

PRESIDENT = [datetime.date(2003, 2, 17),
             datetime.date(2004, 2, 16),
             datetime.date(2005, 2, 21),
             datetime.date(2006, 2, 20),
             datetime.date(2007, 2, 19),
             datetime.date(2008, 2, 18),
             datetime.date(2009, 2, 16),
             datetime.date(2010, 2, 15),
             datetime.date(2011, 2, 21),
             datetime.date(2012, 2, 20),
             datetime.date(2013, 2, 18),
             datetime.date(2014, 2, 17),
             datetime.date(2015, 2, 16)]

WST_EASTER = [datetime.date(2003, 4, 20),
              datetime.date(2004, 4, 11),
              datetime.date(2005, 3, 27),
              datetime.date(2006, 4, 16),
              datetime.date(2007, 4, 8),
              datetime.date(2008, 3, 23),
              datetime.date(2009, 4, 12),
              datetime.date(2010, 4, 4),
              datetime.date(2011, 4, 24),
              datetime.date(2012, 4, 8),
              datetime.date(2013, 3, 31),
              datetime.date(2014, 4, 20),
              datetime.date(2015, 4, 5)]

TKS_GIVING = [datetime.date(2003, 11, 27),
              datetime.date(2004, 11, 25),
              datetime.date(2005, 11, 24),
              datetime.date(2006, 11, 23),
              datetime.date(2007, 11, 22),
              datetime.date(2008, 11, 27),
              datetime.date(2009, 11, 26),
              datetime.date(2010, 11, 25),
              datetime.date(2011, 11, 24),
              datetime.date(2012, 11, 22),
              datetime.date(2013, 11, 28),
              datetime.date(2014, 11, 27),
              datetime.date(2015, 11, 26)]

ST_PATRICKS = [datetime.date(2003, 3, 17),
               datetime.date(2004, 3, 17),
               datetime.date(2005, 3, 17),
               datetime.date(2006, 3, 17),
               datetime.date(2007, 3, 17),
               datetime.date(2008, 3, 17),
               datetime.date(2009, 3, 17),
               datetime.date(2010, 3, 17),
               datetime.date(2011, 3, 17),
               datetime.date(2012, 3, 17),
               datetime.date(2013, 3, 17),
               datetime.date(2014, 3, 17),
               datetime.date(2015, 3, 17)]

VETERANS = [datetime.date(2003, 11, 11),
            datetime.date(2004, 11, 11),
            datetime.date(2005, 11, 11),
            datetime.date(2006, 11, 11),
            datetime.date(2007, 11, 11),
            datetime.date(2008, 11, 11),
            datetime.date(2009, 11, 11),
            datetime.date(2010, 11, 11),
            datetime.date(2011, 11, 11),
            datetime.date(2012, 11, 11),
            datetime.date(2013, 11, 11),
            datetime.date(2014, 11, 11),
            datetime.date(2015, 11, 11)]

COLUMBUS = [datetime.date(2003, 10, 13),
            datetime.date(2004, 10, 11),
            datetime.date(2005, 10, 10),
            datetime.date(2006, 10, 9),
            datetime.date(2007, 10, 8),
            datetime.date(2008, 10, 13),
            datetime.date(2009, 10, 12),
            datetime.date(2010, 10, 11),
            datetime.date(2011, 10, 10),
            datetime.date(2012, 10, 8),
            datetime.date(2013, 10, 14),
            datetime.date(2014, 10, 13),
            datetime.date(2015, 10, 12)]

DST_SPRING_DAY = np.array([(d - START_DATE).days for d in DST_SPRING])
DST_AUTUMN_DAY = np.array([(d - START_DATE).days for d in DST_AUTUMN])

PRESIDENT_DAY = np.array([(d - START_DATE).days for d in PRESIDENT])
WST_EASTER_DAY = np.array([(d - START_DATE).days for d in WST_EASTER])
TKS_GIVING_DAY = np.array([(d - START_DATE).days for d in TKS_GIVING])
ST_PATRICKS_DAY = np.array([(d - START_DATE).days for d in ST_PATRICKS])
VETERANS_DAY = np.array([(d - START_DATE).days for d in VETERANS])
COLUMBUS_DAY = np.array([(d - START_DATE).days for d in COLUMBUS])

SPRING_AVOID = np.concatenate((PRESIDENT_DAY, WST_EASTER_DAY, ST_PATRICKS_DAY))
AUTUMN_AVOID = np.concatenate((TKS_GIVING_DAY, VETERANS_DAY, COLUMBUS_DAY))


def dstOverlapHoliday(dst, avoid, wc):
    
    forward = avoid - dst
    backward = dst - avoid
    
    forwardOverlap = (wc * 7 <= forward) & (forward < wc * 7 + 7)
    backwardOverlap = (wc * 7 - 7 < backward) & (backward <= wc * 7)
    
    return not np.any(forwardOverlap | backwardOverlap)


DST_SPRING_CONTROL = []
for dst in DST_SPRING_DAY:
    
    if dstOverlapHoliday(dst, SPRING_AVOID, 2):
        
        DST_SPRING_CONTROL.append(2)
        
    else:
        
        if dstOverlapHoliday(dst, SPRING_AVOID, 3):
            
            DST_SPRING_CONTROL.append(3)
            
        else:
            
            if dstOverlapHoliday(dst, SPRING_AVOID, 1):
            
                DST_SPRING_CONTROL.append(1)
                
            else:
                
                warnings.warn('Cannot control by ±1 points!\n'
                              'None of ±2 weeks, ±3 weeks and ±1 week avoids '
                              'celebrations completely. Fall back to ±3 weeks ...')
                
                DST_SPRING_CONTROL.append(3)
                

DST_AUTUMN_CONTROL = []
for dst in DST_AUTUMN_DAY:
    
    if dstOverlapHoliday(dst, AUTUMN_AVOID, 2):
        
        DST_AUTUMN_CONTROL.append(2)
        
    else:
        
        # Try ±1 week first because we want to avoid the Tksgvng period that has
        # temporally long and large impact on diagnosis recording.
        
        if dstOverlapHoliday(dst, AUTUMN_AVOID, 1):
        
            DST_AUTUMN_CONTROL.append(1)
            
        else:
            
            if dstOverlapHoliday(dst, AUTUMN_AVOID, 3):
                
                DST_AUTUMN_CONTROL.append(3)
                
            else:
                
                warnings.warn('Cannot control by ±1 points!\n'
                              'None of ±2 weeks, ±3 weeks and ±1 week avoids '
                              'celebrations completely. Fall back to ±1 weeks ...')
                
                DST_AUTUMN_CONTROL.append(1)
            
    
print('Expectation is calculated by ±', DST_SPRING_CONTROL,
      'weeks in Spring')

print('and ±', DST_AUTUMN_CONTROL, 'weeks in Autumn')

# -----------------------------------------------------------------------------


def isSig(interv):

    return (min(interv) > 1.) or (0. < max(interv) < 1.)


def findCI(theta1, theta2, sigma1, sigma2, alpha):
    
    z = sps.norm.ppf(1 - alpha / 2)
    
    a = theta2**2 - (z**2) * (sigma2**2)
    b = -2*theta1*theta2
    c = theta1**2 - (z**2) * (sigma1**2)
    
    if (b**2 - 4 * a * c < 0):
        print(theta1, theta2, sigma1, sigma2)
        print(a, b, c, b**2 - 4 * a * c)
    
    root = np.roots([a, b, c])
    ci = np.sort(root)
    
    errorBound = sps.norm.cdf(-theta2/sigma2)
    reliable = 'Yes' if (errorBound / (alpha / 2)) < 0.01 else 'No'
       
    return {'CI': ci, 'Error Bound': errorBound, 'Reliable CI?': reliable}


def betaToApproxNormal(a, b):
    
    mu = a / (a + b)
    var = a * b / (((a + b)**2) * (a + b + 1))
    
    return mu, np.sqrt(var)


def relativeRiskFrequentist(xbefore, nbefore, x0, n0, xafter, nafter, n_tests, n_selection,
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
        
    pbefore = xbefore / nbefore
    se_before = (pbefore * (1 - pbefore) / nbefore)**0.5
    pafter = xafter / nafter
    se_after = (pafter * (1 - pafter) / nafter)**0.5
    
    p0 = x0 / n0
    se0 = (p0 * (1-p0) / n0)**0.5
    
    p0_expect = (pbefore + pafter) / 2.
    se0_expect = (0.25*(se_before**2) + 0.25*(se_after**2))**0.5
    
    ci = findCI(theta1=p0, theta2=p0_expect, 
                sigma1=se0, sigma2=se0_expect, alpha=ci_alpha)
    
    ci_bonf = findCI(theta1=p0, theta2=p0_expect, 
                     sigma1=se0, sigma2=se0_expect, alpha=ci_alpha/n_tests)
    
    if isSig(ci['CI']):
        selected += 1
        ci_fcr = findCI(theta1=p0, theta2=p0_expect, 
                        sigma1=se0, sigma2=se0_expect, 
                        alpha=n_selection * ci_alpha/n_tests)
    else:
        ci_fcr = ci
    
    rr = p0 / p0_expect
    
    return rr, ci, ci_bonf, ci_fcr, selected


def relativeRiskHalfBayesian(xbefore, nbefore, x0, n0, xafter, nafter, n_tests, n_selection,
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
        
    # Using Jeffrey's prior Beta(0.5, 0.5)
    beta_a_before = 0.5 + xbefore
    beta_b_before = 0.5 + nbefore - xbefore
    
    beta_a_after = 0.5 + xafter
    beta_b_after = 0.5 + nafter - xafter
    
    beta_a_0 = 0.5 + x0
    beta_b_0 = 0.5 + n0 - x0
    
    mu_before, sd_before = betaToApproxNormal(beta_a_before, beta_b_before)
    mu_after, sd_after = betaToApproxNormal(beta_a_after, beta_b_after)
    
    mu_expect = 0.5 * (mu_before + mu_after)
    sd_expect = (0.25*(sd_before**2) + 0.25*(sd_after**2))**0.5
    
    mu_0, sd_0 = betaToApproxNormal(beta_a_0, beta_b_0)
    
    ci = findCI(theta1=mu_0, theta2=mu_expect, 
                sigma1=sd_0, sigma2=sd_expect, alpha=ci_alpha)
    
    ci_bonf = findCI(theta1=mu_0, theta2=mu_expect, 
                     sigma1=sd_0, sigma2=sd_expect, alpha=ci_alpha/n_tests)
    
    if isSig(ci['CI']):
        selected += 1
        ci_fcr = findCI(theta1=mu_0, theta2=mu_expect, 
                        sigma1=sd_0, sigma2=sd_expect,
                        alpha=n_selection * ci_alpha/n_tests)
    else:
        ci_fcr = ci
    
    rr = mu_0 / mu_expect
    
    return rr, ci, ci_bonf, ci_fcr, selected


def relativeRiskTraceBayesian(xbefore, nbefore, x0, n0, xafter, nafter,
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

            # Hierarchical prior for cc pools information across all conditions.
            # This is a bayesian way to deal with multiple comparisons.
            # cc is a the curvature of trend.
            # rr = cc / E[cc] is the corrected relative risk.
            # E[rr] = 1, according with our prior belief.
            # This correction is equivalent to setting cc_mu == 1.

            cc_mu = pm.HalfCauchy('cc_mu', PRIOR_FLATNESS)
            cc_sd = pm.HalfCauchy('cc_sd', PRIOR_FLATNESS)

            pbefore = pm.Uniform('pbefore',
                                 lower=EPS,
                                 upper=1. - EPS,
                                 shape=modelShape)
            pafter = pm.Uniform('pafter',
                                lower=EPS,
                                upper=1. - EPS,
                                shape=modelShape)
            cc = pm.Gamma('cc',
                          mu=cc_mu,
                          sd=clip0toInf(cc_sd),
                          shape=modelShape)

            p0 = pm.Deterministic('p0', 0.5 * cc * (pbefore + pafter))

            x_obs_b = pm.Binomial('x_obs_b',
                                  n=nbefore,
                                  p=pbefore,
                                  observed=xbefore,
                                  shape=modelShape)
            x_obs_a = pm.Binomial('x_obs_a',
                                  n=nafter,
                                  p=pafter,
                                  observed=xafter,
                                  shape=modelShape)
            x_obs_0 = pm.Binomial('x_obs_0',
                                  n=n0,
                                  p=clip0to1(p0),
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
    
    
def doFrequentistOrHalfBayesian(dis,
                                system,
                                incidences_dst_before,
                                incidences_dst_after,
                                incidences_dst_0,
                                total_before,
                                total_after,
                                total_0,
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
            
            nbefore = total_before[dayOfWeek]
            nafter = total_after[dayOfWeek]
            n0 = total_0[dayOfWeek]
            
            rr, conf, bonf_conf, fcr_conf, dsel = methodFunc(xbefore, nbefore, 
                                                             x0, n0, 
                                                             xafter, nafter, 
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
                                              np.around(total_before.sum() / days), 
                                              incidences_dst_0.sum(), 
                                              np.around(total_0.sum() / days),  
                                              incidences_dst_after.sum(), 
                                              np.around(total_after.sum() / days), 
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
    n0_tensor = np.array([incidencesDict[k][3] for k in diseases])

    xbefore_tensor = np.array([incidencesDict[k][4] for k in diseases])
    nbefore_tensor = np.array([incidencesDict[k][5] for k in diseases])

    xafter_tensor = np.array([incidencesDict[k][6] for k in diseases])
    nafter_tensor = np.array([incidencesDict[k][7] for k in diseases])

    dayTrace = relativeRiskTraceBayesian(xbefore_tensor,
                                         nbefore_tensor,
                                         x0_tensor,
                                         n0_tensor,
                                         xafter_tensor,
                                         nafter_tensor,
                                         correction=correction,
                                         days=days)

    weekTrace = relativeRiskTraceBayesian(xbefore_tensor.sum(axis=1),
                                          np.around(nbefore_tensor.sum(axis=1) / days),
                                          x0_tensor.sum(axis=1),
                                          np.around(n0_tensor.sum(axis=1) / days),
                                          xafter_tensor.sum(axis=1),
                                          np.around(nafter_tensor.sum(axis=1) / days),
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
        print('\x1b[0;37;41m' + 
              'Something happened when doing Bayesian modeling!' + 
              '\x1b[0m')
        df_dict = np.nan
        plotDataDict = np.nan

    return df_dict, plotDataDict


def cleanIncidences(incidDict, bound=OBS_BOUND):

    cleaned = {}
    for k, v in incidDict.items():
        midGreaterThan = np.all(v[2] >= bound)
        befGreaterThan = np.all(v[4] >= bound)
        aftGreaterThan = np.all(v[6] >= bound)
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

        notNaN = isinstance(plotDataDict1[dis], tuple) and isinstance(plotDataDict2[dis], tuple)
        condSys, sex, age = dis.split('$')
        sex = {'M': 'Male', 'F': 'Female'}[sex]
        cond, system = condSys.split('|')
        
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
            fig.suptitle('\n'.join(wrap(cond + ', ' + system)))
            fig.text(0.14, 0.9, sex + ', ' + age, size=10)

            figName = OUTPATH + '/plots' + method + 'Merged/' + saveName + '.pdf'
            fig.savefig(figName, format='pdf')
            plt.close(fig)


def sampleTraces(day_of_interest, 
                 week_of_control,
                 diseasesDict=None,
                 totalDict=None,
                 correction=None,
                 days=7,
                 method=None,
                 df=None,
                 plotDataDict=None):
    
    all_groups = sorted(list(diseasesDict.keys()))

    if (method != 'Frequentist') and (method != 'HalfBayesian'):

        print('Apply the Bayesian Method...')

    incidencesDict = {}

    for studyGrp_index, studyGrp in enumerate(all_groups):

        try:
            condSysName, sex, age = studyGrp.split('$')
            strat = age + ', ' + sex
            totalCount = totalDict[strat]
        except:
            condSysName = studyGrp
            totalCount = totalDict                

        observations = diseasesDict[studyGrp]
        
        condName, sysName = condSysName.split('|')
            
        incidences_dst_0 = np.zeros(days)
        incidences_dst_before = np.zeros(days)
        incidences_dst_after = np.zeros(days)
        total_0 = np.zeros(days)
        total_before = np.zeros(days)
        total_after = np.zeros(days)

        for i, di in enumerate(list(day_of_interest)):

            wc = week_of_control[i]
            dc = wc * 7

            index_0 = np.array(range(di, di + days))
            index_before = np.array(range(di - dc, di - dc + days))
            index_after = np.array(range(di + dc, di + dc + days))

            incidences_dst_0 += np.array([observations[d] for d in index_0])
            incidences_dst_before += np.array([observations[d] for d in index_before])
            incidences_dst_after += np.array([observations[d] for d in index_after])
            total_0 += np.array([totalCount[d] for d in index_0])
            total_before += np.array([totalCount[d] for d in index_before])
            total_after += np.array([totalCount[d] for d in index_after])
            
        midGreaterThan = np.all(incidences_dst_before >= OBS_BOUND)
        befGreaterThan = np.all(incidences_dst_after >= OBS_BOUND)
        aftGreaterThan = np.all(incidences_dst_0 >= OBS_BOUND)
        qc = all([midGreaterThan, befGreaterThan, aftGreaterThan])
        
        if qc:
    
            incidencesDict[studyGrp] = (studyGrp_index,
                                        sysName,
                                        incidences_dst_0, total_0,
                                        incidences_dst_before, total_before,
                                        incidences_dst_after, total_after)
    
    if method != 'Bayesian':
        
        n_diseases = len(incidencesDict)
        day_selected = 0
        week_selected = 0
        
        for studyGrp, v in incidencesDict.items():
            
            (studyGrp_index,
             sysName,
             incidences_dst_0, total_0,
             incidences_dst_before, total_before,
             incidences_dst_after, total_after) = v

            if method == 'Frequentist':
                
                (dfEntry, 
                 plotDataDictEntry, 
                 dsel, 
                 wsel) = doFrequentistOrHalfBayesian(studyGrp,
                                                     sysName,
                                                     incidences_dst_before,
                                                     incidences_dst_after,
                                                     incidences_dst_0,
                                                     total_before,
                                                     total_after,
                                                     total_0,
                                                     days,
                                                     n_diseases,
                                                     n_selection_day=1,
                                                     n_selection_week=1,
                                                     correction=correction,
                                                     methodFunc=relativeRiskFrequentist)
                
                day_selected += dsel
                week_selected += wsel
    
            if method == 'HalfBayesian':
    
                (dfEntry, 
                 plotDataDictEntry, 
                 dsel, 
                 wsel) = doFrequentistOrHalfBayesian(studyGrp,
                                                     sysName,
                                                     incidences_dst_before,
                                                     incidences_dst_after,
                                                     incidences_dst_0,
                                                     total_before,
                                                     total_after,
                                                     total_0,
                                                     days,
                                                     n_diseases,
                                                     n_selection_day=1,
                                                     n_selection_week=1,
                                                     correction=correction,
                                                     methodFunc=relativeRiskHalfBayesian)
                
                day_selected += dsel
                week_selected += wsel
        
        print('Day selection = ', day_selected)
        print('Week selection = ', week_selected)
        
        for studyGrp, v in incidencesDict.items():
            
            (studyGrp_index,
             sysName,
             incidences_dst_0, total_0,
             incidences_dst_before, total_before,
             incidences_dst_after, total_after) = v
             
            if method == 'Frequentist':
                
                (dfEntry, 
                 plotDataDictEntry, 
                 dsel, 
                 wsel) = doFrequentistOrHalfBayesian(studyGrp,
                                                     sysName,
                                                     incidences_dst_before,
                                                     incidences_dst_after,
                                                     incidences_dst_0,
                                                     total_before,
                                                     total_after,
                                                     total_0,
                                                     days,
                                                     n_diseases,
                                                     n_selection_day=day_selected,
                                                     n_selection_week=week_selected,
                                                     correction=correction,
                                                     methodFunc=relativeRiskFrequentist)
                
                df.loc[studyGrp_index] = dfEntry
                plotDataDict[studyGrp] = plotDataDictEntry
    
            if method == 'HalfBayesian':
    
                (dfEntry, 
                 plotDataDictEntry, 
                 dsel, 
                 wsel) = doFrequentistOrHalfBayesian(studyGrp,
                                                     sysName,
                                                     incidences_dst_before,
                                                     incidences_dst_after,
                                                     incidences_dst_0,
                                                     total_before,
                                                     total_after,
                                                     total_0,
                                                     days,
                                                     n_diseases,
                                                     n_selection_day=day_selected,
                                                     n_selection_week=week_selected,
                                                     correction=correction,
                                                     methodFunc=relativeRiskHalfBayesian)
                
                df.loc[studyGrp_index] = dfEntry
                plotDataDict[studyGrp] = plotDataDictEntry    
            
        return df, plotDataDict    
    
    if (method != 'Frequentist') and (method != 'HalfBayesian'):
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

    df_dict, plotDataDict = tracesToResults(incidencesDict, 
                                            correction, 
                                            dayTrace, 
                                            weekTrace, 
                                            ci_alpha=ci_alpha)

    for index in list(df_dict.keys()):
        df.loc[index] = df_dict[index]

    return df, plotDataDict


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    with open(INPUT, 'rb') as f:
        summDict = pickle.load(f)
        COND_DICT = summDict['COND_DICT']
        ENROLL_DICT = summDict['ENROLL_DICT']    


# Frequentist's Method (all diseases ...) -------------------------------------
    
    ci_name = str(100. * (1. - CI_ALPHA)) + '% C.I.'   
    
    columnNames = ['Condition', 'System', 
                   'Relative Risk (Spring)', 
                   ci_name + ' (Spring)', 
                   'False Coverage Rate Corrected ' + ci_name + ' (Spring)', 
                   'Spring Significant?']
    
        
    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}
    
    df_dst_start_frequ, plotDataDict_start_frequ = sampleTraces(day_of_interest=DST_SPRING_DAY,
                                                                week_of_control=DST_SPRING_CONTROL,
                                                                diseasesDict=COND_DICT,
                                                                totalDict=ENROLL_DICT,
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
    
    df_dst_end_frequ, plotDataDict_end_frequ = sampleTraces(day_of_interest=DST_AUTUMN_DAY,
                                                            week_of_control=DST_AUTUMN_CONTROL,
                                                            diseasesDict=COND_DICT,
                                                            totalDict=ENROLL_DICT,
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
    
    
# Half-Bayesian Method (all diseases ...) -------------------------------------
    
    ci_name = str(100. * (1. - CI_ALPHA)) + '% C.I.'   
    
    columnNames = ['Condition', 'System', 
                   'Relative Risk (Spring)', 
                   ci_name + ' (Spring)', 
                   'False Coverage Rate Corrected ' + ci_name + ' (Spring)', 
                   'Spring Significant?']
    
        
    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}
    
    df_dst_start_hfbayes, plotDataDict_start_hfbayes = sampleTraces(day_of_interest=DST_SPRING_DAY,
                                                                    week_of_control=DST_SPRING_CONTROL,
                                                                    diseasesDict=COND_DICT,
                                                                    totalDict=ENROLL_DICT,
                                                                    correction='Spring',
                                                                    days=7,
                                                                    method='HalfBayesian',
                                                                    df=df,
                                                                    plotDataDict=plotDataDict)
    
    
    columnNames = ['Condition', 'System', 
                   'Relative Risk (Autumn)', 
                   ci_name + ' (Autumn)', 
                   'False Coverage Rate Corrected ' + ci_name + ' (Autumn)', 
                   'Autumn Significant?']
    
    df = pd.DataFrame(columns=columnNames)
    plotDataDict = {}
    
    df_dst_end_hfbayes, plotDataDict_end_hfbayes = sampleTraces(day_of_interest=DST_AUTUMN_DAY,
                                                                week_of_control=DST_AUTUMN_CONTROL,
                                                                diseasesDict=COND_DICT,
                                                                totalDict=ENROLL_DICT,
                                                                correction='Autumn',
                                                                days=7,
                                                                method='HalfBayesian',
                                                                df=df,
                                                                plotDataDict=plotDataDict)
        
    df_dst_day_hfbayes = pd.merge(df_dst_start_hfbayes,
                                  df_dst_end_hfbayes,
                                  on=['Condition', 'System']).dropna().reset_index(drop=True)
    df_dst_day_sig_hfbayes = df_dst_day_hfbayes[(df_dst_day_hfbayes['Spring Significant?'] == 'Yes') &
                                                (df_dst_day_hfbayes['Autumn Significant?'] == 'No')]
    df_dst_day_bothsig_hfbayes = df_dst_day_hfbayes[(df_dst_day_hfbayes['Spring Significant?'] == 'Yes') &
                                                    (df_dst_day_hfbayes['Autumn Significant?'] == 'Yes')]

    df_dst_day_hfbayes = df_dst_day_hfbayes.sort_values(by=['System', 'Condition'])
    df_dst_day_hfbayes.reset_index(drop=True, inplace=True)

    df_dst_day_sig_hfbayes = df_dst_day_sig_hfbayes.sort_values(by=['System', 'Condition'])
    df_dst_day_sig_hfbayes.reset_index(drop=True, inplace=True)

    df_dst_day_bothsig_hfbayes = df_dst_day_bothsig_hfbayes.sort_values(by=['System', 'Condition'])
    df_dst_day_bothsig_hfbayes.reset_index(drop=True, inplace=True)
    
    plotBarsMerged(plotDataDict_start_hfbayes, plotDataDict_end_hfbayes, days=7, 
                   method='HalfBayesian', ci_lab='FCR Corrected ' + ci_name)
    
    df_dst_day_hfbayes.to_csv(OUTPATH + '/allResults_HalfBayesian.csv')
    df_dst_day_sig_hfbayes.to_csv(OUTPATH + '/interestingResults_HalfBayesian.csv')
    df_dst_day_bothsig_hfbayes.to_csv(OUTPATH + '/interestingResultsBothSig_HalfBayesian.csv')
    
    
# Bayesian Method (all diseases ...) ------------------------------------------

    incidencesSpr, dayTraceSpr, weekTraceSpr = sampleTraces(day_of_interest=DST_SPRING_DAY,
                                                            week_of_control=DST_SPRING_CONTROL,
                                                            diseasesDict=COND_DICT,
                                                            totalDict=ENROLL_DICT,
                                                            correction='Spring',
                                                            days=7,
                                                            method='Bayesian')

    incidencesAut, dayTraceAut, weekTraceAut = sampleTraces(day_of_interest=DST_AUTUMN_DAY,
                                                            week_of_control=DST_AUTUMN_CONTROL,
                                                            diseasesDict=COND_DICT,
                                                            totalDict=ENROLL_DICT,
                                                            correction='Autumn',
                                                            days=7,
                                                            method='Bayesian')

    CI_ALPHA = 1e-3
    df_dst_start_bayes, plotDataDict_start_bayes = wrapResults(incidencesSpr,
                                                               'Spring',
                                                               dayTraceSpr,
                                                               weekTraceSpr,
                                                               ci_alpha=CI_ALPHA)

    df_dst_end_bayes, plotDataDict_end_bayes = wrapResults(incidencesAut,
                                                           'Autumn',
                                                           dayTraceAut,
                                                           weekTraceAut,
                                                           ci_alpha=CI_ALPHA)

    df_dst_day_bayes = pd.merge(df_dst_start_bayes,
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
    
    with open(OUTPATH + '/samplingOut/incidences.bpkl3', 'wb') as f:
        pickle.dump({'incidencesSpr': incidencesSpr,
                     'incidencesAut': incidencesAut}, f)

    plotBarsMerged(plotDataDict_start_bayes,
                   plotDataDict_end_bayes, days=7, 
                   method='Bayesian',
                   ci_lab='Highest Posterior Density ' + ci_name)
    
    with open(OUTPATH + '/plotSource.bpkl3', 'wb') as f:
        pickle.dump({'plotDataDict_start_bayes': plotDataDict_start_bayes,
                     'plotDataDict_end_bayes': plotDataDict_end_bayes,
                     'plotDataDict_start_frequ': plotDataDict_start_frequ,
                     'plotDataDict_end_frequ': plotDataDict_end_frequ,
                     'plotDataDict_start_hfbayes': plotDataDict_start_hfbayes,
                     'plotDataDict_end_hfbayes':  plotDataDict_end_hfbayes}, f)
    