# Health Effects of Shifting to and from Daylight Saving Time (DST) in the US and Sweden

<p align="center">
  <img src="./DST_model.png" width="600">
</p>

## This project features

* a simple and beautiful Bayesian framework for relative risk estimation,
* a paradigm for correcting the multiple comparisons problem in a Bayesian way,
* one of the very few real-world examples indicating the equivalence between the Bayesian hierarchical model and the Frequentist method with false coverage rate (FCR) adjustment, 
* findings of elevated risks distributed in various systems: circulatory and cardiovascular, injuries, pregnancy and childbirth, mental and behavioral, stress-related immune, etc.

<p align="left">
  <img src="./us_allpat.png" width="900">
</p>

<p align="left">
  <img src="./us_inpat.png" width="800">
</p>

## Contents

For US all patients (inpatients and outpatients) and US inpatients, we performed analyses on

* all states adopting DST,
* all states adopting DST on pseudo-DST dates as a negative control,
* all states without DST as another negative control,
* all northern, southern, eastern, and western DST states separately,
* first diagnoses of conditions of 0–10-y/o children who were completely recorded from birth in our data,
* first diagnoses on pseudo-DST dates as a negative control

using three approaches -- a Bayesian hierarchical model, a Frequentist approximation method, and a half-Bayesian approximation method.

For Swedish inpatients, we performed analyses on 

* medical records since 1980 when Sweden started observing DST,
* medical records before 1980 as a negative control

using two approaches -- a Bayesian hierarchical model and a Frequentist approximation method.

## Effective population sizes

The effective sizes of population strata, computed as sum of population sizes on the DST spring change dates from 2003 to 2014 (or 2015 for the inpatient data), are 

| Stratum       | US all-patient |  US inpatient  | 
|:--------------|:--------------:|:--------------:|
| F, 0–10       | 24,303,479     | 25,910,513     | 
| M, 0–10       | 25,421,528     | 27,103,968     | 
| F, 11–20      | 29,496,593     | 31,473,631     | 
| M, 11–20      | 30,677,787     | 32,741,080     | 
| F, 21–40      | 57,320,746     | 61,587,468     | 
| M, 21–40      | 52,366,989     | 56,436,933     | 
| F, 41–60      | 74,364,939     | 79,345,761     | 
| M, 41–60      | 66,205,024     | 79,345,761     | 
| F, 61–        | 33,281,829     | 35,507,396     | 
| M, 61–        | 28,399,046     | 30,241,444     | 

