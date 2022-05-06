#!/usr/bin/env python3
import numpy as np
import pandas as pd
import copy
import sklearn as sk
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression

import model
import verifier
import cscUpdater
import updater

import importlib as imp

import itertools
import time

from hummingbird.ml import convert

import warnings
warnings.filterwarnings('ignore')

import acsDataParallel

#-------------------------------------------------------------------------------
test_size = 0.2 #train-test split

acs_task = 'income' # options: employment, income, public_coverage, mobility, and travel_time.
acs_year = 2018 #must be >= 2014. Upper bound unknown.
acs_states = ['NY']
acs_horizon='1-Year' #1-Year or 5-Year
acs_survey='person' #'person' or 'household'

# for subsampling rows: can specify first and last of data to be pulled. currently pulling everything.
row_start = 0
row_end = 30000

# for subsampling columns. note: can only subsample consecutive columns with current implementation
col_start=0
col_end=-1

[train_x, train_y, test_x, test_y, demo_group_functions, demo_group_indicators, min_age, mid_age] = acsDataParallel.get_data(test_size, acs_task, acs_year, acs_states,acs_horizon=acs_horizon, acs_survey=acs_survey, row_start = row_start,row_end = row_end, col_start=col_start, col_end=col_end)

#-------------------------------------------------------------------------------
def g1(X):
    return ((X['WKHP'] == 40))

truth_series = g1(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf1 = sk.ensemble.RandomForestClassifier(n_estimators=100, max_depth=11)
clf1.fit(X_train,y_train)
clf1GPU = convert(clf1, 'pytorch')
clf1GPU.to('cuda')

def h1(x):
    return clf1GPU.predict(x)

def g2(X):
    return ((X['WKHP'] <= 20))

truth_series = g2(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf2 = sk.ensemble.RandomForestClassifier(n_estimators=100, max_depth=11)
clf2.fit(X_train,y_train)
clf2GPU = convert(clf2, 'pytorch')
clf2GPU.to('cuda')
def h2(x):
    return clf2GPU.predict(x)

def g3(X):
    return ((X['RAC1P'] >= 3))

truth_series = g3(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf3 = sk.ensemble.RandomForestClassifier(n_estimators=200, max_depth=13)
clf3.fit(X_train,y_train)
clf3GPU = convert(clf3, 'pytorch')
clf3GPU.to('cuda')
def h3(x):
    return clf3GPU.predict(x)

def g4(X):
    return ((X['RAC1P'] == 1))

truth_series = g4(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf4 = sk.ensemble.RandomForestClassifier(n_estimators=200, max_depth=15)
clf4.fit(X_train,y_train)
clf4GPU = convert(clf4, 'pytorch')
clf4GPU.to('cuda')
def h4(x):
    return clf4GPU.predict(x)

def g5(X):
    return (X['SCHL'] <= 12)

truth_series = g5(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]
clf5 = sk.ensemble.RandomForestClassifier(n_estimators=100, max_depth=14)
clf5.fit(X_train,y_train)
clf5GPU = convert(clf5, 'pytorch')
clf5GPU.to('cuda')
def h5(x):
    return clf5GPU.predict(x)

def g6(X):
    return (X['SCHL'] >= 16)

truth_series = g6(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf6 = sk.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200,max_depth = 3,random_state=0)
clf6.fit(X_train,y_train)
clf6GPU = convert(clf6, 'pytorch')
clf6GPU.to('cuda')
def h6(x):
    return clf6GPU.predict(x)

def g7(X):
    return (X['AGEP'] <=30)

truth_series = g7(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]
clf7 = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,max_depth = 4,random_state=0)
clf7.fit(X_train,y_train)
clf7GPU = convert(clf7, 'pytorch')
clf7GPU.to('cuda')
def h7(x):
    return clf7GPU.predict(x)

def g8(X):
    return (X['COW'] == 1)

truth_series = g8(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf8 = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500,max_depth = 3,random_state=0)
clf8.fit(X_train,y_train)
clf8GPU = convert(clf8, 'pytorch')
clf8GPU.to('cuda')

def h8(x):
    return clf8GPU.predict(x)

def g9(X):
    return ((X['POBP'] <= 20) & (X['SEX'] == 2))

truth_series = g9(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf9 = sk.ensemble.RandomForestClassifier(n_estimators=200, max_depth=15)
clf9.fit(X_train,y_train)
clf9GPU = convert(clf9, 'pytorch')
clf9GPU.to('cuda')
def h9(x):
    return clf9GPU.predict(x)

def g10(X):
    return ((X['OCCP'] <= 100))

truth_series = g10(train_x)
X_train = train_x[truth_series]
y_train = train_y[truth_series]

clf10 = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=30,max_depth = 4,random_state=0)
clf10.fit(X_train,y_train)
clf10GPU = convert(clf10, 'pytorch')
clf10GPU.to('cuda')
def h10(x):
    return clf10GPU.predict(x)

#-------------------------------------------------------------------------------

def verify_size(x, group):
# helper function that checks that the discovered group isn't too small to run on
    g_indices = group(x) == 1
    g_xs = x[g_indices]
    if len(g_xs) == 0:
        return False
    else:
        return True

#-------------------------------------------------------------------------------

initial_model = DecisionTreeClassifier(max_depth = 1, random_state=0)
initial_model.fit(train_x, train_y);

#-------------------------------------------------------------------------------

start_time = time.time()
def find_best_worst_perm(train_x, train_y, test_x, test_y):
    permutations = list(itertools.permutations([1,2,3,4,5,6,7,8]))
    best_error = 100
    worst_error = 0
    best_permutation = None
    worst_permutation = None
    #set dummies
    g0 = None
    h0 = None

    #initialize lists
    group_list = [g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10]
    predicate_list = [h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10]
    counter = 0

    for permutation in permutations:
        #round count
        if counter%10 == 0:
            print('Time for ' + str(counter) + ' permutations (seconds):', time.time() - start_time)
        #reinitialize model
        mod = model.PointerDecisionList(initial_model.predict, [])
        mod.test_errors.append(cscUpdater.measure_group_errors(mod, test_x, test_y))
        mod.train_errors.append(cscUpdater.measure_group_errors(mod, train_x, train_y))

        #begin updates
        for index in permutation:
            improvement_check = verifier.is_proposed_group_good_csc(mod, test_x, test_y, predicate_list[index],group_list[index])
            if improvement_check:
            # run the update
                cscUpdater.iterative_update(mod, predicate_list[index], group_list[index], train_x, train_y, test_x, test_y, 'g'+str(index))
        total_error = mod.test_errors[-1][0]
        if total_error < best_error:
            best_error = total_error
            best_permutation = permutation
        if total_error > worst_error:
            worst_error = total_error
            worst_permutation = permutation
        counter += 1
    finish_time = time.time()
    delta_time = finish_time - start_time
    print('Best error:',best_error)
    print('Best Order:', best_permutation)
    print('Worst error:',worst_error)
    print('Worst Order:', worst_permutation)
    print('Total time (seconds):',delta_time)

find_best_worst_perm(train_x, train_y, test_x, test_y)
