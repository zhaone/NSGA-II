#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 zhaoyi <yizhaome@gmail.com>
#
# Distributed under terms of the Apache V2.0 license.
import numpy as np
import random

# SBX cross
def sbx_cross(population, cross_prob=0.5, expo=1):
    for _ in range(len(population)):
        if np.random.random() < cross_prob:
            # compute para
            mu = np.random.random()
            if mu < 0.5:
                para = np.power(2*mu, 1/(expo+1))
            else:
                para = np.power(1/(2*(1-mu)), 1/(expo+1))
            # cross
            ind1, ind2 = random.sample(range(len(population)), 2)
            p1, p2 = population[ind1], population[ind2]
            temp1 = p1
            p1 = 0.5*((1+para)*p1 + (1-para)*p2)
            p2 = 0.5*((1-para)*temp1 + (1+para)*p2)
            population[ind1] = p1
            population[ind2] = p2
    return population

# ramdom cross
def random_cross(population, cross_prob=0.5, expo=1):
    for _ in range(len(population)):
        if np.random.random() < cross_prob:
            # compute para
            mu = np.random.random()
            if mu < 0.5:
                para = 1 - mu
            else:
                para = mu
            # cross
            ind1, ind2 = random.sample(range(len(population)), 2)
            p1, p2 = population[ind1], population[ind2]
            temp1 = p1
            p1 = para*p1 + (1-para)*p2
            p2 = (1-para)*temp1 + para*p2
            population[ind1] = p1
            population[ind2] = p2
    return population


# polynomial mutation
def poly_mutate(population, mute_prob=0.1, expo=1):
    for _ in range(len(population)):
        if np.random.random() < mute_prob:
            # compute para
            mu = np.random.random()
            if mu < 0.5:
                addition = np.power(2*mu, 1/(expo+1)) - 1
            else:
                addition = np.power(1-(2*(1-mu)), 1/(expo+1))
            # cross
            ind = random.sample(range(len(population)), 1)
            population[ind] += addition
    return population

# polynomial mutation
def random_mutate(population, mute_prob=0.2, _min=-1000, _max=1000):
    for _ in range(len(population)):
        if np.random.random() < mute_prob:
            ind = random.sample(range(len(population)), 1)
            population[ind] = _min + np.random.random() * (_max-_min)
    return population
