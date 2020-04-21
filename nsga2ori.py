#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 zhaoyi <yizhaome@gmail.com>
#
# Distributed under terms of the Apache V2.0 license.
import numpy as np
import random

from cmfunc import sbx_cross, poly_mutate
from objfunc import toy_objectives

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(objectives):
    popSize, objFucNum = objectives.shape
    ps = np.repeat(objectives, popSize, axis=0)
    qs = np.tile(objectives, (popSize, 1))
    # compute n_p
    dominated_flags = np.sum(
        qs-ps < 0, axis=1).reshape(popSize, popSize) == objFucNum
    N = np.sum(dominated_flags, axis=1)
    # compute S_p
    S = []
    dominate_flags = np.sum(
        ps-qs < 0, axis=1).reshape(popSize, popSize) == objFucNum
    for dominate_flag in dominate_flags:
        S_p = []
        for q, flag in enumerate(dominate_flag):
            if flag:
                S_p.append(q)
        S.append(S_p)

    # compute front
    front = []
    while True:
        Fi = np.argwhere(N == 0).reshape(-1)
        if Fi.size == 0:
            break
        front.append(Fi)
        N[Fi] = -1
        for p in Fi:
            for q in S[p]:
                N[q] -= 1
    return front

#Function to calculate crowding distance
def crowding_distance(objectives, front):
    popSize, objFucNum = objectives.shape
    distance = np.zeros(popSize)

    for front_objs_idx in front:
        if len(front_objs_idx) <= 2:
            distance[front_objs_idx] = np.inf

        tmp_dis = np.zeros(len(front_objs_idx))
        front_objs = objectives[front_objs_idx, :]
        ind = np.argsort(front_objs, axis=0).T
        for i in range(objFucNum):
            front_objs_i = front_objs[:, i]
            tmp_dis[ind[i, 0]] = np.inf
            tmp_dis[ind[i, -1]] = np.inf
            extent = front_objs_i[ind[i][-1]] - front_objs_i[ind[i][0]]
            for j in range(1, len(front_objs_idx)-1):
                tmp_dis[ind[i][j]] += (front_objs_i[ind[i][j+1]] -
                                       front_objs_i[ind[i][j-1]]) / extent
        distance[front_objs_idx] = tmp_dis
    return distance


def nsga2(objsFunc=toy_objectives,
          crossFunc=sbx_cross, CFKwargs={'cross_prob': 0.5, 'expo': 1},
          mutationFunc=poly_mutate, MFKwargs={'mute_prob': 0.1, 'expo': 1},
          initPop=None,
          popSize=10,
          interval=(0, 1000),
          iterNum=10,
          return_state=False):

    # initial population
    if not initPop:
        population = population = np.random.random(
            popSize) * (interval[1]-interval[0]) + interval[0]
    else:
        population = initPop

    if return_state:
        state = []

    for it in range(iterNum):
        children = population.copy()
        # cross
        children = crossFunc(children, **CFKwargs)
        # mutate
        children = mutationFunc(children, **MFKwargs)
        # merge
        population = np.unique(np.concatenate([population, children]))
        # get objective
        objectives = objsFunc(population)
        # pareto sort
        front = fast_non_dominated_sort(objectives)
        # compute distance
        distance = crowding_distance(objectives, front)
        # elite select
        next_idx = []
        ptr = 0

        for fidx in front:
            if ptr+len(fidx) > popSize:
                if ptr == popSize:
                    break
                # distance
                rank_dis = distance[fidx]
                next_idx.append(fidx[np.argsort(rank_dis)[ptr-popSize:]])
                break
            else:
                next_idx.append(fidx)
                ptr+=len(fidx)

        if return_state:
            state.append([population[rank_idx] for rank_idx in next_idx])

        population = population[np.concatenate(next_idx)]

    if return_state:
        return population, state

    return population
