#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 zhaoyi <yizhaome@gmail.com>
#
# Distributed under terms of the Apache V2.0 license.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import figaspect

import time

# from nsga2ori import nsga2
from nsga2 import nsga2

# only for 2-d, 2-objective-func data
def visulize(objsFunc, states, interval=(0, 1000), showEvery=1):
    for iterNum, pop in enumerate(states):
        if iterNum % showEvery != 0 and iterNum != len(states)-1:
            continue
        fig = plt.figure(constrained_layout=True, figsize=figaspect(.5))
        gs = fig.add_gridspec(2, 2)
        ax_lu = fig.add_subplot(gs[0, 0])
        ax_lb = fig.add_subplot(gs[1, 0])
        ax_r = fig.add_subplot(gs[:, 1])
        
        cmaps = cm.get_cmap('viridis', len(pop))
        sca_cmaps = [np.expand_dims(np.array(cmaps(i)), axis=0) for i in range(len(pop))]

        x = np.arange(interval[0], interval[1])
        model_obj = objsFunc(x)
        ax_lu.plot(x, model_obj[:, 0])
        ax_lb.plot(x, model_obj[:, 1])

        for rank, front in enumerate(pop):
            objs = objsFunc(front)
            ax_lu.scatter(front, objs[:, 0], marker='o',
                       c=sca_cmaps[rank])
            ax_lb.scatter(front, objs[:, 1], marker='o',
                       c=sca_cmaps[rank])
            ojbsort = np.argsort(objs[:, 0])
            objs = objs[ojbsort, :]
            ax_r.plot(objs[:, 0], objs[:, 1], 'o--', c=cmaps(rank))

        ax_r.set_xlabel("obj1")
        ax_r.set_ylabel("obj2")
        ax_r.set_title('Pareto set at iter {}'.format(iterNum))
        ax_lu.set_xlabel('x')
        ax_lu.set_ylabel('obj1')
        ax_lu.set_title('object 1 value at iter {}'.format(iterNum))
        ax_lb.set_xlabel('x')
        ax_lb.set_ylabel('obj2')
        ax_lb.set_title('object 2 value at iter {}'.format(iterNum))
        
        # plt.savefig('./iter_{}.png'.format(iterNum))
        plt.show()


def sch1Test(popSize=10, iterNum=10):
    from objfunc import sch1_objectives
    finalPop, state = nsga2(objsFunc=sch1_objectives,
                            CFKwargs={'cross_prob': 0.2},
                            MFKwargs={'mute_prob': 0.01},
                            popSize=popSize,
                            interval=(-10, 10),
                            iterNum=iterNum,
                            return_state=True)

    visulize(sch1_objectives, state, interval=(-10, 10))


def sch2Test(popSize=10, iterNum=10):
    from objfunc import sch2_objectives
    finalPop, state = nsga2(objsFunc=sch2_objectives,
                            CFKwargs={'cross_prob': 0.2},
                            MFKwargs={'mute_prob': 0.01},
                            popSize=popSize,
                            interval=(-5, 10),
                            iterNum=iterNum,
                            return_state=True)

    visulize(sch2_objectives, state, interval=(-5, 10))
if __name__ == "__main__":
    sch2Test()
