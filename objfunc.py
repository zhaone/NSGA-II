#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 zhaoyi <yizhaome@gmail.com>
#
# Distributed under terms of the Apache V2.0 license.
import numpy as np

# 10 <= x <= 10
def sch1_objectives(x):
    return np.array([x**2, (x-2)**2]).T

# -5 <= x <= 10
def sch2_objectives(x):
    res = x.copy()
    mask1 = x <= 1
    res[mask1] = -x[mask1]
    mask2 = np.logical_and(x > 1, x <=3)
    res[mask2] = x[mask2] - 2
    mask3 = np.logical_and(x > 3, x <= 4)
    res[mask3] = 4 - x[mask3]
    mask4 = x > 4
    res[mask4] = x[mask4] - 4

    return np.array([res, (x-5)**2]).T

def toy_objectives(x):
    return np.array([(x-400)**2, (x-600)**2]).T
