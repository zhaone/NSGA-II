#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 zhaoyi <yizhaome@gmail.com>
#
# Distributed under terms of the Apache V2.0 license.
import numpy as np

def sch_objectives(x):
    return np.array([x**2, (x-2)**2]).T

def toy_objectives(x):
    return np.array([(x-400)**2, (x-600)**2]).T
