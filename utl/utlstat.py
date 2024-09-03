'''
Created on 2024/09/03

@author: K.Takagi
'''
import numpy as np
from scipy.stats.stats import pearsonr
from datainout import getrandomsampling

def getpcor(p0,a0):
    p=np.reshape(p0, (-1))
    a=np.reshape(a0, (-1))
    ret=pearsonr(p,a)[0]
    return ret

def getpcorsamplingmax(nmax, p0,a0):
    p=np.reshape(p0, (-1))
    a=np.reshape(a0, (-1))
    if len(p)>nmax:
        smpl=getrandomsampling([p,a],nmax)
        p=smpl[0]
        a=smpl[1]
    ret=pearsonr(p,a)[0]
    return ret
