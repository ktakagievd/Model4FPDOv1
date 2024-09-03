'''
Created on 2024/09/03

@author: K.Takagi
'''
import os
import numpy as np
import pandas as pd

from utlstat import getpcorsamplingmax
from datainout import loadgeneinformation

def exegeneselection(param_setting,pgex,gex):
    ""
    path_gsel_tmp=param_setting.gsel_tmp
    fn_genemet=os.path.join(path_gsel_tmp,"gene_metrics.npy")
    if not os.path.exists(fn_genemet):
        gmet=getgenepredictmetrics(pgex,gex)
        np.save(fn_genemet, gmet)
    else:
        gmet=np.load(fn_genemet)

    genesel_threshold=param_setting.genesel_threshold
    genesel_maxgenenum=param_setting.genesel_maxgenenum
    selg=getSelectedGene(gmet,genesel_threshold, genesel_maxgenenum)
    genelist=loadgeneinformation(param_setting)
    
    selectedgene=genelist[selg]
    path_gsel=param_setting.gsel
    fn_gsel_csv=os.path.join(path_gsel,"selectedgene.csv")
    (pd.DataFrame(selectedgene)).to_csv(fn_gsel_csv, index=None, header=None)
    print("Final Gene list")
    for sg in selectedgene:
        print(sg)
    


def getSelectedGene(gmet,th, maxnum):
    paccur=gmet[:,0]
    psd=gmet[:,2]
    ret,_=selectlistthreshold(paccur, psd, th, maxnum)
    return ret

def selectlistthreshold(ac, psd, rt0, maxnum):
    pp=np.argsort(ac)[::-1]
    vma=np.max(psd)
    rt=vma*rt0
    vn=np.where(psd>rt)[0]
    vn=len(vn)
    
    tmin=1.0
    sret=[]
    for p in pp:
        v=psd[p]
        if v>rt and len(sret)<len(ac) and len(sret)<maxnum:
            sret.append(p)
            tmin=ac[p]
    return sret, [rt,tmin]

def getgenepredictmetrics(pgex,gex):

    gn=gex.shape[1]
    ret=[]
    cres=[]
    for i in range(gn):
        p=pgex[:,i]
        a=gex[:,i]
        s=np.std(a)
        v=np.var(a)
        aa=np.mean(a)
        if aa==0:
            aa=1
        sa=s/aa
        va=v/aa
        c=getpcorsamplingmax(100000,p,a)
        if np.isnan(c):
            c=0
        pret=[c,v,s,va,sa]
        ret.append(pret)
        cres.append(c)
    ret=np.array(ret)
    return ret
    

