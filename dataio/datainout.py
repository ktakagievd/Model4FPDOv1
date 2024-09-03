'''
Created on 2024/09/02

@author: K.Takagi
'''
import os
import numpy as np

from utlfunc import getroundint

def loaddataset_img0(param_setting):
    datfol=param_setting.path_inputdat
    fn_img=os.path.join(datfol,"img0.npy")
    img=np.load(fn_img)
    fn_lbl=os.path.join(datfol,"label0.npy")
    lbl=np.load(fn_lbl)
    return img, lbl

def loadgeneinformation(param_setting):
    datfol=param_setting.path_inputdat
    fn_geninf=os.path.join(datfol,"geneinf.npy")
    geninf=np.load(fn_geninf,allow_pickle=True)
    return geninf

def loaddataset_img(param_setting):
    img0, lbl=loaddataset_img0(param_setting)
    img=normalizeimagedat(img0)
    return img, lbl

def normalizeimagedat(img0):
    
    dat=castdtas(img0)
    ret=norm_01(dat)
    return ret


def castdtas(dat):
    ret=dat.astype(np.float32)
    return ret


def norm_01(im):
    ma=np.max(im)
    mi=np.min(im)
    d=ma-mi
    if d==0:
        d=1
    im=(im-mi)/d
    return im


def loaddataset_gene(param_setting):
    datfol=param_setting.path_inputdat
    fn_gex=os.path.join(datfol,"gene_exp.npy")
    gex=np.load(fn_gex)
    return gex

def getgeneexpdata(lab, gex):
    pret=getlabeldat(lab, gex)
    return setnormalizedat(pret)
    
def setnormalizedat(gex):
    gex=normalizemaxat(gex)
    gex = gex.astype(np.float32)
    gex = np.nan_to_num(gex)
    return gex


def normalizemaxat(x_train):
    xm=np.max(np.abs(x_train))
    if xm==0:
        xm=1
    x_train=x_train/xm
    return x_train

def getlabeldat(lab, edat):
    ret=[]
    for l in lab:
        if l<edat.shape[1]:
            pans=edat[:,l]
            ret.append(pans)
        else:
            pans=np.zeros(edat.shape[0])
            print("Lable not found {}".format(l))            
    ret=np.array(ret)
    return ret

"""
def splitdataset_imglabel(rt, dat, lab):
    rp0=np.random.permutation(len(dat))
    n0=len(dat)
    
    n1=getroundint(n0*(1-rt))
    n2=getroundint(n0*rt)
    rp1=rp0[0:n1]
    rp2=rp0[0:n2]
    dat_t=dat[rp1,:,:,:]
    lab_t=lab[rp1]
    dat_v=dat[rp2,:,:,:]
    lab_v=lab[rp2]
    return dat_t, lab_t, dat_v, lab_v
"""
def getrandomsampling(dats,nmax):
    ret=[]
    if len(dats)>0:
        n0=len(dats[0])
        rp0=np.random.permutation(n0)
        n1=nmax
        rp1=rp0[0:n1]
        for dt in dats:
            pret=setpartialdat(rp1, dt)
            ret.append(pret)
    return ret


def setpartialdat(rp1, dt):
    pret=[]
    if len(dt.shape)==1:
        pret=dt[rp1]
    if len(dt.shape)==2:
        pret=dt[rp1,:]
    if len(dt.shape)==3:
        pret=dt[rp1,:,:]
    if len(dt.shape)==4:
        pret=dt[rp1,:,:,:]
    return pret

def splitdatafortrain(rt, dats):
    ret_train=[]
    ret_valid=[]
    if len(dats)>0:
        n0=len(dats[0])
        rp0=np.random.permutation(n0)
        n1=getroundint(n0*(1-rt))
        n2=getroundint(n0*rt)
        rp1=rp0[0:n1]
        rp2=rp0[0:n2]
        for dt in dats:
            pret_t=setpartialdat(rp1, dt)
            pret_v=setpartialdat(rp2, dt)
            ret_train.append(pret_t)
            ret_valid.append(pret_v)
    return ret_train, ret_valid