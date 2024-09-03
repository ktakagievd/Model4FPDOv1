'''
Created on 2024/09/02

@author: K.Takagi
'''

from settings import progsetting
from imagerep import trainimagerep, getimagerep
from datainout import loaddataset_gene, loaddataset_img, getgeneexpdata
from predgenexp import getgenepred, traingenexp
from geneselseq import exegeneselection
from utlfunc import checkfolderlist


def exeprocess():
    ""
    step=-1#All: -1, 0: image representation learning, 1: predict gene expression, 2: select gene 

    param_setting=progsetting()#default setting for program     
    prepareresultfolder(param_setting)
    if step==0 or step==-1:
        trainimagerepresentation(param_setting)
    if step==1 or step==-1:
        traingeneexpression(param_setting)
    if step==2 or step==-1:
        selectgene(param_setting)

def prepareresultfolder(param_setting):
    fols=[]
    fols.append(param_setting.aires_imgrep)
    fols.append(param_setting.aires_img2gene)
    fols.append(param_setting.gsel_tmp)
    fols.append(param_setting.gsel)
    checkfolderlist(fols)

    

def trainimagerepresentation(param_setting):
    dat, lab=loaddataset_img(param_setting)
    outpath=param_setting.aires_imgrep
    trainimagerep(param_setting,outpath,dat,lab)
    
def traingeneexpression(param_setting):
    dat, lab=loaddataset_img(param_setting)
    path_aiimgrep=param_setting.aires_imgrep
    imgrep=getimagerep(path_aiimgrep,dat, lab)
    gex_def=loaddataset_gene(param_setting)
    gex=getgeneexpdata(lab, gex_def)
    traingenexp(param_setting,imgrep,gex)

def selectgene(param_setting):
    dat, lab=loaddataset_img(param_setting)
    path_aiimgrep=param_setting.aires_imgrep
    imgrep=getimagerep(path_aiimgrep,dat, lab)
    gex_def=loaddataset_gene(param_setting)
    gex=getgeneexpdata(lab, gex_def)
    
    path_aipredgene=param_setting.aires_img2gene
    pgex=getgenepred(path_aipredgene,imgrep, gex)
    exegeneselection(param_setting,pgex,gex)

    
    

    
    