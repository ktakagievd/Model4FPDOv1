'''
Created on 2024/09/02

@author: K.Takagi
'''
import os
import numpy as np

def getroundint(n):
    n=np.round(n)
    n=int(n)
    return n

def checkfolderlist(fols):
    ""
    for fol in fols:
        checkfolder(fol)

def checkfolder(fol):
    sfols=str(fol).split("/")
    fol="./"
    for i, sf in enumerate(sfols):
        if i>0:
            fol=os.path.join(fol,sf)
            if not os.path.exists(fol):
                os.mkdir(fol)
            #print(fol)



    