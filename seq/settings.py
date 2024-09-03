'''
Created on 2024/09/02

@author: K.Takagi
'''

class progsetting(object):
    #main datasets
    path_main="./dataset/"
    path_inputdat="./dataset/input/"
    
    #parameters for DL models
    aires_imgrep="./dataset/res/ai/imgrep/"
    aires_img2gene="./dataset/res/ai/img2gene/"
    
    epoch_img2gene=15
    batch_img2gene=50
    epoch_imgrep=100
    batch_imgrep=100    
    valid_rt=0.1#ratio of validation data size
    
    
    #parameters for gene selection
    gsel_tmp="./dataset/res/geneselect/tmp/"
    gsel="./dataset/res/geneselect/"
    genesel_threshold=0.5
    genesel_maxgenenum=100#maximum number of genes to output
    def __init__(self):
        '''
        Constructor
        '''
