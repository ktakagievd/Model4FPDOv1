'''
Created on 2024/09/02

@author: K.Takagi
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from utlfunc import getroundint
from utlstat import getpcor
from datainout import splitdatafortrain
from utltf import initializeGPU

def traingenexp(param_setting,imgrep,gex):
    dts_train, dt_valid=splitdatafortrain(param_setting.valid_rt, [imgrep,gex])
    outpath=param_setting.aires_img2gene
    #print("#{} \t {} \t {} \t {} ".format(dts_train[0].shape,dts_train[1].shape,dt_valid[0].shape,dt_valid[1].shape))
    exetraingenexp(param_setting, outpath, dts_train[0],dts_train[1],dt_valid[0],dt_valid[1])


##load ai model and get gene prediction
def getgenepred(pathfol,imgrep,gex):
    outdim=gex.shape[1]
    initializeGPU()
    prddat=[]
    ne=getroundint(len(imgrep)/1000)
    ne+=1
    model = loadgenepredmodel(pathfol,outdim)
    for i in range(ne):
        ep=(i+1)*1000
        if ep>len(imgrep):
            ep=len(imgrep)
        sp=i*1000
        prddat0=model(imgrep[sp:ep,:],training=False).numpy()
        prddat.extend(prddat0)
    prddat=np.array(prddat)
    return prddat

def loadgenepredmodel(pathfol,outdim):
    model=mogel4img2gene(outdim)
    model.load_weights(pathfol)
    return model    

def exetraingenexp(param_setting, pathfol, img_train,gex_train,img_valid,gex_valid):
    ""
    EPOCHS=param_setting.epoch_img2gene#15
    btsize=param_setting.batch_img2gene#50
    outdim=gex_train.shape[1]
    model=mogel4img2gene(outdim)
    
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    shrep=1#interval for displaying results (loss accuracy)
    n_show=10
    if EPOCHS>n_show:
        shrep=getroundint(EPOCHS/10)
    nn=len(img_train)
    lpsize=nn/btsize
    lpsize=int(lpsize)
    
   
    
    for epoch in range(EPOCHS):
        rp1=np.random.permutation(nn)

        for l in range(lpsize):
            p1=rp1[l*btsize:btsize*(l+1)]
            sdt_in=img_train[p1,:]
            sdt_ans=gex_train[p1,:]
            train_step0mse(model,sdt_in, sdt_ans, optimizer, train_loss)
        if epoch%shrep==0:
            aa=getmodelaccuracy_gex(model, img_valid,gex_valid)
            print("Epoch: {} \t loss: {} \t Acc.:{} ".format(epoch,  train_loss.result().numpy(),aa)) 
            
            
        train_loss.reset_states()
#    model.save_weights(pathfol,encoding='utf_8')
    model.save_weights(pathfol)
    print("saved model path", pathfol)

def train_step0mse(model,sdt_in, sdt_ans,optimizer, train_loss):
    ap=[]

    loss_object=tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        predictions = model(sdt_in)
        loss = loss_object(sdt_ans, predictions)
        ap.append(loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)


def getmodelaccuracy_gex(model, img,gex):
    prd=model(img,training=False).numpy()
    ret=getpcor(prd,gex)
    return ret
    

class mogel4img2gene(Model):
    #image in gene out
    def __init__(self, outdim):
        super(mogel4img2gene, self).__init__()
        nn_dr=[10,18,54,162,outdim]
        self.gn1= Dense(nn_dr[0])
        self.gn2= Dense(nn_dr[1])
        self.gn3= Dense(nn_dr[2])
        self.gn4= Dense(nn_dr[3])
        self.gn5= Dense(nn_dr[4])
        
       

    def call(self, x2):
        z2=x2
        z1=z2
        z1=self.gn1(z1)
        z1=self.gn2(z1)
        z1=self.gn3(z1)
        z1=self.gn4(z1)
        z1=self.gn5(z1)

        return z1

    