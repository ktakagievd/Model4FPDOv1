'''
Created on 2024/09/02

@author: K.Takagi
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax
from tensorflow.keras import Model


from utlfunc import getroundint
from utltf import initializeGPU
from datainout import splitdatafortrain

def trainimagerep(param_setting, outpath, dat, lab):
    ""
    
    dat_train, dat_valid=splitdatafortrain(param_setting.valid_rt, [dat, lab])   
    exetrainmodelimgrep(param_setting,outpath, dat_train[0], dat_train[1], dat_valid[0], dat_valid[1])

def exetrainmodelimgrep(param_setting,outpath, i_train, l_train, i_valid, l_valid):
    latdim=np.max(l_train)+1
    EPOCHS = param_setting.epoch_imgrep#100
    bsize=param_setting.batch_imgrep#100
    
    model = ImageRepModel(latdim)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')    
    nn=len(i_train)
    lpsize=nn/bsize
    lpsize=int(lpsize)
    epoch_showres=getroundint(EPOCHS/10) 
    for epoch in range(EPOCHS):
        for i in range(lpsize):
            rp1=np.random.permutation(len(i_train))
            p1=rp1[0:bsize]
            train_step(model,i_train[p1,:,:,:], l_train[p1], loss_object, optimizer, train_loss, train_accuracy)
            
        if epoch%epoch_showres==0 or epoch==EPOCHS-1:
            #model.save_weights(outpath)
            aa=getmodelaccuracy(model, i_valid, l_valid)
            print("Epoch: {} \t loss: {} \t Acc.:{} \t Acc (train).{} ".format(epoch,  train_loss.result().numpy(),aa, train_accuracy.result().numpy())) 

        train_loss.reset_states()
        train_accuracy.reset_states()

            
    model.save_weights(outpath)

def loadimagerep(pathfol,   latdim):
    model = ImageRepModel(latdim)
    model.load_weights(pathfol)
    return model

##load ai model and get image representation
def getimagerep(pathfol,img,label):
    latdim=np.max(label)+1
    
    initializeGPU()
    repdat=[]
    ne=getroundint(len(img)/1000)
    ne+=1
    model = loadimagerep(pathfol,latdim)
    for i in range(ne):
        ep=(i+1)*1000
        if ep>len(img):
            ep=len(img)
        sp=i*1000
        repdat0=model.getLat(img[sp:ep,:,:,:]).numpy()
        repdat.extend(repdat0)
    repdat=np.array(repdat)
    return repdat

    
def train_step(model,images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

def getmodelaccuracy(model, iv, lv):
    bc=model(iv,training=False).numpy()
    bc=np.argmax(bc, axis=1)
    aa=accur(bc,lv)
    return aa
    
    
    
def accur(b, l):
    ret1=0
    ret2=0
    for bb, ll in zip(b, l):
        if bb==ll:
            ret1+=1
        else:
            ret2+=1
    tret=ret1+ret2
    if tret==0:
        tret=1
    ret=ret1/tret
    return ret
    

class ImageRepModel(Model):
    def __init__(self, latdim):
        super(ImageRepModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, use_bias=None)
        self.d3 = Dense(latdim, use_bias=None)
        self.d4= Softmax()

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)

    def checkdim(self, x):
        x = self.conv1(x)
        print(" conv1 {}".format(x.shape))
        x = self.flatten(x)
        print(" flatten {}".format(x.shape))
        x = self.d1(x)
        print(" d1 {}".format(x.shape))
        x = self.d2(x)
        print(" d2 {}".format(x.shape))
        x = self.d3(x)

    def getnet_d(self):
        return self.d3
    
    def getOut(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    
    def getLat(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


