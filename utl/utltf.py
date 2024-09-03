'''
Created on 2024/09/03

@author: K.Takagi
'''

import tensorflow as tf 

def initializeGPU():
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                #print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            ""
            #print("Not enough GPU hardware devices available")
    except:
        ""
        #print("GPU is not available")
        