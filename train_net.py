from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, Conv1D,Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pickle import dump
import tensorflow as tf
import simeck as sk
import numpy as np
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

bs = 5000
wdir = './good_trained_nets/'
word_size = sk.WORD_SIZE()

if(not os.path.exists(wdir)):
  os.makedirs(wdir)

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

#make residual tower of convolutional blocks
def make_resnet(pairs=2, num_blocks=4, num_filters=32, num_outputs=1, d1=512, d2=64, word_size=word_size, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):

    inp = Input(shape=(int(num_blocks * word_size * 2*pairs),))
    rs = Reshape((pairs, int(2*num_blocks), word_size))(inp)
    perm = Permute((1, 3, 2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters*2, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*2, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2

    dense0 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1, pairs=1):
    
    print("pairs = ", pairs)
    print("num_rounds = ", num_rounds)

    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3","/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(pairs=pairs, depth=depth, reg_param=10**-5,word_size=sk.WORD_SIZE())
        # net.summary()
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # X, Y = sk.make_train_data(2*10**7, num_rounds, pairs=pairs)
    # X_eval, Y_eval = sk.make_train_data(2*10**6, num_rounds, pairs=pairs)
    
    process_number = 50
    with mp.Pool(process_number) as pool:
        accept_XY = pool.starmap(sk.make_train_data, [(int(2*10**7/process_number),num_rounds,pairs,) for i in range(process_number)])

    X = accept_XY[0][0]
    Y = accept_XY[0][1]
    
    for i in range(process_number-1):
        X = np.concatenate((X,accept_XY[i+1][0]))
        Y = np.concatenate((Y,accept_XY[i+1][1]))
   
    with mp.Pool(process_number) as pool:
        accept_XY_eval = pool.starmap(sk.make_train_data, [(int(2*10**6/process_number),num_rounds,pairs,) for i in range(process_number)])

    X_eval = accept_XY_eval[0][0]
    Y_eval = accept_XY_eval[0][1]
    
    for i in range(process_number-1):
        X_eval = np.concatenate((X_eval,accept_XY_eval[i+1][0]))
        Y_eval = np.concatenate((Y_eval,accept_XY_eval[i+1][1]))
  
    print("multiple processing end ......")
  
    # print("make data over")
    
    src = wdir+'simeck'+str(sk.WORD_SIZE()*2)+'_best_model_'+str(num_rounds)+'r_depth'+str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)
    check = make_checkpoint(src+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    # net.save(wdir+'model_'+str(num_rounds)+'r_depth'+str(depth) +
    #      "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
         
    dump(h.history, open(wdir+'simeck'+str(sk.WORD_SIZE()*2)+'_hist'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    
    # 重命名文件
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src +'.h5' , dst+'.h5')
    
    # return(net, h)


if __name__ == "__main__":
    
    # rounds=[15,16,17,18]
    rounds=[13]
    pairs = 8
    for r in rounds:
        train_speck_distinguisher(num_epochs=20, num_rounds=r, depth=5, pairs=pairs)
