from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import simeck32 as sk
import numpy as np
import os
bs = 5000
src_dir = '../good_trained_nets/'
wdir = './temp/'
# 不断修改学习率

strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
print('Number of devices: %d' % strategy.num_replicas_in_sync) 
batch_size = bs * strategy.num_replicas_in_sync
def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

def first_stage(n,num_rounds=12,pairs=8):

    
    test_n = int(n/10) 
    X, Y = sk.make_train_data(n, nr=num_rounds-3, pairs=pairs,diff=(0x0140, 0x0080))
    X_eval,Y_eval= sk.make_train_data(test_n, nr=num_rounds-3, pairs=pairs,diff=(0x0140, 0x0080))
    
    
    with strategy.scope():

        net = load_model("simeck"+str(sk.WORD_SIZE()*2)+"_best_model_"+str(num_rounds-1)+"r_depth5_num_epochs20_pairs"+str(pairs)+"_acc_0.7374.h5")
        net_json = net.to_json()
        net_first = model_from_json(net_json)
        net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_first.load_weights("simeck"+str(sk.WORD_SIZE()*2)+"_best_model_"+str(num_rounds-1)+"r_depth5_num_epochs20_pairs"+str(pairs)+"_acc_0.7374.h5")

    check = make_checkpoint(
        wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_first.fit(X, Y, epochs=10, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval),callbacks=[lr,check])

    print("################################################")
    # net_first.save(wdir+"net_first.h5")


def second_stage(n,num_rounds=12, pairs=8):

    
    test_n = int(n/10)
    X, Y = sk.make_train_data(n, nr=num_rounds, pairs=pairs)
    X_eval, Y_eval = sk.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    
    with strategy.scope():

        net = load_model(wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
        net_json = net.to_json()

        net_second = model_from_json(net_json)
        # net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_second.load_weights(wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
        
    
    check = make_checkpoint(
        wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.0001, 0.00001))
    net_second.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[lr,check])
    print("################################################")
    # net_second.save(wdir+"net_second.h5")


def stage_train(n,num_rounds=12, pairs=8):

    
    test_n = int(n/10)
    X, Y = sk.make_train_data(n, nr=num_rounds, pairs=pairs)
    X_eval, Y_eval = sk.make_train_data(test_n, nr=num_rounds, pairs=pairs)
    
    with strategy.scope():

        net = load_model( wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
        net_json = net.to_json()

        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        net_third.load_weights( wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    src = "staged_simeck"+str(sk.WORD_SIZE()*2)+"_best_model_"+str(num_rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)
    check = make_checkpoint(src+'.h5')
    h = net_third.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src +'.h5' , dst+'.h5')
    print("Best validation accuracy: ", np.max(h.history['val_acc']))

    # net_third.save(wdir+"simon_model_"+str(num_rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
   

for i in range(2):

    first_stage(n=2*10**7, num_rounds=11,pairs=8)
    second_stage(n=2*10**7,num_rounds=11,pairs=8)
    stage_train( n=2*10**7,num_rounds=11,pairs=8)