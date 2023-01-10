import sys
sys.path.append("..")
import simeck as sk
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf 
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
import multiprocessing as mp
import gc


WORD_SIZE = sk.WORD_SIZE()

def key_average(pairs, ctdata0l, ctdata0r, ctdata1l, ctdata1r, ks_nr):

    # print("############")
    rsubkeys = np.arange(0, 2**WORD_SIZE,dtype=np.uint16)
    # print("ks_nr",ks_nr)
    keys = rsubkeys ^ ks_nr
    # print("keys.shape",keys.shape)
    num_key = len(keys)
    # print("ctdata0l",ctdata0l)
    ctdata0l = np.tile(ctdata0l,num_key)
    ctdata0r = np.tile(ctdata0r,num_key)
    ctdata1l = np.tile(ctdata1l,num_key)
    ctdata1r = np.tile(ctdata1r,num_key)
    # print("ctdata0l",ctdata0l)
    keys = np.repeat(keys,pairs)
    # print("keys",keys)
    ctdata0l,ctdata0r = sk.dec_one_round_simeck((ctdata0l, ctdata0r), keys)
    ctdata1l,ctdata1r = sk.dec_one_round_simeck((ctdata1l, ctdata1r), keys)
    
    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
 
    
    secondLast_ctdata0r = sk.rol(ctdata0r, 5) & sk.rol(ctdata0r, 0) ^ sk.rol(ctdata0r, 1) ^ ctdata0l
    secondLast_ctdata1r = sk.rol(ctdata1r, 5) & sk.rol(ctdata1r, 0) ^ sk.rol(ctdata1r, 1) ^ ctdata1l
    
    delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
    
    thirdLast_ctdata0r = ctdata0r ^ sk.rol(secondLast_ctdata0r,5) & sk.rol(secondLast_ctdata0r,0) ^ sk.rol(secondLast_ctdata0r,1)
    thirdLast_ctdata1r = ctdata1r ^ sk.rol(secondLast_ctdata1r,5) & sk.rol(secondLast_ctdata1r,0) ^ sk.rol(secondLast_ctdata1r,1)    
        
    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = sk.convert_to_binary([delta_ctdata0l,delta_ctdata0r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_secondLast_ctdata0r,delta_thirdLast_ctdata0r]);
    # print("X.shape",X.shape)

    X = X.reshape(num_key,pairs,WORD_SIZE*8)
    X = X.reshape(num_key,pairs*WORD_SIZE*8)
     
    return X

def predict(X, net,bs):

 
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:1", "/gpu:2","/gpu:4","/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X, batch_size=batch_size)

    return Z

def wrong_key_decryption(net,bs,n=3000,pairs=8,diff=(0x0000,0x0040),nr=7):
    
    # 生成需要测试的明文和密文
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16)
    keys = np.repeat(keys,pairs).reshape(4,-1)
    # print(keys[0][:8])
    pt0l = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt0r = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    # print("pt0l.shape",pt0l.shape)
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    ks = sk.expand_key_simeck(keys, nr+1)
    # 生成nr+1轮密文，然后用不同密钥进行解密
    ct0l, ct0r = sk.encrypt_simeck((pt0l, pt0r), ks)
    ct1l, ct1r = sk.encrypt_simeck((pt1l, pt1r), ks)
    # print("ct0l.shape",ct0l.shape)

    slices = 10

    ct0l = ct0l.reshape(slices, -1)
    ct0r = ct0r.reshape(slices, -1)
    ct1l = ct1l.reshape(slices, -1)
    ct1r = ct1r.reshape(slices, -1)

    nr_key = np.copy(ks[nr])
    # print("nr_key.shape",nr_key.shape)
    nr_key = nr_key.reshape(slices,-1)
    # print("nr_key.shape",nr_key.shape)
    # 开始执行多进程
    process_number = mp.cpu_count()-4
  
    Z = []
    for i in range(slices):

        pool = mp.Pool(process_number)
        X = pool.starmap(key_average, [
                             (pairs, ct0l[i][j*pairs:(j+1)*pairs], ct0r[i][j*pairs:(j+1)*pairs], \
             ct1l[i][j*pairs:(j+1)*pairs], ct1r[i][j*pairs:(j+1)*pairs], nr_key[i][j*pairs],) for j in range(int(n/slices))])
        print("multiple processing end ......")

        pool.close()
        pool.join()

        X = np.array(X).flatten()

        num_keys=2**16
        X = X.reshape(int(n/slices)*num_keys,pairs*WORD_SIZE*8)
        Z.append(predict(X, net,bs))
        del X 
        gc.collect()

    Z = np.array(Z).flatten()
    Z = Z.reshape(n,-1)
    mean = np.mean(Z,axis=0)
    std = np.std(Z,axis=0)

    print("mean shape",mean.shape)
    print("std shape",std.shape)


    return mean, std


if __name__ == "__main__":


    bs = 4000
    num = 3000
    pairs = 8
    num_rounds = 9
    # 读取模型网络参数
    wdir = "/home/data/zhangliu/cryptanalysis/deep_learning_cryptanalysis/multiple_pairs_simeck/simeck_input_format_8/key_recovery_attack/"
    net = load_model("simeck32_best_model_9r_depth5_num_epochs20_pairs8_acc_0.9954144954681396.h5")
    m,s = wrong_key_decryption(net=net,bs=bs,n=num, pairs = pairs, diff=(0x0000,0x0040), nr=num_rounds)
    np.save(wdir+"simeck"+str(sk.WORD_SIZE()*2)+"_data_wrong_key_mean_"+str(num_rounds)+"r_pairs"+str(pairs)+".npy",m)
    np.save(wdir+"simeck"+str(sk.WORD_SIZE()*2)+"_data_wrong_key_std_"+str(num_rounds)+"r_pairs"+str(pairs)+".npy",s)
     