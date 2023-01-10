import sys
sys.path.append("..")
import simeck as sk
import numpy as np
import gc

from tensorflow.keras.models import load_model
from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
import multiprocessing as mp
import tensorflow as tf 


WORD_SIZE = sk.WORD_SIZE()

# 读取模型网络参数
pairs = 8

m11 = np.load("simeck32_data_wrong_key_mean_11r_pairs8_acc_0.5663520097732544.npy");
s11 = np.load("simeck32_data_wrong_key_std_11r_pairs8_acc_0.5663520097732544.npy"); 

m10 = np.load("simeck32_data_wrong_key_mean_10r_pairs8_acc_0.7370569705963135.npy");
s10 = np.load("simeck32_data_wrong_key_std_10r_pairs8_acc_0.7370569705963135.npy"); 




s11 = 1.0/s11
s10 = 1.0/s10


# binarize a given ciphertext sample
# ciphertext is given as a sequence of arrays
# each array entry contains one word of ciphertext for all ciphertexts given


def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE
        offset = WORD_SIZE - 1 - i % WORD_SIZE
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)

# 汉明重量
def hw(v):
    res = np.zeros(v.shape, dtype=np.uint8)
    for i in range(16):
        res = res + ((v >> i) & 1)
    return(res)


# 将初始输入差分的wt值设置低一些。2^16-1
low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
# 真实密钥和猜测密钥的wt值相差不能超过2
low_weight = low_weight[hw(low_weight) <= 2]

# make a plaintext structure
# takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits

def make_structure(pt0, pt1, diff, neutral_bits):
    # p0和p1是分别是随机生成明文的左右两边
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for subset in neutral_bits:
        d0_sum = 0x0
        d1_sum = 0x0
        for i in subset:
            d = 1 << i
            # d0影响高位，d1控制低位
            d0 = d >> 16
            d1 = d & 0xffff

            d0_sum = d0_sum ^ d0
            d1_sum = d1_sum ^ d1
        p0 = np.concatenate([p0, p0 ^ d0_sum], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1_sum], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return(p0, p1, p0b, p1b)



def gen_key(nr):
    # 就是只要一个密钥
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    ks = sk.expand_key_simeck(key, nr)
    
    return(ks)

# 测试正确对
def test_correct_pairs(pt0l,pt0r,key, nr,diff,target_diff):
    
    
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    pt0l_1, pt0r_1 = sk.dec_one_round_simeck((pt0l,pt0r), 0)
    pt1l_1, pt1r_1 = sk.dec_one_round_simeck((pt1l, pt1r), 0)

    ct0l, ct0r = sk.encrypt_simeck((pt0l_1, pt0r_1), key[:nr])
    ct1l, ct1r = sk.encrypt_simeck((pt1l_1, pt1r_1), key[:nr])
    diff0 = ct0l ^ ct1l
    diff1 = ct0r ^ ct1r
    
    d0 = (diff0 == target_diff[0])
    d1 = (diff1 == target_diff[1])
    d = d0 * d1
   
    return(d)

def gen_plain(n):
    
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return(pt0, pt1)

def gen_challenge(pt0,pt1, key, diff, neutral_bits):
    #           明文对的数量 轮数          差分             中立bit                     

    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    pt0a, pt1a = sk.dec_one_round_simeck((pt0a, pt1a), 0)
    pt0b, pt1b = sk.dec_one_round_simeck((pt0b, pt1b), 0)
    # 加密了nr轮
    ct0a, ct1a = sk.encrypt_simeck((pt0a, pt1a), key)
    ct0b, ct1b = sk.encrypt_simeck((pt0b, pt1b), key)
    return([ct0a, ct1a, ct0b, ct1b])


# having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
def verifier_search(pairs,cts, best_guess, net, use_n):
    # 进来的数据就是一个密文结构内的数据
    # print("verifier search......")
    # print(best_guess);
    # 控制最多wt值相差2
    # ck1数量为137
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight

    n = len(ck1)
    ck1 = np.repeat(ck1, n)
    keys1 = np.copy(ck1)

    ck2 = np.tile(ck2, n)
    keys2 = np.copy(ck2)

    ck1 = np.repeat(ck1, pairs*use_n)
    ck2 = np.repeat(ck2, pairs*use_n)

    ct0a = np.tile(cts[0].transpose().flatten(), n*n)
    ct1a = np.tile(cts[1].transpose().flatten(), n*n)
    ct0b = np.tile(cts[2].transpose().flatten(), n*n)
    ct1b = np.tile(cts[3].transpose().flatten(), n*n)
    pt0a, pt1a = sk.dec_one_round_simeck((ct0a, ct1a), ck1)
    pt0b, pt1b = sk.dec_one_round_simeck((ct0b, ct1b), ck1)
    pt0a, pt1a = sk.dec_one_round_simeck((pt0a, pt1a), ck2)
    pt0b, pt1b = sk.dec_one_round_simeck((pt0b, pt1b), ck2)

    delta_pt0a = pt0a ^ pt0b
    delta_pt1a = pt1a ^ pt1b

    secondLast_pt1a = sk.rol(pt1a, 5) & sk.rol(pt1a, 0) ^ sk.rol(pt1a, 1) ^ pt0a
    secondLast_pt1b = sk.rol(pt1b, 5) & sk.rol(pt1b, 0) ^ sk.rol(pt1b, 1) ^ pt0b
    
    delta_secondLast_pt1a =  secondLast_pt1a ^ secondLast_pt1b
    
    thirdLast_pt1a = pt1a ^ sk.rol(secondLast_pt1a,5) & sk.rol(secondLast_pt1a,0) ^ sk.rol(secondLast_pt1a,1)
    thirdLast_pt1b = pt1b^ sk.rol(secondLast_pt1b,5) & sk.rol(secondLast_pt1b,0) ^ sk.rol(secondLast_pt1b,1)    
        
    delta_thirdLast_pt1a = thirdLast_pt1a ^ thirdLast_pt1b
    X = sk.convert_to_binary([delta_pt0a,delta_pt1a,pt0a,pt1a,pt0b,pt1b,delta_secondLast_pt1a,delta_thirdLast_pt1a]);
    del delta_pt0a,delta_pt1a,pt0a,pt1a,pt0b,pt1b,delta_secondLast_pt1a,delta_thirdLast_pt1a;gc.collect()

    X = X.reshape(n*n,use_n,pairs,16*8)
    X = X.reshape(n*n*use_n,pairs*16*8)
    Z = net.predict(X, batch_size=10000)
    del X;gc.collect() 
    Z = Z / (1 - Z)
    Z = np.log2(Z)
    Z = Z.reshape(-1, use_n)
    v = np.mean(Z, axis=1) * use_n    # 获取均值
    m = np.argmax(v)    # 均值最大值位置
    val = v[m]    # 均值最大值
    key1 = keys1[m]
    key2 = keys2[m]
    return(key1, key2, val)

# here, we use some symmetries of the wrong key performance profile
# by performing the optimization step only on the 14 lowest bits and randomizing the others
# on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
# In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br = np.arange(2**16, dtype=np.uint16) & 0xcfff
tmp_br = np.repeat(tmp_br, 32).reshape(-1, 32)

def bayesian_rank_kr(cand, emp_mean, m, s):
    # print("bayesian rank kr......")
    global tmp_br
    n = len(cand)
    if (tmp_br.shape[1] != n):
        tmp_br = np.arange(2**16, dtype=np.uint16) & 0xcfff
        tmp_br = np.repeat(tmp_br, n).reshape(-1, n)
    tmp = tmp_br ^ cand
    v = (emp_mean - m[tmp]) * s[tmp]
    v = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return(scores)

def bayesian_key_recovery(pairs,cts, num_cand, num_iter, m, s, net):
    # print("bayesian key recovery......")
    # num_cipher = 2**neutral_bits
    # ct shape =(pairs,2**neutral_bits)
    num_cipher = len(cts[0][0])
    # print("num_cipeher = ",num_cipher)
    # print("len(pairs)=%d,len(num_cipher)=%d" % (pairs, num_cipher))
    keys = np.random.choice(2**(WORD_SIZE), num_cand, replace=False) & 0xcfff
    scores = 0

    cts0=cts[0].transpose().flatten()
    cts1=cts[1].transpose().flatten()
    cts2=cts[2].transpose().flatten()
    cts3=cts[3].transpose().flatten()

    ct0a, ct1a, ct0b, ct1b = np.tile(cts0, num_cand), np.tile(
        cts1, num_cand), np.tile(cts2, num_cand), np.tile(cts3, num_cand)

    n = pairs*num_cipher

    scores = np.zeros(2**(WORD_SIZE))
    used = np.zeros(2**(WORD_SIZE))
    # 32*5=160个密钥
    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
    all_v = np.zeros(num_cand * num_iter)
    for i in range(num_iter):#这里的迭代是由于要随机便利最高2位

        k = np.repeat(keys, n)
        c0a, c1a = sk.dec_one_round_simeck((ct0a, ct1a), k)
        c0b, c1b = sk.dec_one_round_simeck((ct0b, ct1b), k)
        
        delta_c0a = c0a ^ c0b
        delta_c1a = c1a ^ c1b

        secondLast_c1a = sk.rol(c1a, 5) & sk.rol(c1a, 0) ^ sk.rol(c1a, 1) ^ c0a
        secondLast_c1b = sk.rol(c1b, 5) & sk.rol(c1b, 0) ^ sk.rol(c1b, 1) ^ c0b

        delta_secondLast_c1a =  secondLast_c1a ^ secondLast_c1b

        thirdLast_c1a = c1a ^ sk.rol(secondLast_c1a,5) & sk.rol(secondLast_c1a,0) ^ sk.rol(secondLast_c1a,1)
        thirdLast_c1b = c1b^ sk.rol(secondLast_c1b,5) & sk.rol(secondLast_c1b,0) ^ sk.rol(secondLast_c1b,1)    

        delta_thirdLast_c1a = thirdLast_c1a ^ thirdLast_c1b
        X = sk.convert_to_binary([delta_c0a,delta_c1a,c0a,c1a,c0b,c1b,delta_secondLast_c1a,delta_thirdLast_c1a]);
        del delta_c0a,delta_c1a,c0a,c1a,c0b,c1b,delta_secondLast_c1a,delta_thirdLast_c1a;gc.collect()
        X = X.reshape(num_cand,num_cipher,pairs*16*8)
        X = X.reshape(num_cand*num_cipher,pairs*16*8)

        Z = net.predict(X, batch_size=10000)  
        
        del X;gc.collect()
        
        Z = Z.reshape(num_cand, -1)
        # 对行求均值
        means = np.mean(Z, axis=1)
        Z = Z/(1-Z)
        Z = np.log2(Z)
        v = np.sum(Z, axis=1)
        all_v[i * num_cand:(i+1)*num_cand] = v
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
        scores = bayesian_rank_kr(keys, means, m=m, s=s)
        # 找到当前效果最好的密钥
        # tmp = np.argpartition(scores+used, num_cand)
        tmp = np.argpartition(scores, num_cand)
        # 重置密钥
        keys = tmp[0:num_cand]
        r = np.random.randint(0, 4, num_cand, dtype=np.uint16)
        r = r << 12
        keys = keys ^ r
    return(all_keys, scores, all_v)


def test_bayes(device,cts,it, cutoff1, cutoff2, net, net_help, m_main, m_help, s_main, s_help):
    # print("test bayes......")
    n = len(cts[0])
    n = int(n/pairs)
    verify_breadth = len(cts[0][0])
    alpha = sqrt(n);
    best_val = -1000.0;    # 这个参数有时需要适当调整
    best_key = (0, 0)
    best_pod = 0    # 密文结构
    # keys = np.random.choice(2**WORD_SIZE, 32, replace=False)
    eps = 0.001
    local_best = np.full(n, -10)    # 第i个密文结构最高的区分器得分
    num_visits = np.full(n, eps)    # 第i个密文结构迭代的次数

    for j in range(it):
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits)
        i = np.argmax(priority)        # 优先测试第i个密文结构
        # print("cipher structure = ",i)
        num_visits[i] = num_visits[i] + 1
        if (best_val > cutoff2):
            improvement = (verify_breadth > 0)
            while improvement:                # 到了最后阶段，验证猜测的密钥是否正确
                # print("the device " +str(device) + " best_pod = "+str(best_pod))                # best_pod为不同的密文结构
                k1, k2, val = verifier_search(pairs,[cts[0][best_pod::n], cts[1][best_pod::n], cts[2]
                                              [best_pod::n], cts[3][best_pod::n]], best_key, net=net_help, use_n=verify_breadth)
                # print("val = ", val)
                improvement = (val > best_val)
                if (improvement):
                    best_key = (k1, k2);best_val = val;
            return(best_key, j)
        # keys shape = (num_cand*num_iter)
        keys, scores, v = bayesian_key_recovery(
            pairs,[cts[0][i::n], cts[1][i::n], cts[2][i::n], cts[3][i::n]], num_cand=32, num_iter=5, net=net, m=m_main, s=s_main)# num_iter是根据不敏感的bit数进行调整的
        vtmp = np.max(v)
        # if vtmp > 0:
        #     print("vtmp = ",vtmp)
        # print("vtmp = ",vtmp)
        if (vtmp > local_best[i]):
            local_best[i] = vtmp
        if (vtmp > cutoff1):
            # print("in 2 round")
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
            for i2 in l2:
                # print("first key %x" % keys[i2])
                # print("vtmp = ",vtmp)
                c0a, c1a = sk.dec_one_round_simeck((cts[0][i::n], cts[1][i::n]), keys[i2])
                c0b, c1b = sk.dec_one_round_simeck((cts[2][i::n], cts[3][i::n]), keys[i2])

                keys2, scores2, v2 = bayesian_key_recovery(
                    pairs,[c0a, c1a, c0b, c1b], num_cand=32, num_iter=5, m=m_help, s=s_help, net=net_help)# num_iter是根据不敏感的bit数进行调整的
                vtmp2 = np.max(v2)
                # print("second key %x" % keys2[np.argmax(v2)])
                # if vtmp2 > 0:
                #     print("vtmp2 = ",vtmp2)
                # print("vtmp2 = ",vtmp2)
                if (vtmp2 > best_val):
                    best_val = vtmp2
                    best_key = (keys[i2], keys2[np.argmax(v2)])
                    best_pod = i
    improvement = (verify_breadth > 0)
    while improvement:
        # print("the device " +str(device) + " best_pod = "+str(best_pod))
        k1, k2, val = verifier_search(pairs,[cts[0][best_pod::n], cts[1][best_pod::n], cts[2]
                                      [best_pod::n], cts[3][best_pod::n]], best_key, net=net_help, use_n=verify_breadth)
        # print("val = ", val)
        improvement = (val > best_val)
        if (improvement):
            best_key = (k1, k2)
            best_val = val
    return(best_key, it)


def test(device,n=20,nr=16, num_structures=2**10, it=2**11, cutoff1=10, cutoff2=10):
    #         轮数  结构数                迭代轮数    c1界          c2界                             

    #  2-round differential probability is 2^(-4)
    #  3-round differential probability is 2^(-8)
    #  4-round differential probability is 2^(-12)

    gpus = tf.config.list_physical_devices(device_type = 'GPU')
    tf.config.set_visible_devices(devices=gpus[device],device_type='GPU')
    net11 = load_model("staged_simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5663520097732544.h5")
    net10 = load_model("simeck32_best_model_10r_depth5_num_epochs20_pairs8_acc_0.7370569705963135.h5")
    net = net11;net_help = net10
    m_main=m11;s_main=s11;  
    m_help=m10;s_help=s10
    # neutral_bits = [[7],[8],[9],[13],[14],[15],[18],[20],[22],[24],[30],[0,31],[10,25]]
    neutral_bits = [[7],[8],[9],[13],[14],[15],[18],[20],[22],[24]]

    arr1 = np.zeros(n, dtype=np.uint16)
    arr2 = np.zeros(n, dtype=np.uint16)
    # 记录开始时间
    t0 = time()
    data = 0
    # 测试n次
    for i in range(n):
        print("Test:", i)
        # 使用相同密钥加密的密文
        pt0, pt1 = gen_plain(num_structures)
        key = gen_key(nr)

        td = test_correct_pairs(pt0,pt1,key,nr=4,diff=(0x0140, 0x0200),target_diff=(0x0000, 0x0040))
        g = np.sum(td)
        if g == 0:
            arr1[i]=0xffff
            arr2[i]=0xffff
        #     print("don't have correct ciphertext structure")
        # else :
        #     print("the device "+str(device)+" td == 1 indice "+str(np.where(td==1)))

        # pt0 = pt0[np.where(td==1)]
        # pt1 = pt1[np.where(td==1)]

        # 生成多明文对
        pt0,pt1,_,_ = make_structure(pt0,pt1,diff=(0x0140, 0x0200),neutral_bits=[[3],[4],[5]])
        pt0 = pt0.transpose().flatten()
        pt1 = pt1.transpose().flatten()
        # 生成多密文对
        ct = gen_challenge(pt0,pt1,key,diff=(0x0140, 0x0200),neutral_bits=neutral_bits)
        # print("true_key  %x %x" % (key[nr-1],key[nr-2]))
        
        guess, num_used = test_bayes(device,ct,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help,
                                     m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help)
        # print("num_used = ", num_used)
        # 这里算的是数据复杂度，所以要取最小值，num_used的值可能会超过100，但数据结构数最多是100.
        del pt0,pt1,ct
        gc.collect()
        num_used = min(num_structures, num_used)
 
        data = data + 2 * (2 ** len(neutral_bits)) * num_used * pairs
        # 两轮密钥猜测结果和真实结果对比
        # 因为key[nr-1]存储了pairs个相同的密钥
        arr1[i] = guess[0] ^ key[nr-1]
        arr2[i] = guess[1] ^ key[nr-2]
        # print("guess 1 round key %x" %guess[0])
        print("Difference between real key and key guess: ",
              hex(arr1[i]), hex(arr2[i]))
    t1 = time()
    # print("Done.")
    d1 = [hex(x) for x in arr1]
    d2 = [hex(x) for x in arr2]
    # print("Differences between guessed and last key:", d1)
    # print("Differences between guessed and second-to-last key:", d2)
    print("Time per attack (average in seconds):", (t1 - t0)/n)
    # 对数除法变加法
    print("Data blocks used (average, log2): ", log2(data) - log2(n))
    # return(arr1, arr2, good)
    return(arr1, arr2)




if __name__ == "__main__":

    
    neutral_bits = [[3],[4],[5],[7],[8],[9],[13],[14],[15], [18],[20],[22],[24],[30],[0, 31],[10, 25]]
    success_rate = []
    nr = 16

    PN = 6
    pool = mp.Pool(PN)
    idx_range = [0,1,2,4,5,6]
    results = pool.starmap(test,[(device,) for device in idx_range])
    results = np.array(results)

    arr1 = []
    arr2 = []
    for result in results:
        for arr in result[0]:
            arr1.append(arr)
        for arr in result[1]:
            arr2.append(arr)
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    print("cutoff_success rate",np.sum(arr1==0)/len(arr1))
    success_rate.append(np.sum(arr1==0)/len(arr1))
    np.save(open(str(nr)+'round_run_sols.npy', 'wb'), [arr1,arr2])