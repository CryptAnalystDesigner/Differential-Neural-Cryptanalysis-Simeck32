import simeck32 as sk
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model
import multiprocessing as mp
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable = True) 
def evaluate(net,X,Y):
    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:4","/gpu:5", "/gpu:6"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = 2000 * strategy.num_replicas_in_sync
    with strategy.scope():
        Z = net.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);
    
pairs = 8
num_rounds=11
wdir = "/home/data/zhangliu/cryptanalysis/deep_learning_cryptanalysis/multiple_pairs_simeck/good_trained_nets/"
net = load_model(wdir+"simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5662869811058044.h5")

X, Y = sk.make_train_data(10**6, num_rounds, pairs=pairs)

print('Testing neural distinguishers');
evaluate(net, X, Y);