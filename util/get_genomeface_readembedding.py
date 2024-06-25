import kmer_counter
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

class PIBlock2(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim,input_dim=None):
        super(PIBlock2, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="tanh"), layers.Dense(embed_dim),]
        )
        self.bn = layers.BatchNormalization()
        self.s = embed_dim
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        return out1 + ffn_output
    def compute_output_shape(self, input_shape):
        return self.ffn[-1].output.shape

path ='/big/work/mcdevol/scripts/src/data/'
# aaq = kmer_counter.find_nMer_distributions(path+'contigs_2k.fasta', 2000)
# aaq = kmer_counter.find_nMer_distributions('/big/work/mcdevol/scripts/src/subset_contigs.fasta', 2000)
import numpy as np
names = np.load(path +'contigs_2knames.npz', allow_pickle=True)['arr_0']
length = np.load(path +'contigs_2klength.npz', allow_pickle=True)['arr_0']
contig_names = names #np.asarray(aaq[-1])
contig_lens = length #np.asarray(aaq[0])
num_contigs = len(contig_names)
# inpts = [np.reshape(aaq[i], (-1, size)) for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]

n_samples = 10

depth_data = np.load(path +'mcdevol_readcounts.npz', allow_pickle=True)['arr_0']
print(depth_data.shape, 'depth data shape')
model_data_in = [np.ones((num_contigs, 136), dtype='float')]
# model_data_in = [np.zeros((num_contigs, 136), dtype='float')]
model_data_in.append(depth_data.reshape((-1, n_samples, 1)))
with tf.device('/cpu:0'):
    datasets = [tf.data.Dataset.from_tensor_slices(arr) for arr in model_data_in]
dataset = tf.data.Dataset.zip(tuple(datasets))
dataset = dataset.batch(8192 // 2)

datasets = [tf.data.Dataset.from_tensor_slices(arr) for arr in model_data_in]
dataset = tf.data.Dataset.zip(tuple(datasets))
batch_size = 8192 // 2
dataset = dataset.batch(batch_size)


depth_input = Input(shape=(n_samples, 1))
kmer_inputs = [Input(shape=(v,)) for v in [512,136,32,10,2,528,256,136,64,36]]
# Abundance model
y = layers.TimeDistributed(layers.Dense(16, activation='linear'))(depth_input)
for _ in range(4):
    y = PIBlock2(16, 16, 512)(y)

y = layers.Flatten()(y)
y = tf.math.l2_normalize(y, axis=1)

model_adb_eval = Model([Input(shape=(136,)), depth_input], y)
model_adb_eval.compile()
path_weight = "/big/software/genomeface/share/genomeface/weights/model9_eval.m"
model_adb_eval.load_weights(path_weight)
model_adb_eval.summary()


y21_cat = np.zeros((num_contigs, 16 * n_samples))
done = 0
for idx, b in enumerate(dataset):
    otmp = model_adb_eval.predict(x=b, verbose=0, batch_size=8192 // 2)
    y21_cat[done:done + len(otmp), :] = otmp
    done += batch_size

y21_cat /= np.linalg.norm(y21_cat, axis=1, keepdims=True)
# np.save('/big/work/mcdevol/scripts/src/genomeface_kmer_embedding_depth1.npy',y20_cat)
np.save('/big/work/mcdevol/scripts/src/genomeface_readembed.npy',y21_cat)
print(y21_cat.shape)