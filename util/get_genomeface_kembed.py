import kmer_counter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# information returned as tuple of 12 data (11 kmer counts and one contig names)
# (
#     pre_l4n1mers.into_pyarray(py), -> length of contig
#     pre_5mers.into_pyarray(py), -> 512D
#     pre_4mers.into_pyarray(py), -> 136D
#     pre_3mers.into_pyarray(py), -> 32D
#     pre_2mers.into_pyarray(py), -> 10D
#     pre_1mers.into_pyarray(py), -> 2D
#     pre_10mers.into_pyarray(py), -> 528D
#     pre_9mers.into_pyarray(py), -> 256D
#     pre_8mers.into_pyarray(py), -> 136D
#     pre_7mers.into_pyarray(py), -> 64D
#     pre_6mers.into_pyarray(py), -> 36D
#     contig_names
# )
path ='/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/'
aaq = kmer_counter.find_nMer_distributions(path+'augment_seq1.fasta', 2000)
# aaq = kmer_counter.find_nMer_distributions('/big/work/mcdevol/scripts/src/subset_contigs.fasta', 2000)
import numpy as np
contig_names = np.asarray(aaq[-1])
contig_lens = np.asarray(aaq[0])
num_contigs = len(contig_names)
inpts = [np.reshape(aaq[i], (-1, size)) for i, size in enumerate([512, 136, 32, 10, 2, 528, 256, 136, 64, 36], start=1)]

n_samples = 10
# depth_data = np.ones((num_contigs, n_samples), dtype=np.float32)
# with depth not given, depth set to zero or depth set to one all gave identical kmer embedding.
# Hence, depth is not given to composition model


# generate num_C x 136 array filled with zeros
model_data_in = [np.zeros((inpts[0].shape[0], 136), dtype='float')]
for i in range(len(inpts)):
    model_data_in.append(inpts[i])
# model_data_in.append(depth_data.reshape((-1, n_samples, 1)))
datasets = [tf.data.Dataset.from_tensor_slices(arr) for arr in model_data_in]
dataset = tf.data.Dataset.zip(tuple(datasets))
batch_size = 8192 // 2
dataset = dataset.batch(batch_size)


# depth_input = Input(shape=(n_samples, 1))


kmer_inputs = [Input(shape=(v,)) for v in [512,136,32,10,2,528,256,136,64,36]]


# Compositional model
x = layers.Concatenate()(kmer_inputs)
x = layers.BatchNormalization()(x)

for units in [1024 * 4, 1024 * 8 * 2]:
    x = layers.Dense(units, activation='tanh', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

x = layers.Dense(512, use_bias=False, activation=None)(x)
x = layers.BatchNormalization()(x)

x = tf.math.l2_normalize(x, axis=1)

# model2 = Model([Input(shape=(136,)),*kmer_inputs, depth_input], x)
model2 = Model([Input(shape=(136,)),*kmer_inputs], x)
model2.compile()
path_weight = "/big/software/genomeface/share/genomeface/weights/general_t2eval.m"
model2.load_weights(path_weight) #"/pscratch/sd/r/richardl/trainings/genearl/general_t2eval.m")
model2.summary()
y20_cat = np.zeros((num_contigs, 512))
done = 0
for idx, b in enumerate(dataset):
    y20_cat[done : done + batch_size, :] = model2.predict(x=b, verbose=0, batch_size=batch_size)
    done += batch_size

y20_cat /= np.linalg.norm(y20_cat, axis=1, keepdims=True)
np.save('/big/work/mcdevol/scripts/src/genomeface_kmer_embedding_augment1.npy',y20_cat)
# np.save('/big/work/mcdevol/scripts/src/genomeface_kmersubset.npy',y20_cat)
print(y20_cat.shape)