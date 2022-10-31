
import numba as nb
import numpy as np

# Uses C language code to calculate distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cosine

__all__ = (
    "wmin_dist", "wmin_dist_2d",
    "calculate_distances_and_get_top",
    "calculate_distances", "get_top_using_scores"
)


@nb.jit(nopython=True, cache=True, parallel=True)
def wmin_dist(u, v, p=2, w=None):
    """This function calculates distance between two points quicker
    using numba.

        Parameters:
            u: vector
            v: vector of same shape as `u` 
            p: The order of the norm of the difference
            w: weights of shape same as `v`
    """
    # https://stackoverflow.com/questions/53021522/how-to-speed-up-distance-matrix-calculation
    diff = u - v

    res = np.power(diff, p)
    if w is not None: res = np.multiply(res, w)

    result = res.sum()

    return np.abs(np.power(result, 1 / p))


@nb.jit(nopython=True, cache=True, parallel=True)
def wmin_dist_2d(u, v, p=2, w=None):
    """This function calculates distance between two points quicker
    using numba.

        Parameters:
            u: 2D array
            v: vector of same shape as `u` or broadcastable
            p: The order of the norm of the difference
            w: weights of shape same as `v`
    """
    # https://stackoverflow.com/questions/53021522/how-to-speed-up-distance-matrix-calculation
    diff = u - v

    res = np.power(diff, p)
    if w is not None: res = np.multiply(res, w)

    result = res.sum(axis=1)

    return np.abs(np.power(result, 1 / p))


def calculate_distances_and_get_top(calculated_embeddings, embeddings, type_="euclidean", top=1, batch_size=10_000,
                                    return_top_scores = False):
    """Calculate similarity between given sentence embeddings and
        embeddings of sentences to check against and give out scores.

        Parameters:
            calculated_embeddings: Embedding of sentence to get scores for.
            embeddings: Embeddings of sentences against which
                `calculated_embeddings` is scored.
            type_: Which method to use to get similarity.
                `dot` or `cosine` or `euclidean`.
    """
    if top == -1:
        top = len(embeddings)
    final_result = np.zeros((calculated_embeddings.shape[0], top))
    final_scores = np.zeros((calculated_embeddings.shape[0], len(embeddings)))

    count = 0
    while count <= calculated_embeddings.shape[0]:
        curr_embbds = calculated_embeddings[count : count+batch_size]
        result = calculate_distances(curr_embbds, embeddings, type_=type_)
        final_scores[count:count+batch_size, :] = result
        final_result[count:count+batch_size, :] = get_top_using_scores(result, top=top, similarity_type=type_).reshape((curr_embbds.shape[0], top))
        count += batch_size
    if return_top_scores:
        return final_result, final_scores
    else:
        return final_result


def calculate_distances(calculated_embeddings, embeddings, type_="euclidean"):
    """Calculate similarity between given sentence embeddings and
        embeddings of sentences to check against and give out scores.

        Parameters:
            calculated_embeddings: Embedding of sentence to get scores for.
            embeddings: Embeddings of sentences against which
                `calculated_embeddings` is scored.
            type_: Which method to use to get similarity.
                `dot` or `cosine` or `euclidean`.
    """

    if type_ == "dot":
        if len(calculated_embeddings.shape) > 1:
            result = np.dot(embeddings, calculated_embeddings.T)
        else:
            result = np.dot(embeddings, calculated_embeddings.reshape(-1, 1))

        return result
    elif type_ == "cosine":
        if len(calculated_embeddings.shape) > 1:
            result = cosine_similarity(calculated_embeddings, embeddings)
            # result = cdist(calculated_embeddings, embeddings, cosine)
        else:
            result = cosine_similarity([calculated_embeddings], embeddings)[0]
            # result = cdist([calculated_embeddings], embeddings, cosine)[0]

        return result
    elif type_ == "euclidean":
        if len(calculated_embeddings.shape) > 1:
            # result = euclidean_distances(calculated_embeddings, embeddings)
            # result = cdist(calculated_embeddings, embeddings, euclidean_distances)
            result = cdist(calculated_embeddings, embeddings, wmin_dist, p=2, w=None)
        else:
            # result = euclidean_distances([calculated_embeddings], embeddings)[0]
            # result = cdist([calculated_embeddings], embeddings, euclidean_distances)[0]
            result = cdist([calculated_embeddings], embeddings, wmin_dist, p=2, w=None)[0]

        return result


def get_top_using_scores(scores, index_labels=None, top=1, return_index=True, similarity_type='euclidean'):
    """Get top matches using scores and labels.

        Parameters:
            scores: Scores for every label in `index_labels`
            index_labels: Labels for all strings given string wa
                matched against.
            top: How many top labels in sorted order is needed.
            return_index: Weather to return indexes of top labels or not.
            similarity_type: Similarity type used to get `scores`.
    """
    if top == -1:
        top = len(scores)

    if len(scores.shape) == 1:
        if similarity_type in ["euclidean"]:
            top_idx = np.argsort(scores.reshape(-1))[:top]
        else:
            top_idx = np.argsort(scores.reshape(-1))[::-1][:top]

        if return_index and index_labels is not None:
            if top == 1:
                return top_idx[0], np.array(index_labels)[top_idx][0]
            else:
                return top_idx, np.array(index_labels)[top_idx]
        elif return_index and index_labels is None:
            if top == 1:
                return top_idx[0]
            else:
                return top_idx
        else:
            if top == 1:
                return np.array(index_labels)[top_idx][0]
            else:
                return np.array(index_labels)[top_idx]
    elif len(scores.shape) == 2:
        if similarity_type in ["euclidean"]:
            top_idx = np.argsort(scores)[:, :top]
        else:
            top_idx = np.argsort(scores)[:, ::-1][:, :top]

        if return_index and index_labels is not None:
            if top == 1:
                return top_idx.reshape(-1), np.array(index_labels)[top_idx.reshape(-1)]
            else:
                raise Exception("Not Implemented Yet.")
                #return top_idx.tolist(), np.array(index_labels)[top_idx].tolist()
        elif return_index and index_labels is None:
            if top == 1:
                return top_idx.reshape(-1)
            else:
                return top_idx
        else:
            if top == 1:
                return np.array(index_labels)[top_idx.reshape(-1)]
            else:
                raise Exception("Not Implemented Yet.")
                #return np.array(index_labels)[top_idx].tolist()



import tensorflow as tf
tf.config.set_visible_devices(
    [], device_type='GPU'
)
# # Multi Threading
# N_JOBS = 4
# tf.config.threading.set_inter_op_parallelism_threads(N_JOBS)
# tf.config.threading.set_intra_op_parallelism_threads(N_JOBS)
# Enable JIT Optimizer
# tf.config.optimizer.set_jit(True)
# # Enable soft placement
# tf.config.set_soft_device_placement(True)
try: tf.enable_eager_execution()
except: pass
# # Memory Growth
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     try: tf.config.experimental.set_memory_growth(device, True)
#     except: pass
import tensorflow_hub as hub

from tensorflow.keras.models import Model   # Keras is the new high level API for TensorFlow
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cosine

# from ml_models.helping_modules.distance_functions import wmin_dist, wmin_dist_2d

import gc
import numpy as np
import sys
import math
import os

import tensorflow as tf
tf.config.set_visible_devices(
    [], device_type='GPU'
)

__all__=("GoogleUniversalSentenceEncoder"
)

class GoogleUniversalSentenceEncoder(object):
    """
    """
    __slots__=("guse_embedder", "max_len")

    def __init__(self,model_url=None):
        """
        """
        if model_url is None:
            self.guse_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        else:
            self.guse_embedder = hub.load(model_url)
        
        self.max_len = 128 # Max number of tokens GUSE can embedd

    def _divide_sentences(self, lst, max_len=128):
        """
        """
        _current_string_len_div_mapping = np.zeros((len(lst),), dtype=np.int)
        final_lst = []
        
        _word_counts = [] # ToDo: word count != number of tokens made by GUSE.
        
        for i, string in enumerate(lst):
            string_list = string.split(" ")

            temp_lst = []
            itr = 0
            count = 0
            while itr < len(string_list):
                final_lst.append(" ".join(string_list[itr : itr+self.max_len]))
                
                _word_counts.append(len(string_list[itr : itr+self.max_len]))
                
                itr += self.max_len
                count += 1
            
            _current_string_len_div_mapping[i] = np.int(count)
        
        _word_counts = np.array(_word_counts)
        
        return final_lst, _word_counts, _current_string_len_div_mapping

    def _weighted_avg(self, embds, word_counts):
        """
        """
        # https://github.com/tensorflow/hub/issues/244
        embds = np.multiply(embds, np.sqrt(word_counts)[:, np.newaxis])

        embds = embds.sum(axis=0) / np.sqrt(word_counts.sum())

        return embds
    
    def _merge_embeddings_for_divisions(self, embeddings, _word_counts, _current_string_len_div_mapping):
        """
        """
        final_embeddings = np.zeros((
            len(_current_string_len_div_mapping),
            embeddings.shape[1]
        ))

        count_done = 0
        for itr, curr_len in enumerate(_current_string_len_div_mapping):
            final_embeddings[itr, :] = self._weighted_avg(
                embeddings[count_done : count_done+curr_len,:],
                _word_counts[count_done : count_done+curr_len]
                )
        
        return final_embeddings

    def _get_for_string(self, sentence):
        """
        """
        result = self.guse_embedder([sentence]).numpy()[0].astype(np.float32)

        return result
    
    def _get_for_list_raw(self, lst):
        """Get embeddings for list of strings in `lst`.
        """
        fast_calc = False
        if len(lst) < 1_000: fast_calc = True

        if not fast_calc:
            gc.collect()
            tf.keras.backend.clear_session()

        result = np.empty((len(lst), 512), dtype=np.float32)
            
        batch_size = 1_000
        for itr in range(0, len(lst), batch_size):
            temp_tf_res = self.guse_embedder(lst[itr : itr+batch_size])
            temp_np_res = temp_tf_res.numpy()
            temp_res = temp_np_res.astype(np.float32, copy=False)
            
            if itr+batch_size > len(lst):
                result[itr:, :] = temp_res
            else:
                result[itr : itr+batch_size] = temp_res

            if not fast_calc:
                tf.keras.backend.clear_session()
                del temp_res, temp_tf_res, temp_np_res
                gc.collect()
            
        return result

    def get(self, sentence, get_for_large_string=False):
        """
        """
        if get_for_large_string:
            list_of_sentences,_word_counts, _current_string_len_div_mapping = self._divide_sentences([sentence])
            
            results = self._get_for_list_raw(list_of_sentences)
            
            final_embeddings = self._merge_embeddings_for_divisions(results,_word_counts, 
                                                                    _current_string_len_div_mapping)

            return final_embeddings[0, :]
        else:
            return self._get_for_string(sentence)

    def get_for_list(self, lst, get_for_large_string=False):
        """
        """
        if get_for_large_string:
            list_of_sentences,_word_counts, _current_string_len_div_mapping = self._divide_sentences(lst)
            
            results = self._get_for_list_raw(list_of_sentences)
            
            final_embeddings = self._merge_embeddings_for_divisions(results, _word_counts,
                                                                     _current_string_len_div_mapping)

            return final_embeddings
        else:
            return self._get_for_list_raw(lst)
