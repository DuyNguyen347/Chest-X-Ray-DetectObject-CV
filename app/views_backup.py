from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from keras import layers

from sklearn import preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pyvi import ViTokenizer
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle # để ghi file nhị phân
import nltk
from keras.layers import Dense, Embedding, LSTM, Dropout,SimpleRNN,LSTM,Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.models import model_from_json

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# from full_convmf_vn import ConvMF, Recommender
import pickle
import numpy as np
import inspect

import copy
import os
import warnings
from datetime import datetime
from glob import glob
import math
import tensorflow
import time
import numbers
from tqdm.auto import trange
import tensorflow.compat.v1 as tf
import dill
MEASURE_DOT = "dot product aka. inner product"

def uniform(shape=None, low=0.0, high=1.0, random_state=None, dtype=np.float32):
    return get_rng(random_state).uniform(low, high, shape).astype(dtype)


def get_rng(seed):
    '''Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    '''
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))

def xavier_uniform(shape, random_state=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer
    also known as 'Glorot' initializer on Uniform distribution.
    """
    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    limit = np.sqrt(3.0) * std
    return uniform(shape, -limit, limit, random_state, dtype)


def clip(values, lower_bound, upper_bound):
    """Perform clipping to enforce values to lie
    in a specific range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values

class CornacException(Exception):
    pass


class ScoreException(CornacException):
    pass

def conv_layer(
    input,
    num_input_channels,
    filter_height,
    filter_width,
    num_filters,
    seed=None,
    use_pooling=True,
):
    shape = [filter_height, filter_width, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05, seed=seed))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
    layer = tf.nn.conv2d(
        input=input, filter=weights, strides=[1, 1, 1, 1], padding="VALID"
    )
    layer = layer + biases
    if use_pooling:
        layer = tf.nn.max_pool(
            value=layer,
            ksize=[1, input.shape[1] - filter_height + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_feature = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_feature])
    return layer_flat, num_feature


def fc_layer(input, num_input, num_output, seed=None):
    weights = tf.Variable(
        tf.truncated_normal([num_input, num_output], stddev=0.05, seed=seed)
    )
    biases = tf.Variable(tf.constant(0.05, shape=[num_output]))
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.tanh(layer)
    return layer
class CNN_module:
    def __init__(
        self,
        output_dimension,
        dropout_rate,
        emb_dim,
        max_len,
        filter_sizes,
        num_filters,
        hidden_dim,
        seed,
        init_W,
        learning_rate=0.001,
    ):
        self.drop_rate = dropout_rate
        self.max_len = max_len
        self.seed = seed
        self.learning_rate = learning_rate
        self.init_W = tf.constant(init_W)
        self.output_dimension = output_dimension
        self.emb_dim = emb_dim
        self.filter_lengths = filter_sizes
        self.nb_filters = num_filters
        self.vanila_dimension = hidden_dim

        self._build_graph()

    def _build_graph(self):
        # create Graph
        self.model_input = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        self.v = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension))
        self.sample_weight = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.embedding_weight = tf.Variable(initial_value=self.init_W)

        self.seq_emb = tf.nn.embedding_lookup(self.embedding_weight, self.model_input)
        self.reshape = tf.reshape(self.seq_emb, [-1, self.max_len, self.emb_dim, 1])
        self.convs = []

        # Convolutional layers
        for i in self.filter_lengths:
            convolutional_layer, weights = conv_layer(
                input=self.reshape,
                num_input_channels=1,
                filter_height=i,
                filter_width=self.emb_dim,
                num_filters=self.nb_filters,
                use_pooling=True,
            )

            flat_layer, _ = flatten_layer(convolutional_layer)
            self.convs.append(flat_layer)

        self.model_output = tf.concat(self.convs, axis=-1)
        # Fully-connected layers
        self.model_output = fc_layer(
            input=self.model_output,
            num_input=self.model_output.get_shape()[-1],
            num_output=self.vanila_dimension,
        )
        # Dropout layer
        self.model_output = tf.nn.dropout(self.model_output, self.drop_rate)
        # Output layer
        self.model_output = fc_layer(
            input=self.model_output,
            num_input=self.vanila_dimension,
            num_output=self.output_dimension,
        )
        # Weighted MEA loss function
        self.mean_square_loss = tf.losses.mean_squared_error(
            labels=self.v,
            predictions=self.model_output,
            reduction=tf.losses.Reduction.NONE,
        )
        self.weighted_loss = tf.reduce_sum(
            tf.reduce_sum(self.mean_square_loss, axis=1, keepdims=True)
            * self.sample_weight
        )
        # RMSPro optimizer
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self.weighted_loss)

class Recommender:
    """Generic class for a recommender model. All recommendation models should inherit from this class.

    Parameters
    ----------------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    Attributes
    ----------
    num_users: int
        Number of users in training data.

    num_items: int
        Number of items in training data.

    total_users: int
        Number of users in training, validation, and test data.
        In other words, this includes unknown/unseen users.

    total_items: int
        Number of items in training, validation, and test data.
        In other words, this includes unknown/unseen items.

    uid_map: int
        Global mapping of user ID-index.

    iid_map: int
        Global mapping of item ID-index.

    max_rating: float
        Maximum value among the rating observations.

    min_rating: float
        Minimum value among the rating observations.

    global_mean: float
        Average value over the rating observations.
    """

    def __init__(self, name, trainable=True, verbose=False):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.is_fitted = False

        # attributes to be ignored when saving model
        self.ignored_attrs = ["train_set", "val_set", "test_set"]

        # useful information getting from train_set for prediction
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        self.max_rating = None
        self.min_rating = None
        self.global_mean = None

        self.__user_ids = None
        self.__item_ids = None

    @property
    def total_users(self):
        """Total number of users including users in test and validation if exists"""
        return len(self.uid_map) if self.uid_map is not None else self.num_users

    @property
    def total_items(self):
        """Total number of items including users in test and validation if exists"""
        return len(self.iid_map) if self.iid_map is not None else self.num_items

    @property
    def user_ids(self):
        """Return the list of raw user IDs"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return the list of raw item IDs"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

    def reset_info(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        ignored_attrs = set(self.ignored_attrs)
        for k, v in self.__dict__.items():
            if k in ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    @classmethod
    def _get_init_params(cls):
        """Get initial parameters from the model constructor"""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]

        return sorted([p.name for p in parameters])

    def clone(self, new_params=None):
        """Clone an instance of the model object.

        Parameters
        ----------
        new_params: dict, optional, default: None
            New parameters for the cloned instance.

        Returns
        -------
        object: :obj:`cornac.models.Recommender`
        """
        new_params = {} if new_params is None else new_params
        init_params = {}
        for name in self._get_init_params():
            init_params[name] = new_params.get(name, copy.deepcopy(getattr(self, name)))

        return self.__class__(**init_params)

    def save(self, save_dir=None, save_trainset=False):
        """Save a recommender model to the filesystem.
        """
        if save_dir is None:
            return

        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
#         model_file = os.path.join(model_dir, "{}.pkl".format(timestamp))
        model_file = "/kaggle/working/weights_" + "{}.pkl".format(timestamp)
        print('model_file dir: ', model_file)

        saved_model = copy.deepcopy(self)
        pickle.dump(saved_model, open(model_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print("{} model is saved to {}".format(self.name, model_file))

        if save_trainset:
            pickle.dump(
                self.train_set,
                open(model_file + ".trainset", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.
        """
        # try:
        # import pickle as pp
        if os.path.isdir(model_path):
            model_file = sorted(glob("{}/*.pkl".format(model_path)))[-1]
        else:
            model_file = model_path

        print('model file: ', model_file)
        model = pickle.load(open(model_file, "rb"))
        model.trainable = trainable
        model.load_from = model_file  # for further loading
        return model
        # except Exception as e:
        #     print('eror load: ',e)
        #     return None

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.
        """
        if self.is_fitted:
            warnings.warn(
                "Model is already fitted. Re-fitting will overwrite the previous model."
            )

        self.reset_info()
        train_set.reset()
        if val_set is not None:
            val_set.reset()

        # get some useful information for prediction
        self.num_users = train_set.num_users
        self.num_items = train_set.num_items
        self.uid_map = train_set.uid_map
        self.iid_map = train_set.iid_map
        self.min_rating = train_set.min_rating
        self.max_rating = train_set.max_rating
        self.global_mean = train_set.global_mean

        # just for future wrapper to call fit(), not supposed to be used during prediction
        self.train_set = train_set
        self.val_set = val_set

        self.is_fitted = True

        return self

    def knows_user(self, user_idx):
        """Return whether the model knows user by its index
        """
        return user_idx is not None and user_idx >= 0 and user_idx < self.num_users

    def knows_item(self, item_idx):
        """Return whether the model knows item by its index
        """
        return item_idx is not None and item_idx >= 0 and item_idx < self.num_items

    def is_unknown_user(self, user_idx):
        """Return whether the model knows user by its index. Reverse of knows_user() function,
        for better readability in some cases.
        """
        return not self.knows_user(user_idx)

    def is_unknown_item(self, item_idx):
        """Return whether the model knows item by its index. Reverse of knows_item() function,
        for better readability in some cases.
        """
        return not self.knows_item(item_idx)

    def transform(self, test_set):
        """Transform test set into cached results accelerating the score function.
        This function is supposed to be called in the `cornac.eval_methods.BaseMethod`
        before evaluation step. It is optional for this function to be implemented.
        """
        pass

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.
        """
        # print('score in recommender class')
        raise NotImplementedError("The algorithm is not able to make score prediction!")

    def default_score(self):
        """Overwrite this function if your algorithm has special treatment for cold-start problem"""
        return self.global_mean

    def rate(self, user_idx, item_idx, clipping=True):
        """Give a rating score between pair of user and item
        """
        try:
            rating_pred = self.score(user_idx, item_idx)
        except ScoreException:
            rating_pred = self.default_score()

        if clipping:
            rating_pred = clip(rating_pred, self.min_rating, self.max_rating)

        return rating_pred

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):
        """Rank all test items for a given user.
        """
        # obtain item scores from the model
        try:
            print('in try')
            known_item_scores = self.score(user_idx, **kwargs)
        except ScoreException:
            print('in catch')
            known_item_scores = np.ones(self.total_items) * self.default_score()

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.total_items) * np.min(known_item_scores)
            all_item_scores[: self.num_items] = known_item_scores

        # rank items based on their scores
        item_indices = (
            np.arange(self.num_items)
            if item_indices is None
            else np.asarray(item_indices)
        )
        item_scores = all_item_scores[item_indices]
        if (
            k != -1
        ):  # O(n + k log k), faster for small k which is usually the case
            partitioned_idx = np.argpartition(item_scores, -k)
            top_k_idx = partitioned_idx[-k:]
            sorted_top_k_idx = top_k_idx[np.argsort(item_scores[top_k_idx])]
            partitioned_idx[-k:] = sorted_top_k_idx
            ranked_items = item_indices[partitioned_idx[::-1]]
            score_items_sorted = item_scores[partitioned_idx[::-1]]
        else:  # O(n log n)
            ranked_items = item_indices[item_scores.argsort()[::-1]]
#             score_items_sorted = item_scores[item_scores.argsort()[::-1]]

        return ranked_items, score_items_sorted

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None):
        """Generate top-K item recommendations for a given user. Key difference between
        this function and rank() function is that rank() function works with mapped
        user/item index while this function works with original user/item ID. This helps
        hide the abstraction of ID-index mapping, and make model usage and deployment cleaner.
        """
        user_idx = self.uid_map.get(user_id, -1)
        if user_idx == -1:
            raise ValueError(f"{user_id} is unknown to the model.")

        if k < -1 or k > self.total_items:
            raise ValueError(f"k={k} is invalid, there are {self.total_users} users in total.")

        item_indices = np.arange(self.total_items)
        if remove_seen:
            seen_mask = np.zeros(len(item_indices), dtype="bool")
            if train_set is None:
                raise ValueError("train_set must be provided to remove seen items.")
            if user_idx < train_set.csr_matrix.shape[0]:
                seen_mask[train_set.csr_matrix.getrow(user_idx).indices] = True
                item_indices = item_indices[~seen_mask]

        item_rank, score_items_sorted = self.rank(user_idx, item_indices,k)
        if k != -1:
            item_rank = item_rank[:k]
            score_items_sorted = score_items_sorted[:k]

        recommendations = [self.item_ids[i] for i in item_rank]
        return recommendations, score_items_sorted

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.
        Note: `val_set` could be `None` thus it needs to be checked before usage.
        """
        raise NotImplementedError()

    def early_stop(self, train_set, val_set, min_delta=0.0, patience=0):
        """Check if training should be stopped when validation loss has stopped improving.
        """
        self.current_epoch += 1
        current_value = self.monitor_value(train_set, val_set)
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            print("Early stopping:")
            print("- best epoch = {}, stopped epoch = {}".format(self.best_epoch, self.stopped_epoch))
            print(
                "- best monitored value = {:.6f} (delta = {:.6f})".format(
                    self.best_value, current_value - self.best_value
                )
            )
            return True
        return False

class ConvMF(Recommender):
    """
    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the user and item latent factors.

    n_epochs: int, optional, default: 50
        Maximum number of epochs for training.

    cnn_epochs: int, optional, default: 5
        Number of epochs for optimizing the CNN for each overall training epoch.

    cnn_bs: int, optional, default: 128
        Batch size for optimizing CNN.

    cnn_lr: float, optional, default: 0.001
        Learning rate for optimizing CNN.

    lambda_u: float, optional, default: 1.0
        The regularization hyper-parameter for user latent factor.

    lambda_v: float, optional, default: 100.0
        The regularization hyper-parameter for item latent factor.

    emb_dim: int, optional, default: 200
        The embedding size of each word. One word corresponds with [1 x emb_dim] vector in the embedding space

    max_len: int, optional, default 300
        The maximum length of item's document

    filter_sizes: list, optional, default: [3, 4, 5]
        The length of filters in convolutional layer

    num_filters: int, optional, default: 100
        The number of filters in convolutional layer

    hidden_dim: int, optional, default: 200
        The dimension of hidden layer after the pooling of all convolutional layers

    dropout_rate: float, optional, default: 0.2
        Dropout rate while training CNN

    give_item_weight: boolean, optional, default: True
        When True, each item will be weighted base on the number of user who have rated this item

    init_params: dict, optional, default: {'U':None, 'V':None, 'W': None}
        Initial U and V matrix and initial weight for embedding layer W

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
    """

    def __init__(
        self,
        name="ConvMF",
        k=50,
        n_epochs=50,
        cnn_epochs=5,
        cnn_bs=128,
        cnn_lr=0.001,
        lambda_u=1,
        lambda_v=100,
        emb_dim=200,
        max_len=300,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        hidden_dim=200,
        dropout_rate=0.2,
        give_item_weight=True,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.give_item_weight = give_item_weight
        self.n_epochs = n_epochs
        self.cnn_bs = cnn_bs
        self.cnn_lr = cnn_lr
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.k = k
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.name = name
        self.verbose = verbose
        self.cnn_epochs = cnn_epochs
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)
        self.W = self.init_params.get("W", None)

    def _init(self, train_set):
        rng = get_rng(self.seed)
        n_users, n_items = train_set.num_users, train_set.num_items
        vocab_size = train_set.item_text.vocab.size

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), rng)
        if self.W is None:
            self.W = xavier_uniform((vocab_size, self.emb_dim), rng)

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.
        """
        Recommender.fit(self, train_set, val_set)

        self._init(train_set)

        if self.trainable:
            self._fit_convmf(train_set)

        return self

    @staticmethod
    def _build_data(csr_mat):
        data = []
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        data.append(index_list)
        data.append(rating_list)
        return data

    def _fit_convmf(self, train_set):
        user_data = self._build_data(train_set.matrix)
        item_data = self._build_data(train_set.matrix.T.tocsr())

        n_user = len(user_data[0])
        n_item = len(item_data[0])

        # R_user and R_item contain rating values
        R_user = user_data[1]
        R_item = item_data[1]

        if self.give_item_weight:
            item_weight = np.array([math.sqrt(len(i)) for i in R_item], dtype=float)
            item_weight = (float(n_item) / item_weight.sum()) * item_weight
        else:
            item_weight = np.ones(n_item, dtype=float)

        # Initialize cnn module
        # import tensorflow.compat.v1 as tf
        # from .convmf import CNN_module

        tf.disable_eager_execution()

        # less verbose TF
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)

        tf.set_random_seed(self.seed)
        cnn_module = CNN_module(
            output_dimension=self.k,
            dropout_rate=self.dropout_rate,
            emb_dim=self.emb_dim,
            max_len=self.max_len,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
            init_W=self.W,
            learning_rate=self.cnn_lr,
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())  # init variable

        document = train_set.item_text.batch_seq(
            np.arange(n_item), max_length=self.max_len
        )

        feed_dict = {cnn_module.model_input: document}
        theta = sess.run(cnn_module.model_output, feed_dict=feed_dict)

        endure = 3
        converge_threshold = 0.01
        history = 1e-50
        loss = 0

        for epoch in range(1, self.n_epochs + 1):
            if self.verbose:
                print("Epoch: {}/{}".format(epoch, self.n_epochs))

            tic = time.time()

            user_loss = 0.0
            for i in range(n_user):
                idx_item = user_data[0][i]
                V_i = self.V[idx_item]
                R_i = R_user[i]

                A = self.lambda_u * np.eye(self.k) + V_i.T.dot(V_i)
                B = (V_i * (np.tile(R_i, (self.k, 1)).T)).sum(0)
                self.U[i] = np.linalg.solve(A, B)

                user_loss += self.lambda_u * np.dot(self.U[i], self.U[i])

            item_loss = 0.0
            for j in range(n_item):
                idx_user = item_data[0][j]
                U_j = self.U[idx_user]
                R_j = R_item[j]

                A = self.lambda_v * item_weight[j] * np.eye(self.k) + U_j.T.dot(U_j)
                B = (U_j * (np.tile(R_j, (self.k, 1)).T)).sum(
                    0
                ) + self.lambda_v * item_weight[j] * theta[j]
                self.V[j] = np.linalg.solve(A, B)

                item_loss += np.square(R_j - U_j.dot(self.V[j])).sum()

            loop = trange(
                self.cnn_epochs, desc="Optimizing CNN", disable=not self.verbose
            )
            for _ in loop:
                for batch_ids in train_set.item_iter(
                    batch_size=self.cnn_bs, shuffle=True
                ):
                    batch_seq = train_set.item_text.batch_seq(
                        batch_ids, max_length=self.max_len
                    )
                    feed_dict = {
                        cnn_module.model_input: batch_seq,
                        cnn_module.v: self.V[batch_ids],
                        cnn_module.sample_weight: item_weight[batch_ids],
                    }

                    sess.run([cnn_module.optimizer], feed_dict=feed_dict)

            feed_dict = {
                cnn_module.model_input: document,
                cnn_module.v: self.V,
                cnn_module.sample_weight: item_weight,
            }
            theta, cnn_loss = sess.run(
                [cnn_module.model_output, cnn_module.weighted_loss], feed_dict=feed_dict
            )

            loss = 0.5 * (user_loss + item_loss + self.lambda_v * cnn_loss)

            toc = time.time()
            elapsed = toc - tic
            converge = abs((loss - history) / history)

            if self.verbose:
                print("Loss: %.5f Elapsed: %.4fs Converge: %.6f ")

            history = loss
            if converge < converge_threshold:
                endure -= 1
                if endure == 0:
                    break

        tf.reset_default_graph()

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.
        """
        # print('score in convmf class ')
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            return self.V.dot(self.U[user_idx, :])

        return self.V[item_idx, :].dot(self.U[user_idx, :])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.
        """
        return self.V

# try:
#     base_path = os.path.dirname(__file__)
#     full_path = os.path.join(base_path, 'weights_2024-05-31_02-59-14-954041.pkl')
#     model = ConvMF.load(full_path)
# except Exception as e:
#     print('eror: ',e)

# def create_model():
#     base_path = os.path.dirname(__file__)
#     full_path = os.path.join(base_path, 'weights_2024-05-31_02-59-14-954041.pkl')
#     model = ConvMF.load('./weights_2024-05-31_02-59-14-954041.pkl')
#     return model

# model_conv = create_model()
    
# base_path = os.path.dirname(__file__)
# full_path = os.path.join(base_path, 'weights_2024-05-31_02-59-14-954041.pkl')
model = ConvMF.load('./weights_2024-05-31_02-59-14-954041.pkl')
try:

    # Obtain item recommendations for user U1
    recs,score_item = model.recommend(user_id="9", k=50)
    print(recs)
    print(score_item)
except ValueError as e:
    # Handle the ValueError if the user_id is unknown
    print(f"Error: {e}")

def predictTopK_convmf(user_id, topK):
    try:
        # Obtain item recommendations for user U1
        recs,score_item = model.recommend(user_id=user_id, k=topK)
        print(recs)
        print(score_item)
        return recs
    except ValueError as e:
        # Handle the ValueError if the user_id is unknown
        print(f"Error: {e}")
        return None
    
# print(predictTopK_convmf(9,5))


# class LayerScale(layers.Layer):
#     """Layer scale module.

#     References:
#       - https://arxiv.org/abs/2103.17239

#     Args:
#       init_values (float): Initial value for layer scale. Should be within
#         [0, 1].
#       projection_dim (int): Projection dimensionality.

#     Returns:
#       Tensor multiplied to the scale.
#     """

#     def __init__(self, init_values, projection_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.init_values = init_values
#         self.projection_dim = projection_dim

#     def build(self, input_shape):
#         self.gamma = tf.Variable(
#             self.init_values * tf.ones((self.projection_dim,))
#         )

#     def call(self, x):
#         return x * self.gamma

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "init_values": self.init_values,
#                 "projection_dim": self.projection_dim,
#             }
#         )
#         return config

# class StochasticDepth(layers.Layer):
#   """Stochastic Depth module.

#   It performs batch-wise dropping rather than sample-wise. In libraries like
#   `timm`, it's similar to `DropPath` layers that drops residual paths
#   sample-wise.

#   References:
#     - https://github.com/rwightman/pytorch-image-models

#   Args:
#     drop_path_rate (float): Probability of dropping paths. Should be within
#       [0, 1].

#   Returns:
#     Tensor either with the residual path dropped or kept.
#   """

#   def __init__(self, drop_path_rate, **kwargs):
#       super().__init__(**kwargs)
#       self.drop_path_rate = drop_path_rate

#   def call(self, x, training=None):
#       if training:
#           keep_prob = 1 - self.drop_path_rate
#           shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
#           random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
#           random_tensor = tf.floor(random_tensor)
#           return (x / keep_prob) * random_tensor
#       return x

#   def get_config(self):
#       config = super().get_config()
#       config.update({"drop_path_rate": self.drop_path_rate})
#       return config

# # model = load_model("./model/my_model.h5", compile=False, custom_objects={ "LayerScale": LayerScale, "StochasticDepth": StochasticDepth } )

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# img_size = (300, 300)
# channels = 3
# img_shape = (img_size[0], img_size[1], channels)
# # tf.random.set_seed(123)
# # np.random.seed(123)

# # Create the pre-trained ConvNeXtTiny base model
# base_model = tf.keras.applications.convnext.ConvNeXtBase(
#     include_top=False,
#     weights='imagenet',
#     input_shape=img_shape,
#     pooling='max',
# )

# # Build the model on top of the base model
# model = Sequential([
#     base_model,

#     Dense(4, activation='softmax')
# ])

# model.compile(Adam(learning_rate= 0.0001), loss= 'categorical_crossentropy', metrics= ['accuracy', recall_m, precision_m, f1_m])

# model.summary()
# model.load_weights('./ck_ep3_2.ckpt')

# Create your views here.
def home(request):
    return render(request, 'app/pages/home.html')

def services(request):
    context = {}
    context['service'] = ''
    context['result'] = ''
    # IMAGE_HEIGHT = 300
    # IMAGE_WIDTH = 300
    # result = ""
    if request.method == 'POST':
        # if request.FILES.get('img') != None:
        #     uploaded_file = request.FILES['img']
        #     context['service'] = 'img'
        # fileSystemStorage = FileSystemStorage()
        # fileSystemStorage.save(uploaded_file.name, uploaded_file)
        # context['url'] = fileSystemStorage.url(uploaded_file)
        # img = '.' + context['url']
        # image = cv2.imread(img)
        # image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # image = tf.image.rot90(image, k=np.random.randint(0, 4))
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        # image = tf.image.random_brightness(image, max_delta=0.2)
        # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # input_image = np.expand_dims(image, axis=0)  # Thêm chiều batch
        # # Thực hiện phân loại
        # predictions = model.predict(input_image)
        # print('predictions: ',predictions)

        # # Lấy chỉ số của lớp có xác suất cao nhất
        # predicted_class_index = np.argmax(predictions, axis=1)[0]
        # print('predicted_class_index: ', predicted_class_index)
        # name_class = ['MildDermented','ModerateDemented','NonDemented','VeryMildDemented']
        recs,score_item = model.recommend(user_id='9', k=5)
        print(recs)
        context['result'] = "Hình ảnh thuộc lớp "
    print('--------------context------------: ', context)
    

    return render(request, 'app/pages/services.html', context)

def about(request):
    return render(request, 'app/pages/about.html')

def contact(request):
    return render(request, 'app/pages/contact.html')

# chat bot view
# data = pd.read_excel('./Data4.xlsx')
# X=data['question'].tolist()
# y=data['answer']


# le = preprocessing.LabelEncoder()
# le.fit(y)
# list_label = list(le.classes_)
# label = le.transform(y)

# onehot_encoder = OneHotEncoder(sparse_output=False)
# label = label.reshape(len(label), 1)
# onehot_encoder = onehot_encoder.fit_transform(label)

# def tienxuly(document):
#     document = ViTokenizer.tokenize(document)
#     document = document.lower()
#     document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
#     document = re.sub(r'\s+', ' ', document).strip()
#     return document

# for i in range(0,len(X)):
#     X[i] = tienxuly(X[i])

# tokens = []

# for i in range(0,len(X)):
#     tokens.extend(X[i].split(" "))

# freq = nltk.FreqDist(tokens)
# # freq.plot(10, cumulative=False)

# # stopwords_file = open('./stop_words_Vietnamese.txt','r')
# # content = stopwords_file.read()

# # Mở tệp stopwords_file với mã hóa utf-8
# with open('./stop_words_Vietnamese.txt', 'r', encoding='utf-8') as stopwords_file:
#     content = stopwords_file.read()
# stopwords_VN = content.splitlines()
# stopwords_file.close()

# stopword = ['nghành', 'ngành', 'trường']

# stopword = stopwords_VN + stopword

# def remove_stopwords(line):
#     words = []
#     for word in line.strip().split():
#         if word not in stopword:
#             words.append(word)
#     return ' '.join(words)

# for i in range(0,len(X)):
#     X[i] = remove_stopwords(X[i])

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# text_to_sequence = tokenizer.texts_to_sequences(X)

# max_length_sequence = max([len(s) for s in text_to_sequence])

# padded_sequence = pad_sequences(text_to_sequence, maxlen=max_length_sequence, padding='pre')

# X_train,X_test,Y_train,Y_test = train_test_split(padded_sequence,onehot_encoder,test_size=0.2,random_state=42)

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3,restore_best_weights=True)
# TOT_SIZE = len(tokenizer.word_index) + 1

# def RNN_model():
#     chatbot_model = Sequential()
#     # model.add(Embedding(TOT_SIZE,output_dim=16, input_length=max_length_sequence))
#     chatbot_model.add(Embedding(313,output_dim=16, input_length=24))
#     chatbot_model.add(SimpleRNN(300))
#     chatbot_model.add(Dense(200, activation='relu'))
#     chatbot_model.add(Dense(99, activation='softmax'))
#     chatbot_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return chatbot_model

# chatbot_model = RNN_model()
# chatbot_model.summary()

# chatbot_model.load_weights('./weights_chatbot.ckpt')

# def pre(x):
#     X = [x]
#     for i in range(0,len(X)):
#         X[i] = tienxuly(X[i])
#     for i in range(0,len(X)):
#         X[i] = remove_stopwords(X[i])
#     text_to_sequence = tokenizer.texts_to_sequences(X)
#     padded_sequence = pad_sequences(text_to_sequence, maxlen=24, padding='pre')
#     pred = chatbot_model.predict(padded_sequence)
#     print(list_label[np.argmax(pred)])
#     return list_label[np.argmax(pred)]


# def chatbot(request):
#     if request.method == 'POST':
#         try:
#             # Trích xuất payload JSON từ request body
#             payload = json.loads(request.body)
#             user_message = payload.get('messages', [{}])[0].get('content', '')

#             # Xử lý tin nhắn của người dùng ở đây (ví dụ: gửi tin nhắn đến mô hình GPT-3.5-turbo)
#             # Giả sử chúng ta trả lại tin nhắn của người dùng như một phần của phản hồi
#             response_message = pre(user_message)

#             return JsonResponse({
#                 'status': 'success',
#                 'message': response_message,
#                 'data': payload  # Trả về payload để kiểm tra
#             })
#         except json.JSONDecodeError:
#             return JsonResponse({
#                 'status': 'fail',
#                 'message': 'Invalid JSON payload'
#             }, status=400)
#     else:
#         return JsonResponse({
#             'status': 'fail',
#             'message': 'Only POST method is allowed'
#         }, status=405)
