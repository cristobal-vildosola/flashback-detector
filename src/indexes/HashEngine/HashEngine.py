# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import time
from typing import Tuple, List

import numpy as np
from nearpy.engine import Engine
from nearpy.storage import MemoryStorage
from nearpy.distances.euclidean import EuclideanDistance


class HashEngine(Engine):
    """
    Optimized engine, stores indexes instead of full vectors in the storage.
    Must receive full matrix of vectors at construction.
    """

    def __init__(
            self, vectors, labels,
            lshashes=None,
            vector_filters=None,
            distance=EuclideanDistance(),
            fetch_vector_filters=None,
            verbose=False
    ):
        super().__init__(
            dim=vectors.shape[1],
            lshashes=lshashes,
            distance=distance,
            fetch_vector_filters=fetch_vector_filters,
            vector_filters=vector_filters,
            storage=OptimizedMemoryStorage())

        self.vectors = vectors
        self.labels = labels

        t0 = time.time()
        for i in range(self.vectors.shape[0]):
            self.store_vector(i)

            if verbose and (i + 1) % (self.vectors.shape[0] // 10) == 0:
                print(f'indexed {i + 1:,} ({round((i + 1) / self.vectors.shape[0] * 100)}%) vectors'
                      f' in {time.time() - t0:.1f} seconds')

    def store_vector(self, v_i, data=None):
        """
        Hashes vector i and stores it in all matching buckets in the storage.
        The data argument must be JSON-serializable. It is stored with the
        vector and will be returned in search results.
        """
        # Store vector in each bucket of all hashes
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(self.vectors[v_i]):
                # store the vector index instead of the vector itself in the storage
                self.storage.store_vector(lshash.hash_name, bucket_key, v_i)

        return

    def delete_vector(self, v_i, v=None):
        """ Deletes vector v_i in all matching buckets in the storage. """
        centered_vector = self.vectors[v_i]

        # Delete vector index in each hashes
        for lshash in self.lshashes:
            keys = lshash.hash_vector(centered_vector)
            self.storage.delete_vector(lshash.hash_name, keys, v_i)

        return

    def candidate_count(self, vector):
        """ Counts candidates from all buckets from all hashes """
        candidates = 0
        centered_vector = vector

        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(centered_vector):
                candidates += len(self.storage.get_bucket(lshash.hash_name, bucket_key))

        return candidates

    def _get_candidates(self, vector) -> List[Tuple[str, np.ndarray]]:
        """ Collect candidates from all buckets from all hashes """
        candidates_indexes = []
        centered_vector = vector

        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(centered_vector):
                bucket_indexes = self.storage.get_bucket(lshash.hash_name, bucket_key)
                candidates_indexes.extend(bucket_indexes)

        # retrieve real vectors from indexes
        candidates = [(self.vectors[v_i], self.labels[v_i]) for v_i in candidates_indexes]
        return candidates

    def _append_distances(self, v, distance, candidates) -> List[Tuple[str, np.ndarray, float]]:
        if distance:
            candidates = [(*x, self.distance.distance(x[0], v)) for x in candidates]

        return candidates

    def analize_storage(self):
        """
        Only for testing purposes.
        Prints all storage keys and number of vectors in bucket.
        """
        self.storage.analize_storage()
        return


class OptimizedMemoryStorage(MemoryStorage):
    """ Simple implementation using python dicts. """

    def __init__(self):
        super().__init__()

    def store_vector(self, hash_name, bucket_key, v_i, data=None):
        """ Stores vector in bucket with specified key. """

        if hash_name not in self.buckets:
            self.buckets[hash_name] = {}

        if bucket_key not in self.buckets[hash_name]:
            self.buckets[hash_name][bucket_key] = []

        self.buckets[hash_name][bucket_key].append(v_i)
        return

    def delete_vector(self, hash_name, bucket_keys, v_i):
        """ Deletes vector in buckets with specified keys. """
        for key in bucket_keys:
            bucket = self.get_bucket(hash_name, key)
            bucket[:] = [i for i in bucket if i != v_i]

        return

    def analize_storage(self):
        """
        Only for testing purposes.
        Prints all storage keys and number of vectors in bucket.
        """

        bucket_sizes = []
        for table in self.buckets:
            for key in self.buckets[table]:
                bucket_sizes.append(len(self.buckets[table][key]))

        mean = np.mean(bucket_sizes)
        print(f'storage stats:\n'
              f'\ttotal buckets: {len(bucket_sizes)}\n'
              f'\tmax: {max(bucket_sizes)}\n'
              f'\tmean: {mean:.1f}\n'
              f'\tmin: {min(bucket_sizes)}\n')

        return
