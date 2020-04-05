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
from typing import Tuple

import numpy as np
from nearpy.storage import MemoryStorage

from indexes.HashEngine.NearestFilter import NearestFilter


class HashEngine:
    """
    Optimized engine, stores indexes instead of full vectors in the storage.
    Must receive full matrix of vectors at construction.
    """

    def __init__(
            self, vectors, labels,
            lshashes=None,
            k: int = 100,
            verbose=False
    ):
        self.lshashes = lshashes
        for lshash in self.lshashes:
            lshash.reset(vectors.shape[1])

        self.nearest_filter = NearestFilter(k=k)
        self.storage = OptimizedMemoryStorage()

        self.vectors = vectors.astype('float32')
        self.labels = labels

        t0 = time.time()
        for i in range(self.vectors.shape[0]):
            self.store_vector(i)

            if verbose and (i + 1) % (self.vectors.shape[0] // 10) == 0:
                print(f'indexed {i + 1:,} ({round((i + 1) / self.vectors.shape[0] * 100)}%) vectors'
                      f' in {time.time() - t0:.1f} seconds')

    def store_vector(self, v_i):
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

    def neighbours(self, v):
        """
        Hashes vector v, collects all candidate vectors from the matching
        buckets in storage, applys the (optional) distance function and
        finally the (optional) filter function to construct the returned list
        of either (vector, data, distance) tuples or (vector, data) tuples.
        """

        # Collect candidates from all buckets from all hashes
        points, labels = self._get_candidates(v)

        # brute force search over candidates
        points, labels = self.nearest_filter.filter(v, points, labels)

        return points, labels

    def candidate_count(self, vector):
        """ Counts candidates from all buckets from all hashes """
        candidates = set()

        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(vector):
                bucket_content = self.storage.get_bucket(lshash.hash_name, bucket_key)
                candidates.update(bucket_content)

        return len(candidates)

    def _get_candidates(self, vector) -> Tuple[np.ndarray, np.ndarray]:
        """ Collect candidates from all buckets from all hashes """
        cand_indexes = set()
        centered_vector = vector

        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(centered_vector):
                bucket_indexes = self.storage.get_bucket(lshash.hash_name, bucket_key)
                cand_indexes.update(bucket_indexes)

        # retrieve real vectors and labels from indexes
        indexes = list(cand_indexes)
        return self.vectors[indexes], self.labels[indexes]

    def analize_storage(self):
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

    def analize_storage(self):
        """
        Only for testing purposes.
        Prints all storage keys and number of vectors in bucket.
        """

        bucket_sizes = []
        for table in self.buckets:
            for key in self.buckets[table]:
                bucket_sizes.append(len(self.buckets[table][key]))

        print(f'storage stats:\n'
              f'\tbuckets / table: {len(bucket_sizes) / len(self.buckets)}\n'
              f'\ttop_10: {sorted(bucket_sizes, reverse=True)[:10]}\n'
              f'\tmean: {np.mean(bucket_sizes):.1f}\n'
              f'\tmedian: {np.median(bucket_sizes):.1f}\n'
              f'\tmin: {min(bucket_sizes)}\n')

        # import matplotlib.pyplot as plt
        # plt.hist(bucket_sizes, bins=60, range=(0, 300))
        # plt.show()

        return
