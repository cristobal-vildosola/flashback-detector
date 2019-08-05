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

from nearpy.engine import Engine


class OptimizedEngine(Engine):
    """
    Optimized engine, stores indexes instead of full vectors in the storage.
    Has to receive full matrix of vectors at construction.
    """

    def __init__(self, vectors, lshashes=None, distance=None, fetch_vector_filters=None, vector_filters=None,
                 storage=None, verbose=False):
        super().__init__(vectors.shape[1], lshashes, distance, fetch_vector_filters, vector_filters, storage)
        self.vectors = vectors

        t0 = time.process_time()
        for i in range(self.vectors.shape[0]):
            self.store_vector(i, i)

            if verbose and (i + 1) % (self.vectors.shape[0] // 10) == 0:
                print(f'indexed {i + 1} ({round((i + 1) / self.vectors.shape[0] * 100)}%) vectors'
                      f' in {time.process_time() - t0:.1f} seconds')

    def store_vector(self, i, data=None):
        """
        Hashes vector i and stores it in all matching buckets in the storage.
        The data argument must be JSON-serializable. It is stored with the
        vector and will be returned in search results.
        """
        # Store vector in each bucket of all hashes
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(self.vectors[i]):
                # store the vector index instead of the vector itself in the storage
                self.storage.store_vector(lshash.hash_name, bucket_key, i, data)

        return

    def delete_vector(self, data, i=None):
        """
        Deletes vector i and his id (data) in all matching buckets in the storage.
        The data argument must be JSON-serializable.
        """

        # Delete data id in each hashes
        for lshash in self.lshashes:
            if i is None:
                keys = self.storage.get_all_bucket_keys(lshash.hash_name)
            else:
                keys = lshash.hash_vector(self.vectors[i])
            self.storage.delete_vector(lshash.hash_name, keys, data)

        return

    def _get_candidates(self, v):
        """ Collect candidates from all buckets from all hashes """
        candidates = []
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v, querying=True):
                bucket_content = self.storage.get_bucket(lshash.hash_name, bucket_key)

                # retrieve real vectors from indexes
                bucket_candidates = [(self.vectors[i], data) for i, data in bucket_content]
                candidates.extend(bucket_candidates)

        return candidates

    def _append_distances(self, v, distance, candidates):
        if distance:
            candidates = [(x[0], x[1], self.distance.distance(x[0], v)) for x in candidates]

        return candidates
