# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
import faiss

try:
    from faiss.contrib.datasets_fb import DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import DatasetBigANN

# from datasets import load_sift1M


k = int(sys.argv[1])
todo = sys.argv[2:]

print("load data")

# xb, xq, xt, gt = load_sift1M()

ds = DatasetBigANN()
ds.basedir = '/home/smrc/nicedavid98/faiss_dataset/bigann/'

xq = ds.get_queries()
gt = ds.get_groundtruth()
xt = ds.get_train()

nq, d = xq.shape

if todo == []:
    todo = 'search save'.split()


def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))


if 'save' in todo:
    print("HNSW Index 저장")
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 40

    print("add")
    # to see progress
    index.verbose = True
    for batch in ds.database_iterator(bs=1000000):
        index.add(batch)

    faiss.write_index(index, "hnsw_index")


if 'search' in todo:

    # print("Testing HNSW Flat")
    #
    # index = faiss.IndexHNSWFlat(d, 32)
    #
    # # training is not needed
    #
    # # this is the default, higher is more accurate and slower to
    # # construct
    # index.hnsw.efConstruction = 40
    #
    # print("add")
    # # to see progress
    # index.verbose = True
    #
    # for batch in ds.database_iterator(bs=1000000):
    #     index.add(batch)

    print("HNSW Index 불러오기")
    index = faiss.read_index("hnsw_index")

    print("search")
    for efSearch in 16, 32, 64, 128, 256:
        for bounded_queue in [True, False]:
            print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
            index.hnsw.search_bounded_queue = bounded_queue
            index.hnsw.efSearch = efSearch
            evaluate(index)

