# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
bench_all_ivf.py 에 로그 생성 기능을 추가한 벤치마킹 파일
"""

import argparse
import os
import sys
import time
import json

import faiss
import numpy as np

try:
    import datasets_fb as datasets
except ModuleNotFoundError:
    import datasets_oss as datasets

sanitize = datasets.sanitize


######################################################################################
#                       실험을 위한 함수들 정의 (로깅, 설정변경)                             #
######################################################################################
import logging
import threading
import subprocess


def setup_logging():
    log_filename = get_unique_log_filename('benchmark.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_unique_log_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    else:
        base, ext = os.path.splitext(base_filename)
        counter = 1
        while True:
            new_filename = f"{base}{counter}{ext}"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1


def record_log(stop_event):
    while not stop_event.is_set():
        with open("/proc/vmstat", "r") as vmstat_file:
            vmstat = vmstat_file.read()

        with open("/sys/devices/system/node/node0/meminfo", "r") as meminfo_node0_file:
            meminfo_node0 = meminfo_node0_file.read()

        with open("/sys/devices/system/node/node1/meminfo", "r") as meminfo_node1_file:
            meminfo_node1 = meminfo_node1_file.read()

        with open("/sys/devices/system/node/node2/meminfo", "r") as meminfo_node2_file:
            meminfo_node2 = meminfo_node2_file.read()

        with open("/sys/devices/system/node/node3/meminfo", "r") as meminfo_node3_file:
            meminfo_node3 = meminfo_node3_file.read()

        with open("/sys/devices/system/node/node4/meminfo", "r") as meminfo_node4_file:
            meminfo_node4 = meminfo_node4_file.read()

        with open("/sys/devices/system/node/node5/meminfo", "r") as meminfo_node5_file:
            meminfo_node5 = meminfo_node5_file.read()


        logging.info(f"VMSTAT:\n{vmstat}")
        logging.info(f"NODE0 MEMINFO:\n{meminfo_node0}")
        logging.info(f"NODE1 MEMINFO:\n{meminfo_node1}")
        logging.info(f"NODE2 MEMINFO:\n{meminfo_node2}")
        logging.info(f"NODE3 MEMINFO:\n{meminfo_node3}")
        logging.info(f"NODE4 MEMINFO:\n{meminfo_node4}")
        logging.info(f"NODE5 MEMINFO:\n{meminfo_node5}")

        time.sleep(10)


def enable_cores_on_node(node=1):
    """
    특정 NUMA 노드에 속한 모든 코어를 활성화.

    Parameters:
    - node (int): 활성화할 NUMA 노드 번호 (0 또는 1)
    """
    time.sleep(5)
    if node == 0:
        node_cpus = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71
        ]
    elif node == 1:
        node_cpus = [
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
            88, 89, 90, 91, 92, 93, 94, 95
        ]
    else:
        print("Invalid NUMA node. Please specify 0 or 1.")
        return

    for cpu in node_cpus:
        try:
            # echo 명령어를 사용하여 코어 활성화
            subprocess.run(f"echo 1 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
            print(f"CPU {cpu} has been enabled.")
        except subprocess.CalledProcessError:
            print(f"Failed to enable CPU {cpu}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    time.sleep(30)


def disable_cores_on_node(node=1):
    """
    특정 NUMA 노드에 속한 모든 코어를 비활성화.
    Node 0의 CPU 0은 비활성화 불가.

    Parameters:
    - node (int): 비활성화할 NUMA 노드 번호 (0 또는 1)
    """
    time.sleep(5)
    if node == 0:
        node_cpus = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71
        ]
    elif node == 1:
        node_cpus = [
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
            88, 89, 90, 91, 92, 93, 94, 95
        ]
    else:
        print("Invalid NUMA node. Please specify 0 or 1.")
        return

    for cpu in node_cpus:
        # CPU 0은 비활성화하지 않음
        if node == 0 and cpu == 0:
            print("Skipping CPU 0 (cannot be disabled).")
            continue

        try:
            # echo 명령어를 사용하여 코어 비활성화
            subprocess.run(f"echo 0 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
            print(f"CPU {cpu} has been disabled.")
        except subprocess.CalledProcessError:
            print(f"Failed to disable CPU {cpu}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    time.sleep(30)


def set_numa_balancing(config_num):
    try:
        # Use the value argument in the echo command
        command = f"echo {config_num} | sudo tee /proc/sys/kernel/numa_balancing"
        subprocess.run(command, shell=True, check=True)
        print("Successfully set NUMA balancing.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set NUMA balancing: {e}")
    time.sleep(10)


def set_demotion_enabled(config_num):
    """
    Sets the value of /sys/kernel/mm/numa/demotion_enabled to the provided config_num.

    Args:
        config_num (int): The value to set for demotion_enabled (0 or 1).
    """
    try:
        # Use the value argument in the echo command
        command = f"echo {config_num} | sudo tee /sys/kernel/mm/numa/demotion_enabled"
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully set demotion_enabled to {config_num}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set demotion_enabled: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    time.sleep(10)


######################################################################################
#                                                                                    #
######################################################################################


def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        if vt is None:
            vt = lambda x: x
        else:
            vt = faiss.downcast_VectorTransform(vt)
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None


def apply_AQ_options(index, args):
    # if not(
    #    isinstance(index, faiss.IndexAdditiveQuantize) or
    #    isinstance(index, faiss.IndexIVFAdditiveQuantizer)):
    #    return
    if args.RQ_train_default:
        print("set default training for RQ")
        index.rq.train_type
        index.rq.train_type = faiss.ResidualQuantizer.Train_default
    if args.RQ_beam_size != -1:
        print("set RQ beam size to", args.RQ_beam_size)
        index.rq.max_beam_size
        index.rq.max_beam_size = args.RQ_beam_size
    if args.LSQ_encode_ils_iters != -1:
        print("set LSQ ils iterations to", args.LSQ_encode_ils_iters)
        index.lsq.encode_ils_iters
        index.lsq.encode_ils_iters = args.LSQ_encode_ils_iters
    if args.RQ_use_beam_LUT != -1:
        print("set RQ beam LUT to", args.RQ_use_beam_LUT)
        index.rq.use_beam_LUT
        index.rq.use_beam_LUT = args.RQ_use_beam_LUT



def eval_setting(index, xq, gt, k, inter, min_time):
    """ evaluate searching in terms of precision vs. speed """
    nq = xq.shape[0]
    ivf_stats = faiss.cvar.indexIVF_stats
    ivf_stats.reset()
    nrun = 0
    t0 = time.time()
    while True:
        D, I = index.search(xq, k)
        nrun += 1
        t1 = time.time()
        if t1 - t0 > min_time:
            break
    ms_per_query = ((t1 - t0) * 1000.0 / nq / nrun)
    res = {
        "ms_per_query": ms_per_query,
        "nrun": nrun
    }
    res["n"] = ms_per_query
    if inter:
        rank = k
        inter_measure = faiss.eval_intersection(gt[:, :rank], I[:, :rank]) / (nq * rank)
        print("%.4f" % inter_measure, end=' ')
        res["inter_measure"] = inter_measure
    else:
        res["recalls"] = {}
        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
            print("%.4f" % recall, end=' ')
            res["recalls"][rank] = recall
    print("   %9.5f  " % ms_per_query, end=' ')
    print("%12d   " % (ivf_stats.ndis / nrun), end=' ')
    print(nrun)
    res["ndis"] = ivf_stats.ndis / nrun
    return res

######################################################
# Training
######################################################

def run_train(args, ds, res):
    nq, d = ds.nq, ds.d
    nb, d = ds.nq, ds.d

    print("build index, key=", args.indexkey)

    index = faiss.index_factory(
        d, args.indexkey, faiss.METRIC_L2 if ds.metric == "L2" else
        faiss.METRIC_INNER_PRODUCT
    )

    index_ivf, vec_transform = unwind_index_ivf(index)

    if args.by_residual != -1:
        by_residual = args.by_residual == 1
        print("setting by_residual = ", by_residual)
        index_ivf.by_residual   # check if field exists
        index_ivf.by_residual = by_residual

    if index_ivf:
        print("Update add-time parameters")
        # adjust default parameters used at add time for quantizers
        # because otherwise the assignment is inaccurate
        quantizer = faiss.downcast_index(index_ivf.quantizer)
        if isinstance(quantizer, faiss.IndexRefine):
            print("   update quantizer k_factor=", quantizer.k_factor, end=" -> ")
            quantizer.k_factor = 32 if index_ivf.nlist < 1e6 else 64
            print(quantizer.k_factor)
            base_index = faiss.downcast_index(quantizer.base_index)
            if isinstance(base_index, faiss.IndexIVF):
                print("   update quantizer nprobe=", base_index.nprobe, end=" -> ")
                base_index.nprobe = (
                    16 if base_index.nlist < 1e5 else
                    32 if base_index.nlist < 4e6 else
                    64)
                print(base_index.nprobe)
        elif isinstance(quantizer, faiss.IndexHNSW):
            hnsw = quantizer.hnsw
            print(
                f"   update HNSW quantizer options, before: "
                f"{hnsw.efSearch=:} {hnsw.efConstruction=:}"
            )
            hnsw.efSearch = 40 if index_ivf.nlist < 4e6 else 64
            hnsw.efConstruction = 200
            print(f"       after: {hnsw.efSearch=:} {hnsw.efConstruction=:}")

    apply_AQ_options(index_ivf or index, args)

    if index_ivf:
        index_ivf.verbose = True
        index_ivf.quantizer.verbose = True
        index_ivf.cp.verbose = True
    else:
        index.verbose = True

    maxtrain = args.maxtrain
    if maxtrain == 0:
        if 'IMI' in args.indexkey:
            maxtrain = int(256 * 2 ** (np.log2(index_ivf.nlist) / 2))
        elif index_ivf:
            maxtrain = 50 * index_ivf.nlist
        else:
            # just guess...
            maxtrain = 256 * 100
        maxtrain = max(maxtrain, 256 * 100)
        print("setting maxtrain to %d" % maxtrain)

    try:
        xt2 = ds.get_train(maxtrain=maxtrain)
    except NotImplementedError:
        print("No training set: training on database")
        xt2 = ds.get_database()[:maxtrain]

    print("train, size", xt2.shape)
    assert np.all(np.isfinite(xt2))

    if (isinstance(vec_transform, faiss.OPQMatrix) and
        isinstance(index_ivf, faiss.IndexIVFPQFastScan)):
        print("  Forcing OPQ training PQ to PQ4")
        ref_pq = index_ivf.pq
        training_pq = faiss.ProductQuantizer(
            ref_pq.d, ref_pq.M, ref_pq.nbits
        )
        vec_transform.pq
        vec_transform.pq = training_pq


    if args.get_centroids_from == '':

        if args.clustering_niter >= 0:
            print(("setting nb of clustering iterations to %d" %
                   args.clustering_niter))
            index_ivf.cp.niter = args.clustering_niter

        if args.train_on_gpu:
            print("add a training index on GPU")
            train_index = faiss.index_cpu_to_all_gpus(
                    faiss.IndexFlatL2(index_ivf.d))
            index_ivf.clustering_index = train_index

    else:
        print("Getting centroids from", args.get_centroids_from)
        src_index = faiss.read_index(args.get_centroids_from)
        src_quant = faiss.downcast_index(src_index.quantizer)
        centroids = src_quant.reconstruct_n()
        print("  centroid table shape", centroids.shape)

        if isinstance(vec_transform, faiss.VectorTransform):
            print("  training vector transform")
            vec_transform.train(xt2)
            print("  transform centroids")
            centroids = vec_transform.apply_py(centroids)

        if not index_ivf.quantizer.is_trained:
            print("  training quantizer")
            index_ivf.quantizer.train(centroids)

        print("  add centroids to quantizer")
        index_ivf.quantizer.add(centroids)
        del src_index

    t0 = time.time()
    index.train(xt2)
    res.train_time = time.time() - t0
    print("  train in %.3f s" % res.train_time)
    return index

######################################################
# Populating index
######################################################

def run_add(args, ds, index, res):

    print("adding")
    t0 = time.time()
    if args.add_bs == -1:
        assert args.split == [1, 0], "split not supported with full batch add"
        index.add(sanitize(ds.get_database()))
    else:
        totn = ds.nb // args.split[0] # approximate
        i0 = 0
        print(f"Adding in block sizes {args.add_bs} with split {args.split}")
        for xblock in ds.database_iterator(bs=args.add_bs, split=args.split):
            i1 = i0 + len(xblock)
            print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                i0, i1, totn, time.time() - t0,
                faiss.get_mem_usage_kb()))
            index.add(xblock)
            i0 = i1

    res.t_add = time.time() - t0
    print(f"  add in {res.t_add:.3f} s index size {index.ntotal}")


######################################################
# Search
######################################################

def run_search(args, ds, index, res):

    index_ivf, vec_transform = unwind_index_ivf(index)

    if args.no_precomputed_tables:
        if isinstance(index_ivf, faiss.IndexIVFPQ):
            print("disabling precomputed table")
            index_ivf.use_precomputed_table = -1
            index_ivf.precomputed_table.clear()

    if args.indexfile:
        print("index size on disk: ", os.stat(args.indexfile).st_size)

    if hasattr(index, "code_size"):
        print("vector code_size", index.code_size)

    if hasattr(index_ivf, "code_size"):
        print("vector code_size (IVF)", index_ivf.code_size)

    print("current RSS:", faiss.get_mem_usage_kb() * 1024)

    precomputed_table_size = 0
    if hasattr(index_ivf, 'precomputed_table'):
        precomputed_table_size = index_ivf.precomputed_table.size() * 4

    print("precomputed tables size:", precomputed_table_size)

    # Index is ready

    xq = sanitize(ds.get_queries())
    nq, d = xq.shape
    gt = ds.get_groundtruth(k=args.k)

    if not args.accept_short_gt: # Deep1B has only a single NN per query
        assert gt.shape[1] == args.k

    if args.searchthreads != -1:
        print("Setting nb of threads to", args.searchthreads)
        faiss.omp_set_num_threads(args.searchthreads)
    else:
        print("nb search threads: ", faiss.omp_get_max_threads())

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    parametersets = args.searchparams

    if args.inter:
        header = (
            '%-40s     inter@%3d time(ms/q)   nb distances #runs' %
            ("parameters", args.k)
        )
    else:

        header = (
            '%-40s     R@1   R@10  R@100  time(ms/q)   nb distances #runs' %
            "parameters"
        )


    res.search_results = {}
    if parametersets == ['autotune']:

        ps.n_experiments = args.n_autotune
        ps.min_test_duration = args.min_test_duration

        for kv in args.autotune_max:
            k, vmax = kv.split(':')
            vmax = float(vmax)
            print("limiting %s to %g" % (k, vmax))
            pr = ps.add_range(k)
            values = faiss.vector_to_array(pr.values)
            values = np.array([v for v in values if v < vmax])
            faiss.copy_array_to_vector(values, pr.values)

        for kv in args.autotune_range:
            k, vals = kv.split(':')
            vals = np.fromstring(vals, sep=',')
            print("setting %s to %s" % (k, vals))
            pr = ps.add_range(k)
            faiss.copy_array_to_vector(vals, pr.values)

        # setup the Criterion object
        if args.inter:
            print("Optimize for intersection @ ", args.k)
            crit = faiss.IntersectionCriterion(nq, args.k)
        else:
            print("Optimize for 1-recall @ 1")
            crit = faiss.OneRecallAtRCriterion(nq, 1)

        # by default, the criterion will request only 1 NN
        crit.nnn = args.k
        crit.set_groundtruth(None, gt.astype('int64'))

        # then we let Faiss find the optimal parameters by itself
        print("exploring operating points, %d threads" % faiss.omp_get_max_threads());
        ps.display()

        t0 = time.time()
        op = ps.explore(index, xq, crit)
        res.t_explore = time.time() - t0
        print("Done in %.3f s, available OPs:" % res.t_explore)

        op.display()

        print("Re-running evaluation on selected OPs")
        print(header)
        opv = op.optimal_pts
        maxw = max(max(len(opv.at(i).key) for i in range(opv.size())), 40)
        for i in range(opv.size()):
            opt = opv.at(i)

            ps.set_index_parameters(index, opt.key)

            print(opt.key.ljust(maxw), end=' ')
            sys.stdout.flush()

            res_i = eval_setting(index, xq, gt, args.k, args.inter, args.min_test_duration)
            res.search_results[opt.key] = res_i

    else:
        print(header)
        for param in parametersets:
            print("%-40s " % param, end=' ')
            sys.stdout.flush()
            ps.set_index_parameters(index, param)

            res_i = eval_setting(index, xq, gt, args.k, args.inter, args.min_test_duration)
            res.search_results[param] = res_i



######################################################
# Driver function
######################################################

def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('general options')
    aa('--nthreads', default=-1, type=int,
        help='nb of threads to use at train and add time')
    aa('--json', default=False, action="store_true",
        help="output stats in JSON format at the end")
    aa('--todo', default=["check_files"],
       choices=["train", "add", "search", "check_files"],
       nargs="+", help='what to do (check_files means decide depending on which index files exist)')

    group = parser.add_argument_group('dataset options')
    aa('--db', default='deep1M', help='dataset')
    aa('--compute_gt', default=False, action='store_true',
        help='compute and store the groundtruth')
    aa('--force_IP', default=False, action="store_true",
        help='force IP search instead of L2')
    aa('--accept_short_gt', default=False, action='store_true',
        help='work around a problem with Deep1B GT')

    group = parser.add_argument_group('index construction')
    aa('--indexkey', default='HNSW32', help='index_factory type')
    aa('--trained_indexfile', default='',
       help='file to read or write a trained index from')
    aa('--maxtrain', default=256 * 256, type=int,
        help='maximum number of training points (0 to set automatically)')
    aa('--indexfile', default='', help='file to read or write index from')
    aa('--split', default=[1, 0], type=int, nargs=2, help="database split")
    aa('--add_bs', default=-1, type=int,
        help='add elements index by batches of this size')

    group = parser.add_argument_group('IVF options')
    aa('--by_residual', default=-1, type=int,
        help="set if index should use residuals (default=unchanged)")
    aa('--no_precomputed_tables', action='store_true', default=False,
        help='disable precomputed tables (uses less memory)')
    aa('--get_centroids_from', default='',
        help='get the centroids from this index (to speed up training)')
    aa('--clustering_niter', default=-1, type=int,
        help='number of clustering iterations (-1 = leave default)')
    aa('--train_on_gpu', default=False, action='store_true',
        help='do training on GPU')

    group = parser.add_argument_group('index-specific options')
    aa('--M0', default=-1, type=int, help='size of base level for HNSW')
    aa('--RQ_train_default', default=False, action="store_true",
        help='disable progressive dim training for RQ')
    aa('--RQ_beam_size', default=-1, type=int,
        help='set beam size at add time')
    aa('--LSQ_encode_ils_iters', default=-1, type=int,
        help='ILS iterations for LSQ')
    aa('--RQ_use_beam_LUT', default=-1, type=int,
        help='use beam LUT at add time')

    group = parser.add_argument_group('searching')
    aa('--k', default=100, type=int, help='nb of nearest neighbors')
    aa('--inter', default=False, action='store_true',
        help='use intersection measure instead of 1-recall as metric')
    aa('--searchthreads', default=-1, type=int,
        help='nb of threads to use at search time')
    aa('--searchparams', nargs='+', default=['autotune'],
        help="search parameters to use (can be autotune or a list of params)")
    aa('--n_autotune', default=500, type=int,
        help="max nb of autotune experiments")
    aa('--autotune_max', default=[], nargs='*',
        help='set max value for autotune variables format "var:val" (exclusive)')
    aa('--autotune_range', default=[], nargs='*',
        help='set complete autotune range, format "var:val1,val2,..."')
    aa('--min_test_duration', default=3.0, type=float,
        help='run test at least for so long to avoid jitter')
    aa('--indexes_to_merge', default=[], nargs="*",
        help="load these indexes to search and merge them before searching")

    args = parser.parse_args()

    if args.todo == ["check_files"]:
        if os.path.exists(args.indexfile):
            args.todo = ["search"]
        elif os.path.exists(args.trained_indexfile):
            args.todo = ["add", "search"]
        else:
            args.todo = ["train", "add", "search"]
        print("setting todo to", args.todo)

    print("args:", args)

    os.system('echo -n "nb processors "; '
            'cat /proc/cpuinfo | grep ^processor | wc -l; '
            'cat /proc/cpuinfo | grep ^"model name" | tail -1')

    # object to collect results
    res = argparse.Namespace()
    res.args = args.__dict__

    res.cpu_model = [
        l for l in open("/proc/cpuinfo", "r")
        if "model name" in l][0]

    print("Load dataset")

    ds = datasets.load_dataset(
        dataset=args.db, compute_gt=args.compute_gt)

    if args.force_IP:
        ds.metric = "IP"

    print(ds)

    if args.nthreads != -1:
        print("Set nb of threads to", args.nthreads)
        faiss.omp_set_num_threads(args.nthreads)
    else:
        print("nb threads: ", faiss.omp_get_max_threads())

    index = None
    if "train" in args.todo:
        print("================== Training index")
        index = run_train(args, ds, res)
        if args.trained_indexfile:
            print("storing trained index", args.trained_indexfile)
            faiss.write_index(index, args.trained_indexfile)

    if "add" in args.todo:
        if not index:
            assert args.trained_indexfile
            print("reading trained index", args.trained_indexfile)
            index = faiss.read_index(args.trained_indexfile)

        print("================== Adding vectors to index")
        run_add(args, ds, index, res)
        if args.indexfile:
            print("storing", args.indexfile)
            faiss.write_index(index, args.indexfile)

    if "search" in args.todo:
        # logging 세팅
        setup_logging()

        # vmstat 로그 기록 스레드 시작
        stop_event = threading.Event()
        logger_thread = threading.Thread(target=record_log, args=(stop_event,))
        logger_thread.start()

        # 하나의 소켓만 사용하기 위해 Node 1의 core 비활성화
        # logging.info(f"Searching : disable cores on node 1")
        # disable_cores_on_node(1)
        # logging.info(f"Searching : disabled cores on node 1")

        # numa_balancing 옵션 설정
        config_num = 2
        logging.info(f"Setting NUMA balancing to {config_num}.")
        set_numa_balancing(config_num)
        logging.info(f"Successfully set NUMA balancing to {config_num}.")

        # demotion_enabled 옵션 설정
        demotion_config = 1
        logging.info(f"Setting demotion_enabled to {demotion_config}.")
        set_demotion_enabled(demotion_config)
        logging.info(f"Successfully set demotion_enabled to {demotion_config}.")


        if not index:
            if args.indexfile:
                print("reading index", args.indexfile)
                index = faiss.read_index(args.indexfile)
            elif args.indexes_to_merge:
                print(f"Merging {len(args.indexes_to_merge)} indexes")
                sz = 0
                for fname in args.indexes_to_merge:
                    print(f"    reading {fname} (current size {sz})")
                    index_i = faiss.read_index(fname)
                    if index is None:
                        index = index_i
                    else:
                        index.merge_from(index_i, index.ntotal)
                    sz = index.ntotal
            else:
                assert False, "provide --indexfile"

        print("================== Searching")

        logging.info(f"Searching : Run Search.")
        start_time = time.time()
        try:
            run_search(args, ds, index, res)
        except Exception as e:
            logging.error(f"Error during searching: {e}")
            print(f"Error during searching: {e}")
            stop_event.set()
            logger_thread.join()
            return
        logging.info(f"Searching : Search completed in {time.time() - start_time:.2f} seconds")
        print(f"Searching completed in {time.time() - start_time:.2f} seconds")

        # Node1 원상 복구 : 실험 종료에 의함.
        # enable_cores_on_node(1)

        # vmstat 로그 기록 스레드 종료
        stop_event.set()
        logger_thread.join()

    if args.json:
        print("JSON results:", json.dumps(res.__dict__))


if __name__ == "__main__":
    main()