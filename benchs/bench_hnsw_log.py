# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
#
# import time
# import sys
# import numpy as np
# import faiss
#
# try:
#     from faiss.contrib.datasets_fb import DatasetBigANN
# except ImportError:
#     from faiss.contrib.datasets import DatasetBigANN
#
# # from datasets import load_sift1M
#
#
# ######################################################################################
# #                       실험을 위한 함수들 정의 (로깅, 설정변경)                             #
# ######################################################################################
# import logging
# import threading
# import subprocess
# import os
#
#
# def setup_logging():
#     log_filename = get_unique_log_filename('benchmark.log')
#     logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#
# def get_unique_log_filename(base_filename):
#     if not os.path.exists(base_filename):
#         return base_filename
#     else:
#         base, ext = os.path.splitext(base_filename)
#         counter = 1
#         while True:
#             new_filename = f"{base}{counter}{ext}"
#             if not os.path.exists(new_filename):
#                 return new_filename
#             counter += 1
#
#
# def record_log(stop_event):
#     while not stop_event.is_set():
#         with open("/proc/vmstat", "r") as vmstat_file:
#             vmstat = vmstat_file.read()
#
#         with open("/sys/devices/system/node/node0/meminfo", "r") as meminfo_node0_file:
#             meminfo_node0 = meminfo_node0_file.read()
#
#         with open("/sys/devices/system/node/node1/meminfo", "r") as meminfo_node1_file:
#             meminfo_node1 = meminfo_node1_file.read()
#
#         with open("/sys/devices/system/node/node2/meminfo", "r") as meminfo_node2_file:
#             meminfo_node2 = meminfo_node2_file.read()
#
#         with open("/sys/devices/system/node/node3/meminfo", "r") as meminfo_node3_file:
#             meminfo_node3 = meminfo_node3_file.read()
#
#         with open("/sys/devices/system/node/node4/meminfo", "r") as meminfo_node4_file:
#             meminfo_node4 = meminfo_node4_file.read()
#
#         with open("/sys/devices/system/node/node5/meminfo", "r") as meminfo_node5_file:
#             meminfo_node5 = meminfo_node5_file.read()
#
#
#         logging.info(f"VMSTAT:\n{vmstat}")
#         logging.info(f"NODE0 MEMINFO:\n{meminfo_node0}")
#         logging.info(f"NODE1 MEMINFO:\n{meminfo_node1}")
#         logging.info(f"NODE2 MEMINFO:\n{meminfo_node2}")
#         logging.info(f"NODE3 MEMINFO:\n{meminfo_node3}")
#         logging.info(f"NODE4 MEMINFO:\n{meminfo_node4}")
#         logging.info(f"NODE5 MEMINFO:\n{meminfo_node5}")
#
#         time.sleep(10)
#
#
# def enable_cores_on_node(node=1):
#     """
#     특정 NUMA 노드에 속한 모든 코어를 활성화.
#
#     Parameters:
#     - node (int): 활성화할 NUMA 노드 번호 (0 또는 1)
#     """
#     time.sleep(5)
#     if node == 0:
#         node_cpus = [
#             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#             20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
#             64, 65, 66, 67, 68, 69, 70, 71
#         ]
#     elif node == 1:
#         node_cpus = [
#             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
#             88, 89, 90, 91, 92, 93, 94, 95
#         ]
#     else:
#         print("Invalid NUMA node. Please specify 0 or 1.")
#         return
#
#     for cpu in node_cpus:
#         try:
#             # echo 명령어를 사용하여 코어 활성화
#             subprocess.run(f"echo 1 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
#             print(f"CPU {cpu} has been enabled.")
#         except subprocess.CalledProcessError:
#             print(f"Failed to enable CPU {cpu}.")
#         except Exception as e:
#             print(f"An error occurred: {e}")
#     time.sleep(30)
#
#
# def disable_cores_on_node(node=1):
#     """
#     특정 NUMA 노드에 속한 모든 코어를 비활성화.
#     Node 0의 CPU 0은 비활성화 불가.
#
#     Parameters:
#     - node (int): 비활성화할 NUMA 노드 번호 (0 또는 1)
#     """
#     time.sleep(5)
#     if node == 0:
#         node_cpus = [
#             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#             20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
#             64, 65, 66, 67, 68, 69, 70, 71
#         ]
#     elif node == 1:
#         node_cpus = [
#             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
#             44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
#             88, 89, 90, 91, 92, 93, 94, 95
#         ]
#     else:
#         print("Invalid NUMA node. Please specify 0 or 1.")
#         return
#
#     for cpu in node_cpus:
#         # CPU 0은 비활성화하지 않음
#         if node == 0 and cpu == 0:
#             print("Skipping CPU 0 (cannot be disabled).")
#             continue
#
#         try:
#             # echo 명령어를 사용하여 코어 비활성화
#             subprocess.run(f"echo 0 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
#             print(f"CPU {cpu} has been disabled.")
#         except subprocess.CalledProcessError:
#             print(f"Failed to disable CPU {cpu}.")
#         except Exception as e:
#             print(f"An error occurred: {e}")
#     time.sleep(30)
#
#
# def set_numa_balancing(config_num):
#     try:
#         # Use the value argument in the echo command
#         command = f"echo {config_num} | sudo tee /proc/sys/kernel/numa_balancing"
#         subprocess.run(command, shell=True, check=True)
#         print("Successfully set NUMA balancing.")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to set NUMA balancing: {e}")
#     time.sleep(10)
#
#
# def set_demotion_enabled(config_num):
#     """
#     Sets the value of /sys/kernel/mm/numa/demotion_enabled to the provided config_num.
#
#     Args:
#         config_num (int): The value to set for demotion_enabled (0 or 1).
#     """
#     try:
#         # Use the value argument in the echo command
#         command = f"echo {config_num} | sudo tee /sys/kernel/mm/numa/demotion_enabled"
#         subprocess.run(command, shell=True, check=True)
#         print(f"Successfully set demotion_enabled to {config_num}.")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to set demotion_enabled: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     time.sleep(10)
#
#
# ######################################################################################
# #                                                                                    #
# ######################################################################################
#
#
# k = int(sys.argv[1])
# todo = sys.argv[2:]
#
# print("load data")
#
# # xb, xq, xt, gt = load_sift1M()
#
# ds = DatasetBigANN()
# ds.basedir = '/home/smrc/nicedavid98/faiss_dataset/bigann/'
#
# xq = ds.get_queries()
# gt = ds.get_groundtruth()
# xt = ds.get_train()
#
# nq, d = xq.shape
#
# if todo == []:
#     todo = 'search save'.split()
#
#
# def evaluate(index):
#     # for timing with a single core
#     # faiss.omp_set_num_threads(1)
#
#     t0 = time.time()
#     D, I = index.search(xq, k)
#     t1 = time.time()
#
#     missing_rate = (I == -1).sum() / float(k * nq)
#     recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
#     print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
#         (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
#
#
# if 'save' in todo:
#     print("HNSW Index 저장")
#     index = faiss.IndexHNSWFlat(d, 32)
#     index.hnsw.efConstruction = 40
#
#     print("add")
#     # to see progress
#     index.verbose = True
#     for batch in ds.database_iterator(bs=1000000):
#         index.add(batch)
#
#     faiss.write_index(index, "hnsw_index")
#
# # HNSW Index 구조 파악을 위한 옵션 추가
# if 'print_level' in todo:
#     index = faiss.read_index("hnsw_index")
#     print("index.hnsw.max_level: ", index.hnsw.max_level)
#
#     levels = faiss.vector_to_array(index.hnsw.levels)
#     print("np.bincount(levels): ", np.bincount(levels))
#
#
# if 'search' in todo:
#     print("Testing HNSW Flat")
#
#     # logging 세팅
#     setup_logging()
#
#     # vmstat 로그 기록 스레드 시작
#     stop_event = threading.Event()
#     logger_thread = threading.Thread(target=record_log, args=(stop_event,))
#     logger_thread.start()
#
#     # 하나의 소켓만 사용하기 위해 Node 1의 core 비활성화
#     # logging.info(f"Searching : disable cores on node 1")
#     # disable_cores_on_node(1)
#     # logging.info(f"Searching : disabled cores on node 1")
#
#     # numa_balancing 옵션 설정
#     autonuma_config = 2
#     logging.info(f"Setting NUMA balancing to {autonuma_config}.")
#     set_numa_balancing(autonuma_config)
#     logging.info(f"Successfully set NUMA balancing to {autonuma_config}.")
#
#     # demotion_enabled 옵션 설정
#     demotion_config = 1
#     logging.info(f"Setting demotion_enabled to {demotion_config}.")
#     set_demotion_enabled(demotion_config)
#     logging.info(f"Successfully set demotion_enabled to {demotion_config}.")
#
#     logging.info(f"Searching : Load HNSW Index file")
#     index = faiss.read_index("hnsw_index")
#
#     for phase in range(10):
#         logging.info(f"Searching Phase {phase}")
#         for efSearch in 16, 32, 64, 128, 256:
#             for bounded_queue in [True, False]:
#                 print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
#                 index.hnsw.search_bounded_queue = bounded_queue
#                 index.hnsw.efSearch = efSearch
#                 evaluate(index)
#
#     # Node1 원상 복구 : 실험 종료에 의함.
#     # enable_cores_on_node(1)
#
#     # vmstat 로그 기록 스레드 종료
#     stop_event.set()
#     logger_thread.join()
#

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
import faiss
import logging
import threading
import subprocess
import os

try:
    from faiss.contrib.datasets_fb import DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import DatasetBigANN


######################################################################################
#                       실험을 위한 함수들 정의 (로깅, 설정변경)                             #
######################################################################################
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

        logging.info(f"VMSTAT:\n{vmstat}")
        logging.info(f"NODE0 MEMINFO:\n{meminfo_node0}")
        logging.info(f"NODE1 MEMINFO:\n{meminfo_node1}")

        time.sleep(10)


def record_dmesg_logs(pattern, log_file, stop_event):
    """
    Runs a dmesg command to filter logs by a specific pattern and writes them to a log file.

    Args:
        pattern (str): The pattern to filter in dmesg output (e.g., "th=", "period:").
        log_file (str): The file where filtered logs will be written.
        stop_event (threading.Event): Event to stop the logging thread.
    """
    with open(log_file, "w") as file:
        process = None
        try:
            # Use a subprocess to run `dmesg -w`
            process = subprocess.Popen(
                f"dmesg -w | grep \"{pattern}\"",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ""):
                if stop_event.is_set():
                    break
                file.write(line)
                file.flush()
        except Exception as e:
            print(f"Error while recording dmesg logs for pattern '{pattern}': {e}")
        finally:
            if process:
                process.terminate()


def set_numa_balancing(config_num):
    try:
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

k = int(sys.argv[1])
todo = sys.argv[2:]

print("load data")

ds = DatasetBigANN()
ds.basedir = '/home/smrc/nicedavid98/faiss_dataset/bigann/'

xq = ds.get_queries()
gt = ds.get_groundtruth()
xt = ds.get_train()

nq, d = xq.shape

if todo == []:
    todo = 'search save'.split()


def evaluate(index):
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
    index.verbose = True
    for batch in ds.database_iterator(bs=1000000):
        index.add(batch)

    faiss.write_index(index, "hnsw_index")

if 'search' in todo:
    print("Testing HNSW Flat")

    # logging 세팅
    setup_logging()

    # vmstat 로그 기록 스레드 시작
    stop_event = threading.Event()
    logger_thread = threading.Thread(target=record_log, args=(stop_event,))
    dmesg_th_thread = threading.Thread(target=record_dmesg_logs, args=("th=", "th.log", stop_event))
    dmesg_period_thread = threading.Thread(target=record_dmesg_logs, args=("period:", "period.log", stop_event))

    logger_thread.start()
    dmesg_th_thread.start()
    dmesg_period_thread.start()

    autonuma_config = 2
    logging.info(f"Setting NUMA balancing to {autonuma_config}.")
    set_numa_balancing(autonuma_config)
    logging.info(f"Successfully set NUMA balancing to {autonuma_config}.")

    demotion_config = 1
    logging.info(f"Setting demotion_enabled to {demotion_config}.")
    set_demotion_enabled(demotion_config)
    logging.info(f"Successfully set demotion_enabled to {demotion_config}.")

    logging.info(f"Searching : Load HNSW Index file")
    index = faiss.read_index("hnsw_index")

    for phase in range(10):
        logging.info(f"Searching Phase {phase}")
        for efSearch in 16, 32, 64, 128, 256:
            for bounded_queue in [True, False]:
                print("efSearch", efSearch, "bounded queue", bounded_queue, end=' ')
                index.hnsw.search_bounded_queue = bounded_queue
                index.hnsw.efSearch = efSearch
                evaluate(index)

    # vmstat 로그 기록 스레드 종료
    stop_event.set()
    logger_thread.join()
    dmesg_th_thread.join()
    dmesg_period_thread.join()
