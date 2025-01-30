import time
import sys
import numpy as np
import faiss
import random
import logging
import threading
import subprocess
import os
import gc

try:
    from faiss.contrib.datasets_fb import DatasetBigANN
except ImportError:
    from faiss.contrib.datasets import DatasetBigANN

start_time = time.time()

def print_time(message):
    print(f"[{time.time() - start_time:.2f}s] {message}")

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

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    if sys.byteorder == 'big':
        d = x[:4][::-1].copy().view('int32')[0]
    else:
        d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

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

def set_numa_balancing(config_num):
    try:
        command = f"echo {config_num} | sudo tee /proc/sys/kernel/numa_balancing"
        subprocess.run(command, shell=True, check=True)
        print("Successfully set NUMA balancing.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set NUMA balancing: {e}")
    time.sleep(10)

def set_demotion_enabled(config_num):
    try:
        command = f"echo {config_num} | sudo tee /sys/kernel/mm/numa/demotion_enabled"
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully set demotion_enabled to {config_num}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set demotion_enabled: {e}")
    time.sleep(10)

def drop_caches():
    try:
        subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True, check=True)
        print_time("Successfully dropped caches")
    except subprocess.CalledProcessError as e:
        print_time(f"Failed to drop caches: {e}")
    time.sleep(10)

def generate_random_queries(dataset, num_queries):
    indices = random.sample(range(dataset.shape[0]), num_queries)
    return dataset[indices]

def perform_search(index, query_vectors, k):
    t0 = time.time()
    D, I = index.search(query_vectors, k)
    t1 = time.time()
    elapsed_time = t1 - t0
    qps = len(query_vectors) / elapsed_time
    print_time(f"{len(query_vectors)} queries completed in {elapsed_time:.2f} seconds (QPS: {qps:.2f})")
    return len(query_vectors)

k = int(sys.argv[1])
ef_search = int(sys.argv[2])

print_time("Load data")
data_path = '/home/smrc/nicedavid98/faiss_dataset/bigann/bigann_base.bvecs'
data = bvecs_mmap(data_path)
drop_caches()  # 데이터 매핑 이후 캐시 삭제
num_vectors = data.shape[0]
num_partitions = 10
partition_size = num_vectors // num_partitions


if 'search' in sys.argv:
    print_time("Testing HNSW Flat")

    # numa_balancing 옵션 설정
    autonuma_config = 0
    set_numa_balancing(autonuma_config)
    print_time(f"Successfully set NUMA balancing to {autonuma_config}.")

    # demotion_enabled 옵션 설정
    demotion_config = 0
    set_demotion_enabled(demotion_config)
    print_time(f"Successfully set demotion_enabled to {demotion_config}.")

    # logging 세팅
    setup_logging()

    # vmstat 로그 기록 스레드 시작
    stop_event = threading.Event()
    logger_thread = threading.Thread(target=record_log, args=(stop_event,))
    logger_thread.start()

    print_time("Load HNSW Index file")
    index = faiss.read_index("hnsw_index")
    print_time("Successfully loaded HNSW Index")

    # numa_balancing 옵션 설정
    autonuma_config = 2
    set_numa_balancing(autonuma_config)
    print_time(f"Successfully set NUMA balancing to {autonuma_config}.")

    # demotion_enabled 옵션 설정
    demotion_config = 1
    set_demotion_enabled(demotion_config)
    print_time(f"Successfully set demotion_enabled to {demotion_config}.")

    time.sleep(30)

    # efsearch 파라미터 설정
    index.hnsw.efSearch = ef_search
    print_time(f"efSearch parameter set to {ef_search}, Searching start.")

    total_queries = 0

    # 아래 배열 순서대로 검색
    partition_order = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
    for partition in partition_order:
        start_idx = partition * partition_size
        end_idx = (partition + 1) * partition_size if partition < num_partitions - 1 else num_vectors
        partition_data = data[start_idx:end_idx]

        print_time(f"Partition {partition + 1}/{num_partitions}: Vectors {start_idx} to {end_idx}")

        partition_start_time = time.time()
        while time.time() - partition_start_time < 20 * 60:  # 20분
            query_vectors = generate_random_queries(partition_data, 10000)
            total_queries += perform_search(index, query_vectors, k)

        del partition_data
        gc.collect()
        drop_caches()
        

    # vmstat 로그 기록 스레드 종료
    stop_event.set()
    logger_thread.join()

    print_time(f"Total queries processed: {total_queries}")