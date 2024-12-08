import time
import sys
import numpy as np
import faiss
import random
import logging
import threading
import subprocess
import os

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


def record_log(stop_event):
    while not stop_event.is_set():
        with open("/proc/vmstat", "r") as vmstat_file:
            vmstat = vmstat_file.read()

        with open("/sys/devices/system/node/node0/meminfo", "r") as meminfo_node0_file:
            meminfo_node0 = meminfo_node0_file.read()

        logging.info(f"VMSTAT:\n{vmstat}")
        logging.info(f"NODE0 MEMINFO:\n{meminfo_node0}")
        time.sleep(10)


def set_numa_balancing(config_num):
    try:
        command = f"echo {config_num} | sudo tee /proc/sys/kernel/numa_balancing"
        subprocess.run(command, shell=True, check=True)
        print("Successfully set NUMA balancing.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set NUMA balancing: {e}")


def set_demotion_enabled(config_num):
    try:
        command = f"echo {config_num} | sudo tee /sys/kernel/mm/numa/demotion_enabled"
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully set demotion_enabled to {config_num}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set demotion_enabled: {e}")


def drop_caches():
    try:
        subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True, check=True)
        print_time("Successfully dropped caches")
    except subprocess.CalledProcessError as e:
        print_time(f"Failed to drop caches: {e}")


k = int(sys.argv[1])
ef_search = int(sys.argv[2])

print_time("load data")

ds = DatasetBigANN()
ds.basedir = '/home/smrc/nicedavid98/faiss_dataset/bigann/'

xt = ds.get_train()
d = xt.shape[1]

def generate_random_queries(dataset, num_queries):
    indices = random.sample(range(dataset.shape[0]), num_queries)
    return dataset[indices]


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

    # 불필요한 캐시 메모리 드랍
    drop_caches()

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
    print_time(f"efSearch parameter set to {ef_search}")

    end_time = time.time() + 3600
    total_queries = 0
    while time.time() < end_time:
        random_queries = generate_random_queries(xt, 10000)
        t0 = time.time()
        D, I = index.search(random_queries, k)
        t1 = time.time()

        # QPS 계산 및 출력
        elapsed_time = t1 - t0
        qps = len(random_queries) / elapsed_time
        total_queries += len(random_queries)
        print_time(f"{len(random_queries)} queries completed in {elapsed_time:.2f} seconds (QPS: {qps:.2f})")

    # vmstat 로그 기록 스레드 종료
    stop_event.set()
    logger_thread.join()

    print_time(f"Total queries processed: {total_queries}")
