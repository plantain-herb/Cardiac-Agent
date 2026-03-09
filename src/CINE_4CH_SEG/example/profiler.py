import json
import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
import nvsmi
import psutil


class Profiler(multiprocessing.Process):
    def __init__(self, delay, pid):
        super(Profiler, self).__init__()

        self.gpu_monitor = GPUMonitor(pid)
        self.cpu_monitor = CPUMonitor(pid)

        self.__pid = pid
        self.stop_signal = multiprocessing.Manager().Value(bool, False)
        self.start_time = multiprocessing.Manager().Value(float, time.time())
        self.end_time = multiprocessing.Manager().Value(float, time.time())
        self.patient_name = multiprocessing.Manager().Value(str, "")
        self.time_flag = multiprocessing.Manager().Value(bool, False)
        self.delay = delay  # 采样间隔，由于采样代码本身会占用0.4s左右，因此并不准确
        time_stamp = time.time()
        self.db = f"./profiler_{time_stamp}.txt"
        self.report = "./report.txt"

    def run(self):
        with open(self.db, "w") as f:
            while not self.stop_signal.value:
                gpu_sample = self.gpu_monitor.sample()
                cpu_sample = self.cpu_monitor.sample()
                samples = {"cpu": cpu_sample, "gpu": gpu_sample}
                json.dump(samples, f)
                f.write("\n")

                if self.time_flag.value:
                    self.time_flag.value = False
                    time_stamp = {"time_stamp": [self.start_time.value, self.end_time.value, self.patient_name.value]}
                    json.dump(time_stamp, f)
                    f.write("\n")
                f.flush()

                time.sleep(self.delay)

    def generate_report(self):
        time_stamps, cpu_mems, cpu_percents = [], [], []
        gpu_mems, gpu_utils = defaultdict(list), defaultdict(list)

        with open(self.db, "r") as f:
            lines = f.readlines()

        os.remove(self.db)
        for line in lines:
            data = json.loads(line)
            if "time_stamp" in data.keys():
                time_stamps.append([data["time_stamp"][0], data["time_stamp"][1]])
            else:
                cpu_mems.append([data["cpu"]["time_stamp"], data["cpu"]["memory"]])
                cpu_percents.append([data["cpu"]["time_stamp"], data["cpu"]["percent"]])
                for gpu_id, gpu_info in data["gpu"].items():
                    gpu_mems[gpu_id].append([gpu_info["time_stamp"], gpu_info["memory"]])
                    gpu_utils[gpu_id].append([gpu_info["time_stamp"], gpu_info["util"]])

        time_stamps, cpu_mems, cpu_percents = np.array(time_stamps), np.array(cpu_mems), np.array(cpu_percents)
        cpu_mems = self.__get_data_in_time_stamps(cpu_mems, time_stamps)
        cpu_percents = self.__get_data_in_time_stamps(cpu_percents, time_stamps)

        for gpu_id, gpu_mem in gpu_mems.items():
            gpu_mem = np.array(gpu_mem)
            gpu_mem = self.__get_data_in_time_stamps(gpu_mem, time_stamps)
            gpu_mems[gpu_id] = gpu_mem

        for gpu_id, gpu_util in gpu_utils.items():
            gpu_util = np.array(gpu_util)
            gpu_util = self.__get_data_in_time_stamps(gpu_util, time_stamps)
            gpu_utils[gpu_id] = gpu_util

        with open(self.report, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
            f.write(self.__generate_report_string("cpu_percentage", cpu_percents[:, 1]))
            f.write(self.__generate_report_string("cpu_memory_usage", cpu_mems[:, 1]))
            f.write(self.__generate_report_string("time_usage", time_stamps[:, 1] - time_stamps[:, 0]))

            for gpu_id, gpu_util in gpu_utils.items():
                f.write(self.__generate_report_string(f"gpu_{gpu_id}_util", gpu_util[:, 1]))
            for gpu_id, gpu_mem in gpu_mems.items():
                f.write(self.__generate_report_string(f"gpu_{gpu_id}_memory", gpu_mem[:, 1]))

    def __generate_report_string(self, name, data):
        ave = np.mean(data)
        max_ = np.max(data)
        per_95 = np.percentile(data, 95)
        return f"{name:} average: {ave}, max: {max_}, 95%: {per_95}\n"

    def __get_data_in_time_stamps(self, data, time_stamps):
        ret_data = []
        for time_stamp in time_stamps:
            mask = np.logical_and(data[:, 0] > time_stamp[0], data[:, 0] < time_stamp[1])
            for d in data[mask]:
                ret_data.append(d)
        return np.array(ret_data)

    def stop(self):
        time.sleep(2 * self.delay + 2)  # 等待写入
        self.stop_signal.value = True

    def set_time_stamp(self, start_time, end_time, patient_name):
        self.start_time.value = start_time
        self.end_time.value = end_time
        self.patient_name.value = patient_name
        self.time_flag.value = True


class CPUMonitor:
    def __init__(self, pid) -> None:
        self.pid = pid
        self.p = psutil.Process(pid)

    def sample(self):
        time_stamp = time.time()
        return {
            "time_stamp": time_stamp,
            "memory": self.p.memory_info().rss / 1048576.0,  # 1024**2=1048576
            "percent": self.p.cpu_percent(),
        }


class GPUMonitor:
    def __init__(self, pid) -> None:
        self.pid = pid

    def sample(self):
        gpu_util_dict = {}
        gpu_mem_dict = defaultdict(int)

        time_stamp = time.time()
        gpu_proc_infos = nvsmi.get_gpu_processes()  # 耗时大约0.3s
        gpu_infos = nvsmi.get_gpus()
        for gpu_info in gpu_infos:
            gpu_util_dict[gpu_info.id] = gpu_info.gpu_util

        for gpu_proc_info in gpu_proc_infos:
            if gpu_proc_info.pid != self.pid:
                continue
            gpu_mem_dict[gpu_proc_info.gpu_id] += gpu_proc_info.used_memory

        gpu_sample_info_dict = {}
        for gpu_id, mem in gpu_mem_dict.items():
            gpu_util = gpu_util_dict[gpu_id]
            gpu_sample_info_dict[gpu_id] = {"time_stamp": time_stamp, "memory": mem, "util": gpu_util}

        return gpu_sample_info_dict
