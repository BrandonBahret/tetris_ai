import numpy as np
import pandas as pd

import os
import time

from Types.Singleton import Singleton


def pad_data(data_list, pad_value = np.nan):
    max_length = max(len(a) for a in data_list)
    data_out = []
    for arr in data_list:
        len_delta = max_length - len(arr)
        if len_delta > 0:
            data_out += [arr + [pad_value] * len_delta]
        else:
            data_out += [arr]

    return data_out

class PerformanceChecker(metaclass=Singleton):

    def __init__(self):
        self.t_delta_accumulator_dict = {}

    @staticmethod
    def check_performance(method, *args, **kwargs):
        return PerformanceChecker().check(method, args, kwargs)

    def check(self, method, args, kwargs):
        method_name = method.__qualname__
        
        timestamps = self.get_timestamps(method_name)
        timestamps.append(time.time())

        t_begin = time.perf_counter_ns()
        result = method.__call__(*args, **kwargs)
        t_end = time.perf_counter_ns()

        accumulator = self.get_accumulator(method_name)
        t_delta_ms = (t_end - t_begin) / 1e6
        accumulator.append(t_delta_ms)

        return result

    def get_accumulator(self, method_name):
        does_accumulator_exist = method_name in self.t_delta_accumulator_dict.keys()
        if not does_accumulator_exist:
            self.t_delta_accumulator_dict[method_name] = [[], []]
                
        return self.t_delta_accumulator_dict[method_name][0]

    def get_timestamps(self, method_name):
        does_accumulator_exist = method_name in self.t_delta_accumulator_dict.keys()
        if not does_accumulator_exist:
            self.t_delta_accumulator_dict[method_name] = [[], []]
                
        return self.t_delta_accumulator_dict[method_name][1]

    @staticmethod
    def format_time(time_in_ms):
        if time_in_ms >= 1000:
            return f"{time_in_ms/1000:0.2f}s"
        elif time_in_ms >= 1:
            return f"{time_in_ms:0.2f}ms"
        elif time_in_ms >= 0.01:
            return f"{time_in_ms*1e3:0.2f}μs"
        else:
            return f"{time_in_ms*1e6:0.2f}ns"

    @staticmethod
    def print_recent_performance(method):
        method_name = method.__qualname__
        t_delta_accumulator = PerformanceChecker().get_accumulator(method_name)
        t_delta_ms = t_delta_accumulator[-1]
        
        t_delta_str = PerformanceChecker.format_time(t_delta_ms)
        print(f"{method_name} (exec): {t_delta_str}ms")

    @staticmethod
    def print_average_performance(method):
        method_name = method.__qualname__
        t_delta_accumulator = PerformanceChecker().get_accumulator(method_name)
        t_delta_ms = np.mean(t_delta_accumulator)
        
        t_delta_str = PerformanceChecker.format_time(t_delta_ms)
        print(f"{method_name} (ave exec): {t_delta_str}ms")

    @staticmethod
    def print_median_performance(method):
        method_name = method.__qualname__
        t_delta_accumulator = PerformanceChecker().get_accumulator(method_name)
        t_delta_ms = np.median(t_delta_accumulator)
        
        t_delta_str = PerformanceChecker.format_time(t_delta_ms)
        print(f"{method_name} (median exec): {t_delta_str}ms")

    def print_median_performances(self):
        os.system("cls")

        for method_name in self.t_delta_accumulator_dict:
            accumulator = self.get_accumulator(method_name)
            t_delta_ms = np.median(accumulator)
            t_delta_str = PerformanceChecker.format_time(t_delta_ms)

            timestamps = self.get_timestamps(method_name)
            timestamps_diff = np.diff(timestamps)
            median_timestamps_diff = np.median(timestamps_diff)
            median_timestamps_diff_str = PerformanceChecker.format_time(median_timestamps_diff*1000)

            t_delta_ave_ms = np.mean(accumulator)
            t_delta_ave_str = PerformanceChecker.format_time(t_delta_ave_ms)
            print(f"{method_name} (ave exec): {t_delta_ave_str}")

            print(f"{method_name} (median exec): {t_delta_str}")
            print(f"{method_name} (median time gap): {median_timestamps_diff_str}")
            print("")

    def save_performance_data(self, filename):
        data = self.t_delta_accumulator_dict
        method_names = data.keys()

        data_array = list(data.values())
        data_array = [d[0] for d in data_array]
        data_array = pad_data(data_array)
        data_array = np.array(data_array)

        df = pd.DataFrame(data_array.T, columns=method_names)
        df.to_csv(filename)
