import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
import json
import pathlib
import setproctitle
setproctitle.setproctitle("ray_read_parquet_info")

# 读取指定后缀的文件路径
def get_file_paths(dir, suffix, subfolder=True, exclude_suffix=None):
    file_path_list = []
    if subfolder == False:
        for file in os.listdir(dir):
            if file.endswith(suffix):
                file_path_list.append(os.path.join(dir, file))
    else:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(suffix):
                    file_path_list.append(os.path.join(root, file))
    if exclude_suffix != None:
        file_path_list = [file_path for file_path in file_path_list if not file_path.endswith(exclude_suffix)]

    return file_path_list

def process_file(file):
    try:
        df = pd.read_parquet(file)
        print("读取文件完成：", file)
    except Exception as e:
        print("读取文件失败：", file)
        print(e)
        return None


    # 初始化统计字典
    score_stats = defaultdict(int)
    length_stats = defaultdict(int)

    # 统计score分布
    if "score" in df.columns:
        score_bins = [-10, 0.2, 0.4, 0.6, 0.8,10]
        score_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
        score_counts = pd.cut(df["score"], bins=score_bins, labels=score_labels, right=False).value_counts()
        for label, count in score_counts.items():
            score_stats[label] += count

    # 统计text长度分布
    if "text" in df.columns:
        df["text_length"] = df["text"].apply(len)
        length_bins = [0, 500, 1000, 2000, 4000, 8000, float('inf')]
        length_labels = ["0-0.5k", "0.5-1k", "1-2k", "2k-4k", "4k-8k", "8k+"]
        length_counts = pd.cut(df["text_length"], bins=length_bins, labels=length_labels, right=False).value_counts()
        for label, count in length_counts.items():
            length_stats[label] += count

    return {"file": file, "score_stats": dict(score_stats), "length_stats": dict(length_stats)}

def aggregate_stats(data_dir_list, suffix=".parquet"):
    # 对于每个dir，读取所有的parquet文件
    dir_and_files = {}
    for data_dir in data_dir_list:
        # 读取所有的parquet文件，包括子目录
        files = get_file_paths(data_dir, suffix, subfolder=True)
        files.sort()
        dir_and_files[data_dir] = files

    # 打印每个dir下的文件数量
    for data_dir, files in dir_and_files.items():
        print(data_dir, "文件数量：", len(files))

    # 使用多进程并行处理文件
    all_stats = []
    for data_dir, files in dir_and_files.items():
        data_dir_name = data_dir.replace("/", "_")
        #如果已经统计过，直接读取
        if os.path.exists(stat_save_dir + "/score_stats_" + data_dir_name + ".json"):
            with open(stat_save_dir + "/score_stats_" + data_dir_name + ".json", "r") as f:
                score_stats = json.load(f)
            with open(stat_save_dir + "/length_stats_" + data_dir_name + ".json", "r") as f:
                length_stats = json.load(f)
            all_stats.append({"file": data_dir, "score_stats": score_stats, "length_stats": length_stats})
            continue

        with Pool(16) as p:
            stats_this_dir = p.map(process_file, files)
        stats_this_dir = [stats for stats in stats_this_dir if stats is not None]
        #将这个dir下的所有文件的统计结果合并
        aggregated_score_stats_this_dir = defaultdict(int)
        aggregated_length_stats_this_dir = defaultdict(int)
        for stats in stats_this_dir:
            for key, value in stats["score_stats"].items():
                aggregated_score_stats_this_dir[key] += value
            for key, value in stats["length_stats"].items():
                aggregated_length_stats_this_dir[key] += value

        #保存这个dir下的统计结果
        data_dir_name = data_dir.replace("/", "_")
        with open(stat_save_dir + "/score_stats_" + data_dir_name + ".json", "w") as f:
            json.dump(dict(aggregated_score_stats_this_dir), f)
        with open(stat_save_dir + "/length_stats_" + data_dir_name + ".json", "w") as f:
            json.dump(dict(aggregated_length_stats_this_dir), f)

        all_stats.append({"file": data_dir, "score_stats": dict(aggregated_score_stats_this_dir), "length_stats": dict(aggregated_length_stats_this_dir)})

    # 聚合所有文件的统计结果
    aggregated_score_stats = defaultdict(int)
    aggregated_length_stats = defaultdict(int)

    for stats in all_stats:
        for key, value in stats["score_stats"].items():
            aggregated_score_stats[key] += value
        for key, value in stats["length_stats"].items():
            aggregated_length_stats[key] += value

    return dict(aggregated_score_stats), dict(aggregated_length_stats)

if __name__ == '__main__':
    data_dir_list = [
        "//backup/dataset/BAAI/CCI3-Data/data_edu_scored",

        "//backup/dataset/BAAI/IndustryCorpus2-edu-scored",

        "//data/dataset/OpenDataLab___MiChao/raw_edu_scored",

        "/backup/dataset/nlp/CN",

        "/data/dataset/CASIA-LM/ChineseWebText/cleaner_subset",

        "//backup/dataset/wudao/WuDaoCorpus2.0_base_200G_edu_scored_edu_scored",

        "/data/datasets/chinese-fineweb-edu-v2-unicom/tele/data_edu_scored_edu_scored",

        "//data/datasets/chinese-fineweb-edu-v2-unicom/skypile/data_edu_scored_edu_scored"
    ]

    data_dir_list=["/share/datasets/chinese_cosmopedia"]

    stat_save_dir="./cosmo-zh-stats"
    pathlib.Path(stat_save_dir).mkdir(parents=True, exist_ok=True)

    score_stats, length_stats = aggregate_stats(data_dir_list, suffix=".parquet")
    print("Score 统计结果：", score_stats)
    print("Text 长度统计结果：", length_stats)
    #保存
    with open(stat_save_dir + "/score_stats_all.json", "w") as f:
        json.dump(score_stats, f)
    with open(stat_save_dir + "/length_stats_all.json", "w") as f:
        json.dump(length_stats, f)