import os

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from text_dedup.utils import INDEX_COLUMN
import ray
import pathlib
from multiprocessing import Pool
from functools import partial

USE_RAY = False

# 读取指定后缀的文件路径
def get_file_paths(dir, suffix, subfolder=True, exclude_suffix=None):
    #如果dir不存在，报错
    if not os.path.exists(dir):
        raise ValueError("dir not exist：",dir)


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


@ray.remote
def load_parquet_ds(file,min_score=0.4):
    #ds=datasets.load_dataset("parquet",data_files=file,cache_dir="/share/datasets/hf_cache/",keep_in_memory=False)["train"]
    try:
        ds=Dataset.from_parquet(file,cache_dir="/backup/datasets/hf_cache/")
        print("读取文件完成：", file)

    except Exception as e:
        print("读取文件失败：",file)
        print(e)
        return None
    #只保留text列和score列
    if "score" not in ds.column_names:
        print("文件中没有score列：",file)
        return None
    ds=ds.select_columns(["text","score"])
    #只保留score大于0.4的样本
    ds=ds.filter(lambda x:np.array(x["score"])>=min_score,desc="Score filter...",batched=True)
    #增加一列，记录文件名
    ds=ds.add_column("file_name",[str(file)]*len(ds))

    return ds

def load_parquet_ds2(file,min_score=0.4):
    #ds=datasets.load_dataset("parquet",data_files=file,cache_dir="/share/datasets/hf_cache/",keep_in_memory=False)["train"]
    try:
        ds=Dataset.from_parquet(file,cache_dir="/backup/datasets/hf_cache/")
        print("读取文件完成：", file)

    except Exception as e:
        print("读取文件失败：",file)
        print(e)
        return None
    #只保留text列和score列
    if "score" not in ds.column_names:
        print("文件中没有score列：",file)
        return None
    ds=ds.select_columns(["text","score"])
    #只保留score大于0.4的样本
    ds=ds.filter(lambda x:np.array(x["score"])>=min_score,desc="Score filter...",batched=True,keep_in_memory=True)
    #增加一列，记录文件名
    ds=ds.add_column("file_name",[str(file)]*len(ds))
    return ds

def load_hf_dataset(data_dir_list, suffix=".parquet", min_score=0.4):
    # 对于每个dir，读取所有的parquet文件
    ds_list = []
    dir_and_files = {}
    for data_dir in data_dir_list:


        # 读取所有的parquet文件，包括子目录
        files = get_file_paths(data_dir, suffix, subfolder=True)
        files.sort()
        dir_and_files[data_dir] = files

    # 打印每个dir下的文件数量
    for data_dir, files in dir_and_files.items():
        print(data_dir, "文件数量：", len(files))

    # 使用Ray并行读取所有的parquet文件
    # 初始化Ray
    for data_dir, files in dir_and_files.items():
        this_dir_save_path = "/backup/datasets/hf_cache/"+data_dir.replace("/", "_")

        #如果已经读取过，直接加载
        if os.path.exists(this_dir_save_path):
            print("已经读取过，直接加载：",this_dir_save_path)
            ds_this_dir=Dataset.load_from_disk(this_dir_save_path)
        else:
            if USE_RAY:
                # 使用Ray远程函数并行加载数据
                ray.init()
                df_load=[load_parquet_ds.remote(file, min_score) for file in files]
                #获取结果
                ray_ds_list=ray.get(df_load)
                ds_this_dir=[ds for ds in ray_ds_list if ds is not None]
                ds_this_dir=concatenate_datasets(ds_this_dir)
                # 关闭Ray
                ray.shutdown()
            else:
                # 使用多进程并行加载数据
                with Pool(32) as p:
                    ds_this_dir = p.map(partial(load_parquet_ds2, min_score=min_score), files)
                ds_this_dir = [ds for ds in ds_this_dir if ds is not None]
                ds_this_dir = concatenate_datasets(ds_this_dir)

            #保存ds_this_dir
            pathlib.Path(this_dir_save_path).mkdir(parents=True,exist_ok=True)
            ds_this_dir.save_to_disk(this_dir_save_path)

        ds_list.append(ds_this_dir)

    ds = concatenate_datasets(ds_list)

    index_column=np.arange(len(ds))
    ds.add_column(INDEX_COLUMN,index_column)

    #ds = ds.map(lambda x, i: {INDEX_COLUMN: i}, with_indices=True, num_proc=64)
    return ds, None


if __name__ == '__main__':
    data_dir_list = [
        "/data/dataset/CASIA-LM/ChineseWebText/cleaner_subset_score_3/2021-43_edu_scored_3"
    ]
    ds, _ = load_hf_dataset(data_dir_list, suffix=".parquet", min_score=0.4)
    print("数据量：", len(ds))

