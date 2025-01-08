import pandas as pd
import os
import pathlib
from multiprocessing import Pool

def filter_and_save(source_dir,save_dir_4,save_dir_3_4,file_name):
    df = pd.read_parquet(os.path.join(source_dir, file_name))
    #按照text列去重
    df.drop_duplicates(subset=["text"],inplace=True)
    #筛选分数大于等于0.6且小于0.8的数据
    df_f=df[(df["score"]>=0.6) & (df["score"]<0.8)]
    df_f.to_parquet(os.path.join(save_dir_3_4, file_name), index=False)
    print("3-4分 筛选前数据量：{}，筛选后数据量：{}".format(len(df),len(df_f)))
    #筛选分数大于等于0.8的数据
    df_f=df[(df["score"]>=0.8) & (df["score"]<2)]
    df_f.to_parquet(os.path.join(save_dir_4, file_name), index=False)
    print("4分以上 筛选前数据量：{}，筛选后数据量：{}".format(len(df),len(df_f)))
    print("文件{}处理完成".format(file_name))

if __name__ == '__main__':
    df_all_dir="cc-scored"
    save_all_dir_4="cc-scored_3-4"
    save_all_dir_3_4="cc-scored_4-5"
    #获取所有文件夹名
    dirs=[d for d in os.listdir(df_all_dir) if os.path.isdir(os.path.join(df_all_dir,d))]

    for dir in dirs:
        df_dir=os.path.join(df_all_dir,dir)
        save_dir_4=os.path.join(save_all_dir_4,dir)
        save_dir_3_4=os.path.join(save_all_dir_3_4,dir)
        pathlib.Path(save_dir_4).mkdir(parents=True,exist_ok=True)
        pathlib.Path(save_dir_3_4).mkdir(parents=True,exist_ok=True)
        print("保存到：",save_dir_4,save_dir_3_4)
        #获取dir下所有parquet文件
        files = [f for f in os.listdir(df_dir) if f.endswith('.parquet')]
        files.sort()
        print("文件数量：",len(files))
        #多进程处理每个文件
        num_workers=16
        with Pool(num_workers) as p:
            p.starmap(filter_and_save,[(df_dir,save_dir_4,save_dir_3_4,file) for file in files])

