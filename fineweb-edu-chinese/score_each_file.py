import multiprocessing
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pyarrow.parquet as pq

multiprocessing.set_start_method('spawn',True)

def process_file_list(file_list,device:str,model_name,batchsize):
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    #print("device:",device)
    import torch

    model=AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   torch_dtype=torch.float16,
                                                                   num_labels=1,
                                                                   problem_type="regression",
                                                                   trust_remote_code=True).to("cuda").eval()

    tokenizer=AutoTokenizer.from_pretrained(model_name)

    for file in file_list:
        df = pd.read_parquet(file)
        print("文件：",file)

        #如果有content列，重命名为text
        if 'content' in df.columns:
            df.rename(columns={'content':'text'},inplace=True)

        df=score_one_df(df, batchsize, model, tokenizer)


        df['new'] = 1

        df.to_parquet(file)
        print("保存文件：",file)

    del model
    torch.cuda.empty_cache()

def score_one_df(df, batchsize, model, tokenizer):
    import torch

    for i in tqdm(range(0,len(df),batchsize),desc="处理数据,bs={}".format(batchsize),mininterval=100):
        texts = df.loc[i:min(i + batchsize, len(df)), 'text'].tolist()
        scores = get_scores(texts,model,tokenizer)
        df.loc[i:min(i + batchsize, len(df)), 'score'] = scores
        if i==0:
            print(scores[:5])

    del model
    torch.cuda.empty_cache()
    return df



def get_scores(texts,model,tokenizer):
    import torch
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.autocast("cuda"):
            outputs = model(**inputs)
        #将输出logits转化为列表
        scores = outputs.logits.squeeze().tolist()
        return scores
    except Exception as e:
        #print(e)
        raise e
        return [0]*len(texts)


if __name__ == '__main__':

    model_name = "/data/models/bge-reranker-base-edu-cls-qwen14b/checkpoint-6000"

    num_workers=16

    devices=([0,1,2,3,4,5,6,7]*100)[:num_workers]

    batchsize=128


    dirs=["//data/datasets//map-cc/cc"]

    for dirname in dirs:
        #df_dir=os.path.join(df_all_dir,dirname)
        df_dir=dirname

        #获取dir下所有jsonl.gz文件
        files = [f for f in os.listdir(df_dir) if f.endswith('.parquet')]
        files.sort(reverse=True)
        print("文件数量：",len(files))

        dir_id=dirs.index(dirname)

        files_to_process=[file for file in files if "new" not in pq.ParquetFile(os.path.join(df_dir,file)).schema.names]
        files_to_process=[os.path.join(df_dir,file) for file in files_to_process]

        print("需要处理的文件数量：",len(files_to_process))

        #将文件分成num_workers份
        files_list = [files_to_process[i::num_workers] for i in range(num_workers)]

        #多进程
        with Pool(num_workers) as p:
            p.starmap(process_file_list,
                      [(files_list[i], str(devices[i]), model_name, batchsize) for i in range(num_workers)])




