import pandas as pd
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
import re


if __name__ == '__main__':

    #apply显示进度条
    tqdm.pandas()
    import swifter

    temp_save_path="magpie_qwen_coder.json"

    # #如果temp已经存在，则直接读取
    # if os.path.exists(temp_save_path):
    #     df=pd.read_json(temp_save_path,orient="records",lines=True)
    #     print("样本数量：",len(df))
    #     print("文件已存在，直接读取")

    all_files=[]
    df_dir_list=["//data/yyj/自己生成微调数据/pipeline_cache/magpie-ultra-v1.0-chinese-qwen-coder/steps_data/magpie_generator_0_c303c425390647461de890e5f061340e8ef8ab15"]

    for df_dir in df_dir_list:
        #获取dir下所有json文件
        files=os.listdir(df_dir)
        files=[os.path.join(df_dir,file) for file in files if file.endswith(".json")]
        all_files.extend(files)

    print("文件数量：",len(all_files))
    df_list=[]
    #读取所有json文件
    for file in tqdm(all_files):
        with open(file,"r") as f:
            data=json.load(f)['data'][0]

        df=pd.DataFrame(data)

        # 如果没有conversation列，但是有instruction列，则将instruction和response组合为conversation
        if "conversation" not in df.columns:
            if "instruction" in df.columns and "response" in df.columns:
                df["conversation"] = df.apply(lambda x: [{"role": "user", "content": x["instruction"]},
                                                         {"role": "assistant", "content": x["response"]}], axis=1)
                df.drop(columns=["instruction", "response"], inplace=True)
            else:
                raise ValueError("没有conversation列")

        #如果file中包含deepseek，则将magpie_source设为deepseek-v2.5，否则设为qwen2.5-72b
        if "deepseek" in file:
            df["magpie_model"]="deepseek-v2.5"
        elif "coder" in file:
            df["magpie_model"]="qwen2.5-coder-32b"
        else:
            df["magpie_model"]="qwen2.5-72b"

        df_list.append(df)
    #合并
    df=pd.concat(df_list,ignore_index=True)
    print("样本数量：",len(df))

    #对conversation中的每个元素进行清洗
    df["conversation"]=df["conversation"].swifter.apply(clean_conversation)
    #删除为None的行
    df=df.dropna(subset=["conversation"])

    #获取第一个prompt
    df["first_prompt"]=df["conversation"].apply(lambda x:x[0]["content"])
    #获取第一个answer
    df["first_answer"]=df["conversation"].apply(lambda x:x[1]["content"])

    # df["second_prompt"]=df["conversation"].apply(lambda x:x[2]["content"])

    #根据first_prompt去重
    df=df.drop_duplicates("first_prompt")

    df.to_json(temp_save_path,orient="records",lines=True)

