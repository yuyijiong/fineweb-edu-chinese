import os
import re
from tqdm import tqdm
import pathlib
from openai import OpenAI
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn',True)

def vllm_one_df(df,device:str,model_path,batchsize,tokenizer):
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    from vllm import LLM, SamplingParams
    #from vllm.transformers_utils.config import
    import torch
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=4000)
    llm=LLM(model=model_path,tensor_parallel_size=1,dtype="auto",trust_remote_code=True, enforce_eager=False,quantization=None)
    for i in tqdm(range(0,len(df),batchsize),desc="处理数据,bs={}".format(batchsize),mininterval=5):
        web_text=df.loc[i:i+batchsize-1,"text"].tolist()

        answers=eval_edu(web_text,llm,sampling_params,tokenizer)
        df.loc[i:i+batchsize-1,"edu_eval"]=answers
        scores=parse_score(answers)
        df.loc[i:i+batchsize-1,"score"]=scores

        if i==0:
            print("Input:",web_text[0])
            print("\n\nOutput:",answers[0])

    del llm
    torch.cuda.empty_cache()
    return df

def apply_template(prompt,tokenizer):
    return tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)


def answer_vllm(prompt,llm,sampling_params):

    if isinstance(prompt,str):
        outputs = llm.generate([prompt], sampling_params)

        return outputs[0].outputs[0].text
    else:
        outputs = llm.generate(prompt, sampling_params,use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

def eval_edu(web_text,llm,sampling_params,tokenizer):

    prompt_temp='''以下是一段网页内容摘录。请使用以下5分制评分系统来评估该网页的教育价值,以及它在小学到中学阶段的教学环境中的实用性:
0分：如果网页没有提供任何教育价值,完全由无关信息(如广告、宣传材料)组成。
1分：如果网页提供了一些与教育主题相关的基本信息,即使包含一些无关或非学术内容(如广告和宣传材料)。
2分：如果网页涉及某些与教育相关的元素,但与教育标准不太吻合。它可能将教育内容与非教育材料混杂,对潜在有用的主题进行浅显概述,或以不连贯的写作风格呈现信息。
3分：如果网页适合教育使用,并介绍了与学校课程相关的关键概念。内容连贯但可能不全面,或包含一些无关信息。它可能类似于教科书的介绍部分或基础教程,适合学习但有明显局限,如涉及对中学生来说过于复杂的概念。
4分：如果网页对不高于中学水平的教育目的高度相关和有益,表现出清晰一致的写作风格。它可能类似于教科书的一个章节或教程,提供大量教育内容,包括练习和解答,极少包含无关信息,且概念对中学生来说不会过于深奥。内容连贯、重点突出,对结构化学习有价值。
5分：如果网页摘录在教育价值上表现出色,完全适合小学或中学教学。它遵循详细的推理过程,写作风格易于理解,对主题提供深刻而全面的见解,不包含任何非教育性或复杂内容。

网页内容摘录:
{}

在审查这段摘录后：请简要地为您的评分进行合理的解释，最多不超过100字，最后以“教育得分：<分数>”的格式结束。请根据所列出的标准系统地赋予分数。'''

    prompt_temp = '''以下是一段网页内容摘录。请使用以下5分制评分系统来评估该网页的写作水平、教育价值和实用性:
0分：如果网页没有提供任何教育价值,完全由无关信息(如广告、宣传材料、少儿不宜内容)组成。
1分：如果网页提供了一些可能有教育价值的基本信息,即使包含一些无关或非学术内容(如广告和宣传材料)。
2分：如果网页涉及某些与教育相关的元素,但与教育标准不太吻合。它可能将教育内容与非教育材料混杂,对潜在的有用的主题进行浅显概述,或以不连贯的写作风格呈现信息。
3分：如果网页适合教育使用,并介绍了与某些学校课程中可能学到的关键概念，或对个人发展有用的实用信息。它的内容连贯但可能不全面,或包含一些无关信息。它可能类似于教科书的一小段节选,可以学习但有明显局限,如涉及过于复杂的概念、过于具体的不重要事件。
4分：如果网页与教育高度相关，对个人学习发展有益,表现出清晰一致的写作风格。它可能类似于教科书的一个章节或教程,提供大量教育内容,极少包含无关信息,且概念对学生来说不会过于深奥。内容连贯、重点突出,对结构化学习有价值。
5分：如果网页摘录在教育价值上表现极好,完全适合小学、中学或大学教学或专业人士学习。它遵循详细的推理过程,写作风格易于理解,对主题提供深刻而全面的见解,不包含任何非教育性或无实用意义内容。

网页内容摘录:
{}

在审查这段网页摘录后：请简要地为您的评分进行合理的解释，最多不超过100字，最后以“教育得分：<分数>”的格式结束。请根据所列出的标准系统地赋予分数。'''

    if isinstance(web_text,str):
        web_text=[web_text]

    #每个text不超过2000字
    web_text=[text[:2000]+"......" if len(text)>2000 else text for text in web_text]

    prompt=[prompt_temp.format(text) for text in web_text]

    # 为每个prompt增加chat模板
    prompt = [apply_template(p, tokenizer) for p in prompt]

    answer=answer_vllm(prompt,llm,sampling_params)

    return answer

def parse_score(answers:list):
    scores=[]
    for answer in answers:
        #从答案中提取分数，即 教育得分： 后面的首个数字。如果提取不到，返回None
        score=re.findall(r"教育得分：[\s\S]*?(\d)",answer)
        if score:
            score=score[0]
        else:
            score=None
        scores.append(score)
    return scores

if __name__ == '__main__':
    devices=[1,2,3,4,5,6]
    num_device=len(devices)

    model_path = "//data/models/Qwen2.5-14B-Instruct"
    batchsize=64

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path)

    import pandas as pd

    save_dir="/data/datasets/教育打分数据集-qwen14b"
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    #获取dir下所有parquet文件
    files=[
        "/data/datasets/csdn/md_0.parquet",
        "/data/dataset/nlp/CN/ChinaNews-cn_edu_scored/part-006853-a894b46e.parquet",
        "/data/dataset/nlp/CN/Exam-cn_edu_scored/part-003756-a894b46e.parquet",
        "/data/dataset/wudao/WuDaoCorpus2.0_base_200G_edu_scored/part-2021009337.parquet",
        "/data/dataset/OpenDataLab___MiChao/raw_edu_scored/part-00000-4fc044a0-7fb1-4890-9bdb-64633e51520d-c000.parquet",

        # "/data/dataset/nlp/CN/WebText-cn_edu_scored/part-000036-a894b46e.parquet",
        # "/data/datasets/chinese_fineweb_edu_parquet_shuffle/00000.parquet",
        # "/data/dataset/nlp/EN/WebText-en/data1.parquet"
    ]
    #files.sort()

    for file in files:
        df=pd.read_parquet(file)
        #如果样本数大于100w条，随机抽取100w条
        if len(df)>500000:
            df=df.sample(n=500000,random_state=0)

        print("文件",file,"样本数",len(df))

        #如果有content列，重命名为text
        if 'content' in df.columns:
            df.rename(columns={'content':'text'},inplace=True)
        elif 'md' in df.columns:
            df.rename(columns={'md':'text'},inplace=True)

        df_list=[]
        #将df分成num_device份
        for i in range(num_device):
            df_=df[i::num_device].copy()
            df_.reset_index(drop=True,inplace=True)
            df_list.append(df_)

        #多进程，使用spawn模式
        with Pool(num_device) as p:
            df_list=p.starmap(vllm_one_df,[(df_list[i],str(device_id),model_path,batchsize,tokenizer) for i,device_id in enumerate(devices)])

        #合并
        df=pd.concat(df_list,ignore_index=True)
        #打印平均分
        print("平均分：",df["score"].mean())
        df.to_parquet(os.path.join(save_dir,os.path.basename(file)))


##CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "/data/models/Qwen/Qwen2-72B-Instruct" --tensor-parallel-size 4 --dtype auto

