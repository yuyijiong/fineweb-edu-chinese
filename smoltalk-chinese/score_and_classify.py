import pandas as pd
import os
from api回复 import model_answer
import json
from tqdm import tqdm
from multiprocessing import Pool
import re
import setproctitle
setproctitle.setproctitle("vllm score")

#打分prompt
PROMPT_SCORE = '''#Instruction
我会给出一条用户的指令，您需要根据用户的指令的清晰度、意图的明确性和表达的连贯性对其质量进行打分。
评分标准如下：
-0分：指令不完整，呈现较多的乱码或无意义内容；或者指令过长，格式不正确，除了用户指令外已经包含了AI助手给出的答案。
-1分：指令描述不清楚、意图模糊、语言不连贯。它缺失了必要的信息和背景。
-2分：指令有点不清楚或缺少重要细节。依然需要大量的澄清。
-3分：指令基本清晰和具体。但为了完全理解，可能需要一些额外的信息。
-4分：指令描述清晰、任务具体，而且格式规范。它为理解用户的意图提供了足够的上下文。
-5分：指令非常清晰、具体。它包含了全面的信息和背景。

##用户指令
```
{query}
```

##输出格式
给定用户指令，您首先需要进行评估，突出用户查询的优点和/或缺点。
然后，您需要通过填写[ ]中的占位符来给出打分，严格按照json字典的形式输出：
```
{{
"explanation":"[…]"，
"score":"[0分/1分/2分/3分/4分/5分]"
}}
```'''

#困难性prompt
PROMPT_DIFFICULTY = '''#Instruction
我会给出一条用户的指令，您首先需要确定这条指令包含的用户意图，然后根据用户的指令，标记出其难度级别。
##用户指令
```
{query}
```

##输出格式
给定用户指令，在输出中，您首先需要确定用户意图和解决此任务所需的知识。
然后，将用户查询的难度级别分类为“非常容易”、“容易”、”中等”、“困难”或“非常困难”。

现在，请填写[]中的占位符，严格按照json字典的形式输出以下用户意图和难度级别：
```
{{
"intent":"用户想要[…]"，
"knowledge":"为了解决这个问题，模型需要知道[…]"，
"difficulty":"[非常容易/容易/中等/困难/非常困难]"
}}
```
'''

#分类prompt
PROMPT_CLASSIFY = '''#Instruction
我会给出一条用户的指令，请将其分类为某一任务类型。
##用户查询
```
{query}
```
##标记用户输入
请为用户查询标记任务标签。您需要分析用户指令，并从下面的列表中选择最相关的任务标签。
all_task_tags=[
"信息寻求"，#用户要求提供有关各种主题的具体信息或事实。
"推理"，#需要逻辑思维、解决问题或处理复杂的想法。
"规划"，#用户在为活动和项目制定计划或策略时需要帮助。
"编辑"，#涉及编辑、改写、校对或其他与一般书面内容组成相关的任务。
"代码与调试"，#用户在编程中寻求编写、审查或修复代码的帮助。
"数学"，#与数学概念、问题和计算相关的查询。
"角色扮演"，#用户需要ChatGPT扮演某一角色或人物的场景。
"数据分析"，#请求涉及解释数据、统计数据或执行分析任务。
"创意写作"，#用户寻求创作故事、诗歌或其他创意文本的帮助。
"寻求建议"，#用户就各种个人或专业问题寻求建议或指导。
"头脑风暴"，#涉及产生想法、创造性思维或探索可能性。
"格式限制"，#用户要求输出必须符合特定格式或标准。
"文档问答"，#用户先给出参考文档，要求根据此文档的内容回答问题。
"总结"，#用户要求对给定的文本进行总结。
"其他" #不符合上述任何类别，具有杂项性质。
]
##输出格式：
请注意，您只能选择一个主标签。其他标签若也符合，可以添加到其他标签列表中。
现在，请通过填写<…>中的占位符，严格按照json格式输出下面的标签：
```
{{
"primary_tag": "<主标签>"，
"other_tags": ["<标签1>","<标签2>",…]
}}
```
'''
def vllm_one_df(df,device:str,model_path,batchsize,tokenizer,input_prompt_col="judge_prompt",output_col="model_reply"):
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    from vllm import LLM, SamplingParams
    #from vllm.transformers_utils.config import
    web_text_max_len = 800
    sampling_params = SamplingParams(temperature=0.0,max_tokens=256)
    llm=LLM(model=model_path,tensor_parallel_size=1,dtype="auto",trust_remote_code=True, enforce_eager=False,quantization=None)
    for i in tqdm(range(0,len(df),batchsize),desc="处理数据,bs={}".format(batchsize),mininterval=5):
        web_text=df.loc[i:min(i+batchsize-1,len(df)),input_prompt_col].tolist()
        try:
            answers=eval_edu(web_text,llm,sampling_params,tokenizer)
        except:
            answers=["no"]*len(web_text)

        df.loc[i:min(i+batchsize-1,len(df)),output_col]=answers
        if i==0:
            print("\n\nOutput:",answers)

    print("\n\nOutput:",answers[0])
    # del llm
    # torch.cuda.empty_cache()
    return df

def answer_vllm(prompt,llm,sampling_params):
    if isinstance(prompt,str):
        outputs = llm.generate([prompt], sampling_params,use_tqdm=False)

        return outputs[0].outputs[0].text
    else:
        outputs = llm.generate(prompt, sampling_params,use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

def eval_edu(prompt:list,llm,sampling_params,tokenizer):

    #为每个prompt增加chat模板
    prompt=[apply_template(p,tokenizer) for p in prompt]

    answer=answer_vllm(prompt,llm,sampling_params)

    return answer

def apply_template(prompt,tokenizer):
    return tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)


def clean_conversation(conversation:list[dict]):
    #每个dict的content都进行strip
    for i in range(len(conversation)):
        clean_content=conversation[i]["content"].replace("||","").replace("<|im_end|>","").strip()
        if len(clean_content)<3:
            return None
        conversation[i]["content"]=clean_content
    return conversation

if __name__ == '__main__':

    #apply显示进度条
    tqdm.pandas()
    import swifter

    final_save_path="magpie_deepseek_tagged.json"


    df=pd.read_json("magpie_qwen_everyday.json",lines=True)

    #df2=pd.read_json("/data_backup/yyj/自己生成微调数据/magpie_a800_temp.jsonl.gz",lines=True)

    #df=pd.concat([df,df2],ignore_index=True)
    print("number of data",len(df))

    #困难性prompt
    df["difficulty_prompt"]=df["first_prompt"].progress_apply(lambda x:PROMPT_DIFFICULTY.format(query=x))
    #打分prompt
    df["score_prompt"]=df["first_prompt"].progress_apply(lambda x:PROMPT_SCORE.format(query=x))
    #分类prompt
    df["classify_prompt"]=df["first_prompt"].progress_apply(lambda x:PROMPT_CLASSIFY.format(query=x))


    #多卡运行
    df_list = []
    num_device=7
    batchsize=250
    devices=[1,2,3,4,5,6,7]

    model_path="//data/models/Qwen/Qwen2.5-7B-Instruct"

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    # 将df分成num_device份
    for i in range(num_device):
        df_ = df[i::num_device].copy()
        df_.reset_index(drop=True, inplace=True)
        df_list.append(df_)


    # 多进程，使用spawn模式
    with Pool(num_device) as p:
        df_list = p.starmap(vllm_one_df,
                            [(df_list[i], str(device_id), model_path, batchsize, tokenizer,"difficulty_prompt","difficulty_reply") for i, device_id in
                             enumerate(devices)])

        df_list = p.starmap(vllm_one_df,
                            [(df_list[i], str(device_id), model_path, batchsize, tokenizer,"score_prompt","score_reply") for i, device_id in
                             enumerate(devices)])

        df_list = p.starmap(vllm_one_df,
                            [(df_list[i], str(device_id), model_path, batchsize, tokenizer,"classify_prompt","classify_reply") for i, device_id in
                             enumerate(devices)])


    # 合并
    df = pd.concat(df_list, ignore_index=True)

    #解析vllm输出结果
    def get_json_value(d,key,z=False):
        #将d中所有中文标点替换为英文标点
        d=d.replace("，",",").replace("。",".").replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'").replace("；",";").replace("：",":")
        d=d.replace("```","").replace("\n","").replace("json","").replace(" ","")

        if z:
            #如果引号不与{}或:相邻，将引号替换为中文引号
            pattern = r'(?<![{:,}])"(?![{:,}])'

            # 使用正则表达式替换匹配的英文引号为中文引号
            d = re.sub(pattern, '“', d)
            d = re.sub(pattern, '”', d)
        # #只提取{}中的内容
        # d=re.search(r"\{.*\}",d).group()
        try:
            return json.loads(d)[key]
        except:
            return None


    df["difficulty"]=df["difficulty_reply"].swifter.apply(lambda x:get_json_value(x,"difficulty"))
    df["score"]=df["score_reply"].swifter.apply(lambda x:get_json_value(x,"score",True))

    df["primary_tag"]=df["classify_reply"].swifter.apply(lambda x:get_json_value(x,"primary_tag"))
    df["other_tags"]=df["classify_reply"].swifter.apply(lambda x:get_json_value(x,"other_tags"))
    df['intent'] = df["difficulty_reply"].swifter.apply(lambda x:get_json_value(x,"intent"))
    df['knowledge'] = df["difficulty_reply"].swifter.apply(lambda x:get_json_value(x,"knowledge"))


    df.drop(columns=["difficulty_prompt","score_prompt","classify_prompt"],inplace=True)


    #删除difficulty为None或者score为None或者classify为None的行
    df_f=df[df["difficulty"].notnull() & df["score"].notnull() & df["classify"].notnull()]

    #提取score中的数字
    df_f["score"]=df_f["score"].swifter.apply(lambda x:None if not x else re.search(r"\d",x).group())

    #保存
    df.to_json(final_save_path,orient="records",lines=True)
    df_f.to_json(final_save_path.replace(".json","_filtered.json"),orient="records",lines=True)

