import multiprocessing
import os
import pathlib
import re
from multiprocessing import Pool
import pandas as pd
# import zhconv
# from openai import OpenAI
from tqdm import tqdm

multiprocessing.set_start_method('spawn',True)

def vllm_one_df(df,device:str,model_path,batchsize,tokenizer):
    os.environ["CUDA_VISIBLE_DEVICES"]=device
    from vllm import LLM, SamplingParams
    #from vllm.transformers_utils.config import
    import torch
    web_text_max_len = 800
    sampling_params = SamplingParams(temperature=0.95, top_p=0.95,max_tokens=4000)
    llm=LLM(model=model_path,tensor_parallel_size=1,dtype="auto",trust_remote_code=True, enforce_eager=False,quantization=None)
    for i in tqdm(range(0,len(df),batchsize),desc="处理数据,bs={}".format(batchsize),mininterval=5):
        if "title" in df.columns:
            titles=df.loc[i:min(i+batchsize-1,len(df)),"title"].tolist()
            texts=df.loc[i:min(i+batchsize-1,len(df)),"text"].tolist()
            #将title和text拼接
            web_text=[title+"\n"+text for title,text in zip(titles,texts)]
        else:
            web_text=df.loc[i:min(i+batchsize-1,len(df)),"text"].tolist()
        #如果超过100字，截断，末尾加上......
        web_text=[text[:web_text_max_len]+"......" if len(text)>web_text_max_len else text for text in web_text]

        answers=eval_edu(web_text,llm,sampling_params,tokenizer)

        df.loc[i:min(i+batchsize-1,len(df)),"data"]=answers
        # rank=dist.get_rank()
        if i==0:
            print("\n\nOutput:",answers[0])
    del llm
    torch.cuda.empty_cache()
    return df

def answer_vllm(prompt,llm,sampling_params):
    if isinstance(prompt,str):
        outputs = llm.generate([prompt], sampling_params,use_tqdm=False)

        return outputs[0].outputs[0].text
    else:
        outputs = llm.generate(prompt, sampling_params,use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

def eval_edu(web_text,llm,sampling_params,tokenizer):
    prompt_college='''这是一段来自网页的摘录：“{}”。
请编写一个针对大学生的足够详细的教科书课程单元，该单元与给定的摘录中的某个概念或多个概念相关。
不需要包含摘录中的所有内容，只需要发掘其中适合作为教科书内容的部分。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度。
要求：1. 严谨性：确保对概念/章节的深入覆盖。
2. 吸引性：用学术、专业且引人入胜的语气撰写，以吸引兴趣。
3. 应用：融入具体的实践例子，例如微积分中要给出公式、严格证明，历史中要给出关键日期和人物，计算机操作中要给出代码。
4.不需要给出参考文献。内容中不应包含广告或涉及隐私的信息。注重主体内容，不需要其它格式化的内容。
请记住，要针对大学生制作内容，他们可能拥有一些基础知识，但不是该领域的专家。内容应该详细且发人深省。
请立即开始撰写教科书，不要使用图片，不要输出除了教科书以外的内容，不要以“课程单元”作为标题而是要有具体的标题。
'''
    prompt_college_textbook='''为面向大学生的教科书编写一个足够长而非常详细的课程单元，主题是“矩阵的秩”。
满足以下要求：1.严谨 - 创建具有挑战性的教科书，深入涵盖材料。
2. 引人入胜 - 教科书具有引人入胜的叙事方式。
3. 具体：应该有详细的名词解释和实际应用例子，以便学生能够理解和应用所学知识。
请记住，要针对大学生制作内容，他们可能拥有一些基础知识，但不是该领域的专家。材料应该详细且发人深省。'''

    prompt_high_textbook = '''这是一段来自网页的摘录：“{}”。
请编写一段针对中学生的教科书单元内容，该单元与此网页摘录中的某个概念或多个概念相关，具有教育意义。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度。
要求：1. 严谨性：确保对概念/章节的深入覆盖。
2. 吸引性：用学术、专业且引人入胜的语气撰写，以吸引兴趣。
3. 应用：融入具体的实践例子，例如微积分中要给出严格证明，历史中要给出关键日期和人物。
请立即开始撰写教科书，不要使用图片，不要输出除了教科书以外的内容。
    '''

    prompt_story = '''写一个与以下文本片段相关的引人入胜的故事：“{}”。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意！可以加入其它知识。
故事应包括： 1.小众概念或兴趣：深入研究特定的概念、爱好、兴趣或幽默情况 
2.意想不到的情节转折或引人入胜的冲突，引入具有挑战性的情况或困境。 
3.对话：故事必须至少包含一个有意义的对话，以揭示人物深度、推进情节或揭开谜团的关键部分
4.反思和洞察：以具有教育意义的新理解、启示的结论结束。 
5.故事中的人物应使用中国式的名字。请勿包含广告或涉及隐私的信息。
请马上开始讲故事，不要输出除了故事以外的内容。'''

    prompt_self_story = '''写一个与以下文本片段相关的引人入胜的故事：“{}”。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意！可以加入其它知识。故事需要以第一人称叙述，模仿一个人在知乎上分享自己的故事。
故事应包括：
1.意想不到的情节转折或引人入胜的冲突。 
2.反思和洞察：以具有教育意义的新理解、启示的结论结束。 
3.请勿包含广告或涉及隐私的信息。
请马上开始讲故事，不要输出除了故事以外的内容。'''

    prompt_baby='''网页摘录：“{}”
创建一个与上述网页摘录中的某个概念相关的具有教育意义的儿童故事，重点针对对世界和人际交往零知识的5岁儿童。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意。
故事应该使用简单的术语。你可以补充额外的知识来帮助理解。
使用易于理解的示例，并将 5 岁儿童可能提出的问题及其答案纳入故事中。故事应涵盖日常行为和常见物品的使用。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是幼儿能理解的内容。如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
请直接开始撰写故事，不要输出除了故事以外的内容。'''

    prompt_middle_edu='''网页摘录：“{}”。
创建一个与上述网页摘录中的某个概念相关的具有教育意义的内容，针对中学生，尽量长而详细。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度，不需要包含摘录中的所有内容。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是中学的内容。
如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。
例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
避免使用技术术语和LaTeX，只讨论中学级别的主题。内容中不应包含广告或涉及隐私的信息。
请直接开始撰写教育内容，不要输出除了教育内容以外的内容。'''

    prompt_wikihow='''网页摘录：“{}”。
以 WikiHow 的风格写一篇长而非常详细的教程，教程与此网页摘录有相关性。
教程中需要包括对每个步骤的深入解释以及它如何帮助实现预期结果。你可以自由补充其他相关知识。
确保清晰性和实用性，让读者能够轻松遵循教程完成任务。内容中不应包含广告或涉及隐私的信息。
不要使用图像。请直接开始撰写教程。
'''
    prompt_chosen=prompt_wikihow

    if isinstance(web_text,str): 
        prompt=prompt_chosen.format(web_text)
        prompt=apply_template(prompt,tokenizer)
    else:
        prompt=[prompt_chosen.format(text) for text in web_text]
        #为每个prompt增加chat模板
        prompt=[apply_template(p,tokenizer) for p in prompt]

    answer=answer_vllm(prompt,llm,sampling_params)

    #每个answer中删除多余的 
    answer=[re.sub(r" ","",ans) for ans in answer]

    return answer

def apply_template(prompt,tokenizer):
    return tokenizer.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False,max_length=1000000)


if __name__ == '__main__':

    model_path ="//data/models/LongWriter-glm4-9b"#

    batchsize=64

    devices=[0,1,2,3,4,5,6,7]
    num_device=len(devices)

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path)

    df_dir="//data/dataset/csdn/split_from_batch1/"#"./csdn/csdn"#"baidubaike_563w"
    save_dir="csdn_200w_glm合成_wikihow"#"baidubaike_563w_longwriter_glm合成_wikihow"
    #获取dir下所有parquet文件或jsonl文件
    file_list=[file for file in os.listdir(df_dir) if file.endswith(".parquet") or file.endswith(".jsonl")]
    file_list.sort(reverse=True)
    file_list=file_list[:]
    print("文件数量：",len(file_list))
    for file in file_list:
        df_path=os.path.join(df_dir,file)
        save_path=os.path.join(save_dir,file.replace(".jsonl",".parquet"))
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True,exist_ok=True)
        print("开始处理文件：",file)
        #如果已经存在，跳过
        if os.path.exists(save_path):
            print("文件已存在，跳过")
            continue

        df = pd.read_parquet(df_path) if df_path.endswith(".parquet") else pd.read_json(df_path,lines=True)

        # #删除标题为 **年 结尾的词条
        # df=df[~df["title"].str.contains("\d+[年月]$")]
        #
        # #删除标题同时包含大写字母和中文的词条
        # df=df[~df["title"].str.contains("[A-Za-z]")&df["title"].str.contains("[\u4e00-\u9fa5]")]

        # #如果tags里面包含“小说”或“人物“，删除
        # df['tags']=df['tags'].apply(lambda x:list(x))
        # df=df[~df['tags'].apply(lambda x:"小说" in x or "人物" in x)]
        # #如果tags中的某个tag包含“游戏”，删除
        # df=df[~df['tags'].apply(lambda x:any(["游戏" in tag for tag in x]))]


        #如果有content列，重命名为text
        if 'content' in df.columns:
            df.rename(columns={'content':'text'},inplace=True)
        if 'md' in df.columns:
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
        df.to_parquet(save_path)
