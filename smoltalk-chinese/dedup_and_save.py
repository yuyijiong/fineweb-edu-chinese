import pandas as pd
import os
import random
from tqdm import tqdm

def deduplicate(df, threshold=0.95):
    import networkx as nx

    # 创建一个空图
    G = nx.Graph()

    # 遍历每一行数据
    for index, row in df.iterrows():
        # 获取当前行的相似数据行号和相似度
        indices = row['nn_indices'][0]
        scores = row['nn_scores'][0]

        # 确保indices和scores是列表
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(scores, list):
            scores = [scores]

        # 遍历相似数据对
        for i, score in zip(indices, scores):
            if score > threshold:
                # 添加边到图中
                G.add_edge(index, i)

    # 找到所有的连通分量
    connected_components = list(nx.connected_components(G))

    # 选择要保留的行
    # 创建一个集合来存储要保留的行号
    to_keep = set()

    for component in connected_components:
        # 选择行号最小的行
        min_index = min(component)
        to_keep.add(min_index)

    # 找出没有被任何连通分量包含的行
    # 这些行本身不与任何其他行相似，直接保留
    all_indices = set(df.index)
    connected_indices = set().union(*connected_components)
    unconnected_indices = all_indices - connected_indices
    to_keep.update(unconnected_indices)

    # 保留这些行
    result_df = df.loc[list(to_keep)]

    # 重置索引
    result_df = result_df.reset_index(drop=True)

    print("去重前：",len(df),"去重后：",len(result_df))

    return result_df

if __name__ == '__main__':
    df_all=[]
    df_dir="./不同类别去重/"
    df_list=[df_dir+file for file in os.listdir(df_dir) if file.endswith(".json") and 'everyday' in file]
    df_list=[pd.read_json(file,lines=True) for file in tqdm(df_list)]
    for file in df_list:
        df=file
        print("number of data",len(df))
        print('system_prompt_key:',df['system_prompt_key'].unique())

        #去重
        n_turn=len(df["conversation"].iloc[0])/2
        print("n_turn:",n_turn)
        df["n_turn"]=n_turn
        #如果n_turn为1，说明只有一轮对话，去重阈值为0.6
        threshold=0.7 if n_turn==1 else 0.85
        if "document" in df.columns:
            threshold=0.6
        df=deduplicate(df,threshold)

        #只保留3分及以上的数据
        df=df[df["score"]>=3]
        #
        #
        # #统计difficulty数据分布
        # df["difficulty"]=df["difficulty"].apply(lambda x:x.replace("[","").replace("]","").replace("'","").replace(",","").strip())
        # difficulty_count=df["difficulty"].value_counts()
        # print(difficulty_count)
        #
        # #统计score数据分布
        # score_count=df["score"].value_counts()
        # print(score_count)
        #
        # #统计classify数据分布
        # legal_classify=["信息寻求","推理","规划","编辑","代码与调试","数学","角色扮演","数据分析","创意写作","寻求建议","头脑风暴","格式限制","文档问答","总结","其他"]
        # #如果有不在legal_classify中的数据，将其改为其他
        # df["classify"]=df["classify"].apply(lambda x:x if x in legal_classify else "其他")
        # classify_count=df["classify"].value_counts()
        # print(classify_count)

        df_all.append(df)

    df_all=pd.concat(df_all,ignore_index=True)
    df_all.drop(columns=["embedding","nn_indices","nn_scores"],inplace=True)
    #统计system_prompt_key数据分布
    system_prompt_key_count=df_all["system_prompt_key"].value_counts()
    print(system_prompt_key_count)
    print("number of data",len(df_all))

    save_dir="./chinese_magpie/"
    import pathlib
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    #按system_prompt_key分组
    df_grouped=df_all.groupby("system_prompt_key")

    for name,group in df_grouped:
        df_path_save=os.path.join(save_dir,name+".parquet")
        print("保存路径：",df_path_save)
        group.to_parquet(df_path_save,index=False)