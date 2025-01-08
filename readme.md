此repo包含 fineweb-edu-chinese, cosmopedia-chinese, smoltalk-chinese 三个数据集的构建代码

[数据集下载](https://huggingface.co/collections/opencsg/high-quality-chinese-training-datasets-66cfed105f502ece8f29643e)


## Fineweb-edu-chinese

1. ``annotate_edu_score.py``: 使用 Qwen2.5-14b 模型，对一部分文本进行教育价值评估，给出0-5分的打分结果

2. ``train_bert_scorer.py``: 使用上一步中标注的数据集，训练一个基于bert的模型，用于文本教育价值打分

3. ``score_each_file.py``: 对语料库中每个数据集文件中的文本进行打分，并保存到原路径

4. ``filter_high_score.py``: 对打过分的数据，筛选出分数大于3分的部分

5. ``bloom_filter.py``: 使用 bloom 过滤器对数据集进行去重

## Cosmopedia-chinese

``make_cosmopedia.py``: 读取种子数据，将每一条种子数据作为topic，由prompt模板指定风格和体裁，然后交给 vllm 中的模型，生成指定风格的合成数据。

## Smoltalk-chinese

1. ``pipeline_magpie_zh.py``: 使用 distilabel 库中的 Magpie 算法，使用 deepseek-v2.5 或 qwen2.5 模型，根据人工设计的对应多种任务的 system_prompt，生成多轮对话数据

2. ``gather_generated_data.py``: 聚合上一步中生成的数据，初步清洗，保存为一个文件

3. ``score_and_classify.py``: 使用 Qwen2.5-7b-instruct 模型，给每条对话数据的第一个 user_query 根据其清晰度、流畅性进行打分，只保留3分以上数据

4. ``encode_and_get_nearest.py``: 使用 gte-large-zh 模型对每条对话数据的第一个 user_query 编码为嵌入向量，并按照余弦相似度查找距离最近的样本

5. ``dedup_and_save.py``: 根据上一步的查找结果，去重（每个相似集合只随机保留一条）。最后按任务类型保存为不同文件





