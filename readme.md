## Fineweb-edu-chinese

``annotate_edu_score.py``: 使用 Qwen2.5-14b 模型，对一部分文本进行教育价值评估，给出0-5分的打分结果

``train_bert_scorer.py``: 使用上一步中标注的数据集，训练一个基于bert的模型，用于文本教育价值打分

``score_each_file.py``: 对语料库中每个数据集文件中的文本进行打分，并保存到原路径

``filter_high_score.py``: 对打过分的数据，筛选出分数大于3分的部分

## Cosmopedia-chinese

``make_cosmopedia.py``: 将每一条种子数据套进prompt模板，交给vllm中的模型，生成新的数据

## Deduplication

``/dedup/bloom_filter.py``: 使用 bloom 过滤器对数据集进行去重


