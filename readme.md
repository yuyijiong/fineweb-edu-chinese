## Fineweb-edu-chinese

``train_bert_scorer.py``: 训练一个基于bert的模型，用于文本打分

``score_each_file.py``: 对语料库中每个数据集文件中的文本进行打分，并保存到原路径

## Cosmopedia-chinese

``make_cosmopedia.py``: 将每一条种子数据套进prompt模板，交给vllm中的模型，生成新的数据

## Deduplication

``/dedup/bloom_filter.py``: 使用 bloom 过滤器对数据集进行去重


