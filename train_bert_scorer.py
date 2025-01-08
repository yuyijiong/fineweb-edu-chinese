# Encoding: UTF-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import transformers
import re
import torch
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import TrainingArguments, Trainer
#import torch.distributed as dist
from datasets import Dataset, concatenate_datasets

from transformers.data.data_collator import *

@dataclass
class DataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        #将labels转化为float
        features["labels"] =features["labels"].float()/5

        return features

#清洗文本，通用
def clean_text_uni_simple(content:str):

    #删除 \□\■
    content = content.replace(' ', '').replace('□', '').replace('■', '')
    #替换长空格为短空格
    content = content.replace(' ', ' ').replace('\u3000', ' ')
    #如果出现两个以上换行符，则最多保留2个
    content=re.sub('[\n\r]{2,}','\n\n',content)
    #如果出现两个以上空格，则最多保留2个
    content=re.sub(' {2,}','  ',content)
    # # -连续出现大于3次，则替换为---
    content = re.sub(r'-{3,}', '---', content)

    return content.strip()

def load_parquet_datasets_for_cls(json_dir, select_num=None, clean_text=False, lang=None):
    #判断是dir还是path
    if os.path.isdir(json_dir):
        path_list = get_file_paths(json_dir, '.parquet')
    else:
        path_list=[json_dir]
    if lang=="en":
        #只保留英文文件
        path_list=[path for path in path_list if '_en' in path]
    elif lang=="zh":
        #只保留中文文件
        path_list=[path for path in path_list if '_zh' in path]

    print('path_list:', path_list)
    dataset_list = []
    for path in tqdm(path_list, desc='加载数据集', total=len(path_list)):
        dataset = Dataset.from_parquet(path)
        #如果有content列，重命名为text
        if 'content' in dataset.column_names:
            dataset = dataset.rename_column('content', 'text')
        #如果有score或scores列，重命名为labels
        if 'score' in dataset.column_names:
            dataset = dataset.rename_column('score', 'labels')
        if 'scores' in dataset.column_names:
            dataset = dataset.rename_column('scores', 'labels')

        #只保留text和labels列
        dataset = dataset.select_columns(['text', 'labels'])

        dataset_list.append(dataset)

    # 合并所有数据集
    concatenated_dataset = concatenate_datasets(dataset_list)
    # 如果select_num不为空，则只选择前select_num个样本
    if select_num is not None:
        concatenated_dataset = concatenated_dataset.select(range(select_num))
    # 删除为None的样本
    concatenated_dataset = concatenated_dataset.filter(lambda x: x['text'] is not None, desc='删除为None的样本',num_proc=16)
    # 删除labels不能转化为float的样本
    def to_float(x):
        try:
            float(x)
            return True
        except:
            return False
    concatenated_dataset = concatenated_dataset.filter(lambda x: to_float(x['labels']), desc='删除labels不能转化为float的样本',num_proc=16)

    # #按text去重
    # concatenated_dataset = concatenated_dataset.d

    # 打印训练集和验证集长度
    print('训练集样本数', len(concatenated_dataset))
    # 获取数据集每个样本的text的平均长度
    len_list = list(map(len, concatenated_dataset['text']))
    print('数据集每个样本的text的平均长度', sum(len_list) / len(len_list))
    print('数据集的text的最大长度', max(len_list))
    print('数据集的text的最小长度', min(len_list))

    return concatenated_dataset

#PeftModel.save_pretrained()

def get_lm_dataset(raw_dataset,tokenizer,model_max_length,training_args):

    def tokenize_function_cls(examples, tokenizer, max_length, add_special_tokens=True, truncation=True, padding=True):

        text=examples['text']
        input_ids=tokenizer(text, max_length=max_length, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding)['input_ids']
        try:
            labels=[float(examples['labels'])]
        except:
            labels=None
        return {'input_ids':input_ids,'labels':labels}


    with training_args.main_process_first(desc="tokenizing"):
        # 将数据的每个sample的text转化为token，不进行padding，不需要加上bos和eos，因为数据中已经有了
        lm_dataset = raw_dataset.map(
            lambda x: tokenize_function_cls(x, tokenizer=tokenizer,
                                                 max_length=model_max_length,
                                                 add_special_tokens=False,
                                                 truncation=True,
                                                 padding=False),
            remove_columns=raw_dataset.column_names,
            num_proc=16,
            desc="Running tokenizer on dataset", )


    #删除labels为None的样本
    lm_dataset=lm_dataset.filter(lambda x: x['labels'] is not None,num_proc=16)
    #打乱训练集
    lm_dataset=lm_dataset.shuffle(seed=1)
    # 预训练样本数
    print('预训练样本数', len(lm_dataset))

    return lm_dataset

def main():
    num_gpus = int(torch.cuda.device_count())
    output_dir="//data/models/bge-reranker-base-edu-cls-qwen14b"#
    model_name ="//data/models/bge-reranker-base"#"

    print(output_dir)

    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        eval_steps=1,
        report_to="tensorboard",
        logging_strategy='steps',
        logging_steps=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        save_strategy='steps',
        save_steps=1000,
        num_train_epochs=3,
        remove_unused_columns=False,
        ignore_data_skip=True,
        save_only_model=True,

        optim="adamw_torch",  #'paged_adamw_8bit',#
        weight_decay=0,

        lr_scheduler_type="cosine",#"constant_with_warmup",#  #
        warmup_ratio=0.05,
        learning_rate=1e-5,
        per_device_train_batch_size=100,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,


        fp16=False,
        bf16=True,
        #deepspeed='ds_config_constant.json',
        auto_find_batch_size=False,
        load_best_model_at_end=False,
        torch_compile_backend="inductor",

        #neftune_noise_alpha=3,
        seed=1,
    )


    model_max_length = 512
    # 准备tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           #use_fast=True,
                                                           padding_side='right',
                                                           add_bos_token=False,
                                                           add_eos_token=False,
                                                           #legacy=False,
                                                           model_max_length=model_max_length,
                                                           trust_remote_code=True,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取预训练数据集
    data_dir= "//data/datasets/教育打分数据集-qwen14b"
    raw_dataset = load_parquet_datasets_for_cls(data_dir, select_num=None, clean_text=True)

    # 准备lm_dataset
    lm_dataset = get_lm_dataset(raw_dataset, tokenizer, model_max_length, training_args)


    # 准备模型

    # 加载模型
    from transformers import AutoModelForSequenceClassification
    model=AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype='auto',num_labels=1,problem_type = "regression",trust_remote_code=True)
    #model.to(torch.bfloat16)
    #准备lora模型
    #model = get_lora_model(model,quantized=True)
    model.requires_grad_(True)

    # 准备训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                             padding=True,
                                             return_tensors="pt",
                                             pad_to_multiple_of=None),
    )
    model.config.use_cache = False
    #data_loader = trainer.get_train_dataloader()
    trainer.train()

    try:
        rank = torch.distributed.get_rank()
        if rank == 0:
            # 保存base模型的config
            model.base_model.model.config.save_pretrained(training_args.output_dir)
            #如果已经量化，则不能merge。只能保存lora参数
            trainer.save_model(output_dir=training_args.output_dir)
    except:
        # 保存base模型的config
        model.base_model.model.config.save_pretrained(training_args.output_dir)

        #保存lora参数
        trainer.save_model(output_dir=training_args.output_dir)

    print('model saved:', training_args.output_dir)


if __name__ == '__main__':
    main()


