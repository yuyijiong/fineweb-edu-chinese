import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from api回复 import model_answer
import json
from tqdm import tqdm
from multiprocessing import Pool
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset
from typing import List, Dict, Any, Optional, Union
from pydantic import Field
from huggingface_hub import Repository
from datasets import search



class FaissNearestNeighbour:
    device: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="The CUDA device ID or a list of IDs to be used. If negative integer,"
        " it will use all the available GPUs.",
    )
    string_factory: Optional[str] = Field(
        default=None,
        description="The name of the factory to be used to build the `faiss` index."
        "Available string factories can be checked here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes.",
    )
    metric_type: Optional[int] = Field(
        default=None,
        description="The metric to be used to measure the distance between the points. It's"
        " an integer and the recommend way to pass it is importing `faiss` and thenpassing"
        " one of `faiss.METRIC_x` variables.",
    )
    k: Optional[int] = Field(
        default=1,
        description="The number of nearest neighbours to search for each input row.",
    )
    search_batch_size: Optional[int] = Field(
        default=50,
        description="The number of rows to include in a search batch. The value can be adjusted"
        " to maximize the resources usage or to avoid OOM issues.",
    )
    train_size: Optional[int] = Field(
        default=None,
        description="If the index needs a training step, specifies how many vectors will be used to train the index.",
    )
    def _build_index(self, inputs: List[Dict[str, Any]]) -> Dataset:
        """Builds a `faiss` index using `datasets` integration.

        Args:
            inputs: a list of dictionaries.

        Returns:
            The build `datasets.Dataset` with its `faiss` index.
        """
        dataset = Dataset.from_pandas(inputs)

        dataset.add_faiss_index(
            column="embedding",
            device=self.device,  # type: ignore
            string_factory=self.string_factory,
            metric_type=self.metric_type,
            train_size=self.train_size,
        )
        return dataset

    def _save_index(self, dataset: Dataset) -> None:
        """Save the generated Faiss index as an artifact of the step.

        Args:
            dataset: the dataset with the `faiss` index built.
        """
        self.save_artifact(
            name="faiss_index",
            write_function=lambda path: dataset.save_faiss_index(
                index_name="embedding", file=path / "index.faiss"
            ),
            metadata={
                "num_rows": len(dataset),
                "embedding_dim": len(dataset[0]["embedding"]),
            },
        )

    def _search(self, dataset: Dataset) -> Dataset:
        """Search the top `k` nearest neighbours for each row in the dataset.

        Args:
            dataset: the dataset with the `faiss` index built.

        Returns:
            The updated dataset containing the top `k` nearest neighbours for each row,
            as well as the score or distance.
        """

        def add_search_results(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            queries = np.array(examples["embedding"])
            results = dataset.search_batch(
                index_name="embedding",
                queries=queries,
                k=self.k + 1,  # type: ignore
            )
            examples["nn_indices"] = [indices[1:] for indices in results.total_indices]
            examples["nn_scores"] = [scores[1:] for scores in results.total_scores]
            return examples

        return dataset.map(
            add_search_results, batched=True, batch_size=self.search_batch_size,desc="Searching nearest neighbours"
        )

    def process(self, inputs) -> "StepOutput":  # type: ignore
        dataset = self._build_index(inputs)
        dataset_with_search_results = self._search(dataset)
        #self._save_index(dataset)
        return dataset_with_search_results.to_pandas()

if __name__ == '__main__':
    import faiss
    #apply显示进度条
    tqdm.pandas()


    df=pd.read_json("只包含用户问题的对话_embed.json",lines=True)
    print("样本数量：",len(df))

    #将embedding转化为np
    df["embedding"]=df["embedding"].progress_apply(np.array)

    bert_model=SentenceTransformer("/data/models/gte-large-zh",device="cuda")


    first_prompt_list=df["first_prompt"].tolist()
    embeddings = bert_model.encode(first_prompt_list,show_progress_bar=True,batch_size=128)
    df["embedding"]=embeddings.tolist()

    df.to_json("只包含用户问题的对话_embed.json",orient="records",lines=True)


    #计算最近邻
    nearest_neighbour=FaissNearestNeighbour()
    nearest_neighbour.device=None
    nearest_neighbour.metric_type=faiss.METRIC_INNER_PRODUCT
    nearest_neighbour.k=5
    nearest_neighbour.string_factory="Flat"
    nearest_neighbour.train_size=None
    nearest_neighbour.search_batch_size=100

    df=nearest_neighbour.process(df)

    df.to_json("只包含用户问题的对话_最近邻.json",orient="records",lines=True)


