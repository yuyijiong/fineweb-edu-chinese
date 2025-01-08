#encoding=utf-8
from typing import Callable

import click
import numpy as np
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import BloomFilterArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_128_digest
#from text_dedup.utils.load import load_hf_dataset
from text_dedup.utils.memory import DisableReferenceCount
from text_dedup.utils.timer import Timer
import ray
from read_parquet import load_hf_dataset
import setproctitle
setproctitle.setproctitle("text_dedup_bloom")


@click.command
@IOArgs.option_group
@MetaArgs.option_group
@BloomFilterArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    bloom_filter_args: BloomFilterArgs,
):
    timer = Timer()
    flags = []

    hash_func: Callable = {
        "md5": md5_digest,  # type: ignore
        "sha256": sha256_digest,  # type: ignore
        "xxh3": xxh3_128_digest,  # type: ignore
    }[bloom_filter_args.hash_func]

    bf = ScalableBloomFilter(
        initial_capacity=bloom_filter_args.initial_capacity,
        mode=ScalableBloomFilter.SMALL_SET_GROWTH,
        error_rate=bloom_filter_args.error_rate,
    )

    with timer("Total"):
        with timer("Loading"):
            ds, _ = load_hf_dataset(data_dir_list=data_dir_list)
            #ds.save_to_disk(io_args.output+"_before_dedup")
        # 关闭Ray
        ray.shutdown()

        LEN_DATASET = len(ds)
        NUM_SHARDS = int(np.ceil(LEN_DATASET / meta_args.batch_size))

        with timer("Processing"):
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                # TODO .map(either preprocessing like example.encode("utf-8") or multithreaded)
                ds_shard = ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                for example in tqdm(ds_shard[meta_args.column], leave=False):
                    h = hash_func(example.encode("utf-8"))
                    # True if the element is seen, False otherwise
                    flags.append(bf.add(h))


        with timer("Filtering"), DisableReferenceCount():
            #put flags into dataset
            ds=ds.add_column("flags", flags)
            #filter out the duplicated examples
            ds=ds.filter(lambda x: not x["flags"], num_proc=io_args.num_proc, desc="Filtering...")


            # #filter out the duplicated examples, by each shard
            # ds_shards=[ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True) for idx in range(NUM_SHARDS)]
            # ds_shards=[ds_shard.filter(lambda x: not x["flags"], num_proc=io_args.num_proc, desc="Filtering...") for ds_shard in ds_shards]
            # #concatenate the shards
            # from datasets import concatenate_datasets
            # ds=concatenate_datasets(ds_shards)

            #flags = np.array(flags)
            # ds = ds.filter(
            #     lambda _, idx: not flags[idx],
            #     with_indices=True,
            #     num_proc=io_args.num_proc,
            #     desc="Filtering...",
            # )

        with timer("Saving"):
            ds.shuffle()
            ds.save_to_disk(io_args.output,num_proc=32)

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    # data_dir_list=["/data/dataset/BAAI/CCI3-Data/data_edu_scored_3",
    #                "/data/dataset/BAAI/IndustryCorpus2_scored_3",
    #                "//data/dataset/OpenDataLab___MiChao/raw_edu_scored_2_edu_scored_3",
    #                "//data/dataset/wudao/WuDaoCorpus2.0_base_200G_edu_scored_4",
    #
    #                "/share/datasets/chinese_fineweb_each_dataset/tele/data_edu_scored_edu_scored_3",
    #                "/share/datasets/chinese_fineweb_each_dataset/skypile/data_edu_scored_edu_scored_3"
    #                ]
    #

    data_dir_list=[
        "//backup/dataset/BAAI/CCI3-Data/data_edu_scored",

       "//backup/dataset/BAAI/IndustryCorpus2-edu-scored",

       "//data/dataset/OpenDataLab___MiChao/raw_edu_scored",

       "/backup/dataset/nlp/CN",

       "/data/dataset/CASIA-LM/ChineseWebText/cleaner_subset",

       "//backup/dataset/wudao/WuDaoCorpus2.0_base_200G_edu_scored_edu_scored",


       "/data/datasets/chinese-fineweb-edu-v2-unicom/tele/data_edu_scored_edu_scored",

       "//data/datasets/chinese-fineweb-edu-v2-unicom/skypile/data_edu_scored_edu_scored",

        "/data/datasets/chinese-fineweb-edu-v2-unicom/map-cc/cc",

        "/data/datasets/opencsg-cc-scored",

    ]

    main()