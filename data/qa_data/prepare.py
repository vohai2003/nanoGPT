import os
from datasets import load_dataset, Dataset
import tiktoken
import numpy
from tqdm import tqdm

num_proc = os.cpu_count() // 2
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    rawdataset = load_dataset("yahoo_answers_qa", num_proc=num_proc_load_dataset)
    merged_text = []
    for row in rawdataset["train"]:
        question = row["question"]
        answer = row["answer"]
        merged = "Question: "+question+"\nAnswer: "+answer
        merged_text.append(merged)
    dataset = Dataset.from_dict({"text": merged_text})
    split_dataset = dataset.train_test_split(test_size=0.2, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    def process(example):
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = numpy.sum(dset['len'], dtype=numpy.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = numpy.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = numpy.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = numpy.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()