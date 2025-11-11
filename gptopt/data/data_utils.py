import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from functools import partial



def get_data_dir(dataset_name):
    if dataset_name == 'slim_pajama1B':
        return "/mnt/ceph/users/mcrawshaw/huggingface"
    elif dataset_name == 'slim_pajama10B':
        return "/mnt/ceph/users/nghosh/huggingface"
    else:
        return "/mnt/ceph/users/cmodi/huggingface"

        
magic_number = 20250401         # used in the header of saved binary files

def tokenize(doc, enc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = magic_number # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def process_and_save_docs(
    dataset,
    filepath,
    encoding,
    shard_size=int(1e8),
    nprocs=1,
    target_tokens=None,
    max_shards=None,
):
    """
    Stream-tokenize `dataset` and write fixed-size shards of uint16 token IDs.

    Shard naming:
      - First shard:  val_000000.bin
      - Subsequent:  train_000001.bin, train_000002.bin, ...

    Early-stop policy:
      - If `max_shards` is provided and `target_tokens` is None, we compute
        `target_tokens = shard_size * max_shards`.
      - If `target_tokens` is provided, we stop as soon as we have written
        that many tokens in total (including the current partial shard).

    Notes:
      - Documents may span shards (they are sliced as needed).
      - Progress bar is per-shard and resets after each shard is written.
      - No shuffling is performed (order is whatever the dataset yields).
    """

    # Decide worker count
    if nprocs == 0:
        nprocs = max(1, os.cpu_count() - 2)
    print("Number of processes used  : ", nprocs)

    # Derive a token budget from max_shards if needed
    if target_tokens is None and max_shards is not None:
        target_tokens = shard_size * max_shards

    # Prepare tokenizer for the pool
    tokenizer = partial(tokenize, enc=encoding)

    with mp.Pool(nprocs) as pool:
        # ---- per-run state (kept to your original variable names) ----
        shard_index = 0                          # which shard we are filling
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # shard buffer
        token_count = 0                          # tokens currently in buffer
        progress_bar = None                      # tqdm progress for this shard
        total_tokens_written = 0                 # total tokens placed (disk + buffer)
        stop = False                             # request to stop after hitting cap

        # Iterate tokenized examples from the pool
        for tokens in pool.imap(tokenizer, dataset, chunksize=16):
            if stop:
                break

            # Pack this document into one or more shards
            start = 0
            while start < len(tokens):
                # Lazily create the progress bar for the current shard
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

                # Capacity left in current shard, and tokens left in this document
                capacity = shard_size - token_count
                remaining_in_doc = len(tokens) - start

                # If we have a token budget, respect it
                if target_tokens is not None:
                    allowed = target_tokens - total_tokens_written
                    if allowed <= 0:
                        # No budget left; stop packing more tokens
                        stop = True
                        break
                    take = min(capacity, remaining_in_doc, allowed)
                else:
                    take = min(capacity, remaining_in_doc)

                # Copy a slice of the document's tokens into the shard buffer
                all_tokens_np[token_count: token_count + take] = tokens[start: start + take]
                token_count += take
                start += take
                total_tokens_written += take
                progress_bar.update(take)

                # If the shard just filled exactly, write it and start a new one
                if token_count == shard_size:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(filepath, f"{split}_{shard_index:06d}.bin")
                    write_datafile(filename, all_tokens_np)   # full shard
                    shard_index += 1
                    token_count = 0
                    # reset the progress bar for the next shard
                    progress_bar.close()
                    progress_bar = None

                # If we reached the budget mid-shard, mark to stop;
                # the final (partial) flush is handled once at the end.
                if target_tokens is not None and total_tokens_written >= target_tokens:
                    stop = True
                    break

            if stop:
                break

        # ---- single final flush ----
        # Writes any trailing partial shard (covers natural end or early stop).
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(filepath, f"{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

        # Ensure progress bar is closed if it exists
        if progress_bar is not None:
            progress_bar.close()
