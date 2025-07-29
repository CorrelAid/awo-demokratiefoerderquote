from transformers import (
    logging as tf_logging,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from lib.experiment_config import TRANSLATOR_URL
import asyncio
import aiohttp
from torch.utils import data as t_data
import torch

tf_logging.set_verbosity_error()

_LANG_MAP = {
    2: ["es"],
    3: ["es", "fr"],
    4: ["es", "fr", "en"],
    5: ["es", "fr", "en", "it"],
}
ALL_PIVOTS = sorted({lang for langs in _LANG_MAP.values() for lang in langs})


def load_and_quantize(pivot, src_lang="de", device="cpu"):
    model_to_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot}"
    model_back_name = f"Helsinki-NLP/opus-mt-{pivot}-{src_lang}"

    tok_to = AutoTokenizer.from_pretrained(
        model_to_name, force_download=True, truncation=True
    )
    # huggingface.co/docs/transformers/v4.53.3/en/model_doc/marian#transformers.MarianMTModel
    model_to = AutoModelForSeq2SeqLM.from_pretrained(model_to_name, force_download=True)

    tok_back = AutoTokenizer.from_pretrained(
        model_back_name, force_download=True, truncation=True
    )
    model_back = AutoModelForSeq2SeqLM.from_pretrained(
        model_back_name, force_download=True
    )

    model_to.half()
    model_back.half()

    return {
        "to_tok": tok_to,
        "to": model_to,
        "back_tok": tok_back,
        "back": model_back,
    }


def translate_one_step_batched(data, tokenizer, model, batch_size, max_length, device):
    tokenized_texts = tokenizer(
        data, padding=True, truncation=True, return_tensors="pt"
    )
    tokenized_dataset = t_data.TensorDataset(*(tokenized_texts.values()))
    tokenized_dataloader = t_data.DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    all_translated_ids = []
    with torch.no_grad():
        for batch in tokenized_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch

            translated_ids_batch = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
            )

            all_translated_ids.append(translated_ids_batch.detach().cpu().numpy())

    all_translated_texts = []
    for translated_ids_batch in all_translated_ids:
        translated_texts = tokenizer.batch_decode(
            translated_ids_batch, skip_special_tokens=True
        )
        all_translated_texts.extend(translated_texts)

    return all_translated_texts


def back_translation_augmenter_single_thread(
    quant_pipes,
    texts,
    labels,
    progress=None,
    factor=2,
    do_sample=False,
    top_k=50,
    top_p=0.95,
    batch_size=4,
    device="cpu",
):
    assert factor in _LANG_MAP, "factor must be one of {2,3,4,5}"
    pivots = _LANG_MAP[factor]

    n, k = len(texts), len(pivots)
    total = sum(n * (2**i) for i in range(k))

    if progress:
        task_id = progress.add_task(
            f"Augmenting via back-translation with expansion factor of {factor}…",
            total=total,
        )

    # gen_kwargs = {
    #     "do_sample": do_sample,
    #     "top_k": top_k,
    #     "top_p": top_p,
    #     "num_beams": 1 if do_sample else 2,
    # }

    curr_texts = list(zip(texts, labels))
    for pivot in pivots:
        tok_to = quant_pipes[pivot]["to_tok"]
        model_to = quant_pipes[pivot]["to"]
        tok_back = quant_pipes[pivot]["back_tok"]
        model_back = quant_pipes[pivot]["back"]
        new_texts = []

        batches = [
            curr_texts[i : i + batch_size]
            for i in range(0, len(curr_texts), batch_size)
        ]

        for batch in batches:
            batch_texts = [t for t, _ in batch]
            res = translate_one_step_batched(
                batch_texts,
                tok_to,
                model_to,
                batch_size=batch_size,
                max_length=512,
                device=device,
            )

            pivot_out = [o["translation_text"] for o in res]

            res = translate_one_step_batched(
                pivot_out,
                tok_back,
                model_back,
                batch_size=batch_size,
                max_length=512,
                device=device,
            )
            backs = [o["translation_text"] for o in res]
            backs = zip(backs, [l for _, l in batch])

            if progress:
                progress.update(task_id, advance=len(batch))

            new_texts.extend(backs)

        curr_texts.extend(new_texts)

    return_texts = [t for t, _ in curr_texts]
    return_labels = [tx for _, tx in curr_texts]

    return return_texts, return_labels


async def fetch_augmentation(session, texts, labels, factor):
    async with session.post(
        TRANSLATOR_URL,
        json={"texts": texts, "labels": labels, "factor": factor, "batch_size": 4},
        headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"},
    ) as response:
        response.raise_for_status()
        result = await response.json()
        return result["augmented_texts"], result["augmented_labels"]


async def remote_back_translation_augmenter(texts, labels, factor=2, concurrency=8):
    chunk_size = len(texts) // concurrency + (1 if len(texts) % concurrency != 0 else 0)
    chunks = [
        (texts[i : i + chunk_size], labels[i : i + chunk_size])
        for i in range(0, len(texts), chunk_size)
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_augmentation(session, chunk_texts, chunk_labels, factor)
            for chunk_texts, chunk_labels in chunks
        ]
        results = await asyncio.gather(*tasks)

    augmented_texts = []
    augmented_labels = []
    for augmented_batch_texts, augmented_batch_labels in results:
        augmented_texts.extend(augmented_batch_texts)
        augmented_labels.extend(augmented_batch_labels)

    return augmented_texts, augmented_labels


async def run_back_translation_augmentation(texts, labels, factor=2):
    return await remote_back_translation_augmenter(texts, labels, factor)
