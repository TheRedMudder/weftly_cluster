"helper functions"

import json
import logging
import pickle
import re
import tomllib
from enum import StrEnum
from pathlib import Path

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from umap import UMAP

FILLERS = [
    "um",
    "uh",
    "erm",
    "hmm",
    "mmm",
    "you know",
    "like",
    "i mean",
    "kind of",
    "sort of",
    "you see",
    "okay",
    "ok",
    "alright",
    "right",
    "so",
    "well",
]


class ModelOptions(StrEnum):
    "Embedding Models"

    OLLAMA_FULL = "bge-m3"  # recommended
    FULL = "BAAI/bge-m3"  # bge-m3 without ollama installed
    SIMPLE = "all-MiniLM-L6-v2"  # worst model, but fastest


def read_config() -> dict:
    """reads config.toml as dictionary

    Returns
    -------
        config.toml content
    """
    with open("config.toml", "rb") as f:
        return tomllib.load(f)


def read_all_transcriptions(
    tran_dir: str, min_words: int = 10
) -> tuple[list, list, list]:
    """read all transcriptions files

    Parameters
    ----------
    tran_dir
        transcription directory
    min_words, optional
        minimum words, transcription must be higher than this number by default 10

    Returns
    -------
    messages
        transcribed messages
    ids
        video ids
    skipped
        skipped video ids
    """
    messages, ids, skipped = [], [], []
    # read all transcriptions files
    transcription_file_paths = Path(tran_dir).resolve().glob("*.json")
    for t_file_path in transcription_file_paths:
        with t_file_path.open() as f:
            transcription = json.load(f)
            # clean data
            full_raw_text = " ".join(text["text"] for text in transcription)
            text = clean_text(full_raw_text)
            # proccess into list
            if len(text) > min_words:
                messages.append(text)
                ids.append(t_file_path.stem)
            else:
                skipped.append(t_file_path.stem)
    if len(skipped) > 0:
        logging.warning("Skipping Videos: %s", skipped)
    return messages, ids, skipped


def clean_text(text: str) -> str:
    """clean text

    Parameters
    ----------
    text
        text

    Returns
    -------
        cleaned text
    """
    # lowercase
    t = text.lower()
    # remove filler words/phrases, punctuation, excess whitespace
    for f in FILLERS:
        # remove fillers
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
        # remove punctuation
        t = re.sub(r"[^\w\s']", " ", t)
        # remove excessive whitespaces
        t = re.sub(r"\s+", " ", t).strip()
    return t


def vectorize_text(
    selected_model: ModelOptions, messages: list, ids: list
) -> tuple[list, list]:
    """vectorize text

    Parameters
    ----------
    selected_model
        selected model
    messages
        transcribed messages
    ids
        video ids

    Returns
    -------
    emb
        embeddings
    ids
        video ids
    """
    # check if cached
    cache_model = f"model_{clean_file_name(str(selected_model))}.pickle"
    cache_ids = f"model_{clean_file_name(str(selected_model))}_ids.pickle"

    if Path(cache_model).exists():
        # load Cache
        emb = read_cache(cache_model)
        # validate cache
        c_ids = read_cache(cache_ids)
        if ids == c_ids:
            # use cache
            logging.warning("Using cache: %s", cache_model)
            return emb, ids
    # perform model (messages->vectors) with/without ollama
    if selected_model == ModelOptions.OLLAMA_FULL:
        # use ollama
        emb = run_ollama_emb(messages, str(selected_model), True)
    else:
        # vectorize without ollama
        model = SentenceTransformer(str(selected_model))
        emb = model.encode(
            messages,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    # save cache
    save_cache(cache_model, emb)
    save_cache(cache_ids, ids, np_check=False)
    return emb, ids


def run_ollama_emb(texts, model_name="bge-m3", batch_size=16, progress=False):
    """run ollama embedding model

    Parameters
    ----------
    texts
        text to vectorize
    model_name, optional
        model name, by default "bge-m3"
    batch_size, optional
        batch size, purely for lowering memory usage, by default 16
    progress
        display progress bar

    Returns
    -------
        embeddings
    """
    vecs = []
    # loop through all messages call ollama embeddings model
    for i in range(0, len(texts), batch_size):
        # ollama embeddings expect one message at a time
        # keep memory usage low via batching/double looping
        batch = texts[i : i + batch_size]
        for t in batch:
            # call model
            r = ollama.embeddings(model=model_name, prompt=t)
            assert len(r["embedding"]) > 0, f"Invalid transcription: {t=}"
            vecs.append(r["embedding"])
        # progress updates
        if progress:
            print(
                f"Embedding Progress: {i}/{len(texts)} - {i/len(texts)*100}%", end="\r"
            )
    return np.array(vecs, dtype=np.float32)


def clean_file_name(text: str) -> str:
    """clean file names

    Parameters
    ----------
    text
        file name

    Returns
    -------
        clean file name
    """
    # removes special characters
    # replaces spaces with underscores
    return clean_text(text).replace(" ", "_")


def save_cache(file_name, data, np_check=True):
    """cache data

    Parameters
    ----------
    file_name
        cache file name
    data
        cache data
    np_check
        use numpy cache validation
    """
    # cache data
    with open(file_name, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # validate cache
    with open(file_name, "rb") as handle:
        cache_games = pickle.load(handle)
    if np_check:
        assert (data == cache_games).all(), "Cache failed"
    else:
        assert data == cache_games, "Cache failed"


def read_cache(file_name):
    """read cache

    Parameters
    ----------
    file_name
        cache file name

    Returns
    -------
        cached data
    """
    with open(file_name, "rb") as handle:
        return pickle.load(handle)


def reduce_dimensionality(embs, n_neighbors, random_state):
    """reduce dimensionality

    Parameters
    ----------
    embs
        embeddings
    n_neighbors
        balances local versus global structure in the data
    random_state
        random seed state

    Returns
    -------
        x50, x2
    """
    reducer50 = UMAP(
        n_neighbors=n_neighbors,
        n_components=50,
        metric="cosine",
        random_state=random_state,
    )
    x50 = reducer50.fit_transform(embs)
    reducer2 = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    x2 = reducer2.fit_transform(embs)
    return x50, x2
