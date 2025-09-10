"cluster liked videos"

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score

import weftly_helper as wh


def test_cluster(selected_model, output_file):
    """create cluster

    Parameters
    ----------
    output_file
        output file name
    """
    # load all data: config/transcription
    config = wh.read_config()
    # read each transcription file into a message and id list, skips <200 words
    messages, ids, __ = wh.read_all_transcriptions(
        tran_dir=config["transcription_dir"], min_words=200
    )

    # convert messages to vectors using a text embeddings model, has caching
    emb, ids = wh.vectorize_text(selected_model, messages, ids)

    # create clusters
    # reduce embeddings dimensionality
    x50, __ = wh.reduce_dimensionality(emb, n_neighbors=30, random_state=42)
    # clustering algorithm
    clusterer = HDBSCAN(
        min_cluster_size=5,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    cluster_ids = clusterer.fit_predict(x50)
    probs = getattr(clusterer, "probabilities_", np.ones_like(cluster_ids, dtype=float))

    # evaluation metrics
    mask = cluster_ids >= 0
    if mask.sum() > 1 and len(set(cluster_ids[mask])) > 1:
        sil = silhouette_score(x50[mask], cluster_ids[mask], metric="euclidean")
        dbi = davies_bouldin_score(x50[mask], cluster_ids[mask])
        print(f"Silhouette: {sil:.3f}, Davies–Bouldin: {dbi:.3f}")

    # video clusters dictionary
    df = pd.DataFrame(
        {
            "video_id": ids,
            "cluster_id": cluster_ids,
            "membership_prob": probs,
            "transcription": messages,
            "path_to_video": [f'{config["video_dir"]}/{id}.mp4' for id in ids],
        }
    )
    # save clusters
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    test_cluster(selected_model=wh.ModelOptions.OLLAMA_FULL, output_file="./cluster.csv")
