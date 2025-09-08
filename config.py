import json
import os

ROOT_PATH = os.getcwd()

STORAGE_PATH = os.path.join(ROOT_PATH, "storage")

def get_data():
    with open(os.path.join(STORAGE_PATH, "chunks.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    paths, words = [], []
    for key, item in corpus.items():
        words.append(item[0])
        paths.append(item[1])
    return words, paths
