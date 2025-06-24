import json
import os

def load_cache():
    if os.path.exists("cache.json"):
        with open("cache.json") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open("cache.json", "w") as f:
        json.dump(cache, f)

def delete_cache():
    if os.path.exists("cache.json"):
        os.remove("cache.json")
        print("Cache file deleted.")
    else:
        print("Cache file does not exist.")
        