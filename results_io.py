"""Loading of the evaluation results (results2.json.gz, or an uncompressed results2.json)."""
import gzip, json, os


def load_results2(path='results2.json.gz'):
    if not os.path.exists(path) and os.path.exists('results2.json'):
        path = 'results2.json'
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def save_results2(obj, path='results2.json.gz'):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(obj, f)
