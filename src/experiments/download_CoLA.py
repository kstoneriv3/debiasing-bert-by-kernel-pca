import argparse
import os
import requests

def download(url, data_path):
    r = requests.get(url, allow_redirects=True)
    open(data_path, 'wb').write(r.content)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()

    url = "https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/tokenized/mixed_dev.tsv"
    data_path = os.path.join(args.data_dir, "mixed_dev.tsv")
    download(url, data_path)
    

