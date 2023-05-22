import os
import geojson
import requests

import concurrent.futures

def read_geojson(file):
    with open(file) as f:
        gj = geojson.load(f)
    return gj['features'][0]

def download_url(url, save_path):
    response = requests.get(url)
    with open("{}.tiff".format(save_path), 'wb') as f:
        f.write(response.content)

def download_urls(title_urls, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        future_to_url = {}
        for filename, url in title_urls:
            save_path = os.path.join(save_dir, filename)
            future = pool.submit(download_url, url, save_path)
            future_to_url[future] = url
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
