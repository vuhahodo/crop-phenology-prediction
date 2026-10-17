import requests
import concurrent.futures
import os
import time
import sys

URL = "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh02.h5.tar.gz"
FILENAME = "data/breizhcrops_raw/2017/L1C/frh02.h5.tar.gz"
NUM_THREADS = 16

def get_file_size(url):
    response = requests.head(url)
    return int(response.headers.get('content-length', 0))

def download_chunk(url, start, end, chunk_index, filename):
    headers = {'Range': f'bytes={start}-{end}'}
    response = requests.get(url, headers=headers, stream=True)
    
    with open(f"{filename}.part{chunk_index}", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return chunk_index

def parallel_download():
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    file_size = get_file_size(URL)
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    if file_size == 0:
        print("Could not get file size. Aborting.")
        return
        
    chunk_size = file_size // NUM_THREADS
    futures = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for i in range(NUM_THREADS):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < NUM_THREADS - 1 else file_size - 1
            print(f"Starting chunk {i}: {start} - {end}")
            futures.append(executor.submit(download_chunk, URL, start, end, i, FILENAME))
            
        for future in concurrent.futures.as_completed(futures):
            print(f"Chunk {future.result()} finished.")
            
    print(f"All chunks downloaded in {time.time() - start_time:.2f}s. Merging...")
    
    with open(FILENAME, "wb") as outfile:
        for i in range(NUM_THREADS):
            part_file = f"{FILENAME}.part{i}"
            with open(part_file, "rb") as infile:
                outfile.write(infile.read())
            os.remove(part_file)
            
    print("Merge complete!")

if __name__ == "__main__":
    parallel_download()
