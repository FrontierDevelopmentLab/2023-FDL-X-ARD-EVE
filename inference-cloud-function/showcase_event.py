import pandas as pd
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor

token = subprocess.run('gcloud auth print-identity-token', shell=True, capture_output=True, text=True).stdout.strip()
headers = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}

def get_date_list():
    start_date = pd.to_datetime('2017-09-26')
    end_date = pd.to_datetime('2017-09-28')
    date_list = pd.date_range(start=start_date, end=end_date, freq='12min').tolist()
    return date_list

def eve_endpoint(data):
    result = requests.post(
        url="https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve",
        headers=headers,
        json=data,
    )
    return result


date_list = get_date_list()

futures = []
with ThreadPoolExecutor() as executor:
    for inference_date in date_list:
        timestamp = inference_date.strftime('%Y-%m-%dT%H:%M:%S')
        print(f"Submitting: {timestamp} to thread pool")

        message = {"time": timestamp}
        futures.append(executor.submit(eve_endpoint, message))

    results = [future.result() for future in futures]

for result in results:
    print(f"Submitted {len(date_list)} requests to EVE endpoint.")
    print(result.text)