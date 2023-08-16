import pandas as pd
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor
from google.cloud import pubsub_v1

import time

token = subprocess.run('gcloud auth print-identity-token', shell=True, capture_output=True, text=True).stdout.strip()
headers = {"Authorization": f"bearer {token}", "Content-Type": "application/json"}
publisher = pubsub_v1.PublisherClient()
topic_name = "projects/us-fdl-x/topics/us-fdl-x-ard-terraform-pubsub-topic-orchestrator"
url = "https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve"


def get_date_list(begin_date, end_date):
    start_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    date_list = pd.date_range(start=start_date, end=end_date, freq='12min').tolist()
    return date_list


def publish_message(data):
    future = publisher.publish(topic_name, data=data.encode("utf-8"))
    time.sleep(0.1)
    return future


for year in range(2013, 2015):

    date_list = get_date_list(begin_date=f'{year}-01-01 00:00:00', end_date=f'{year}-12-31 23:48:00')

    print(f"Sending {len(date_list)} messages for year {year}")

    futures = []
    for inference_date in date_list:
        timestamp = inference_date.strftime('%Y-%m-%dT%H:%M:%S')

        data = str({"timestamp": timestamp, "api_endpoint_url": url})
        future = publish_message(data)
        futures.append(future)


    results = [future.result() for future in futures]
    print(f"Published {len(results)} messages for year {year}")
    time.sleep(1800)