import pandas as pd
import subprocess
from google.cloud import pubsub_v1
from tqdm import tqdm

import time

token = subprocess.run('gcloud auth print-identity-token', shell=True, capture_output=True, text=True).stdout.strip()
publisher = pubsub_v1.PublisherClient()
topic_name = "projects/us-fdl-x/topics/us-fdl-x-ard-terraform-pubsub-topic-orchestrator"
url = "https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve"


def get_date_list(begin_date, end_date):
    start_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    date_list = pd.date_range(start=start_date, end=end_date, freq='12min').tolist()
    return date_list


def publish_message(data):
    publisher.publish(topic_name, data=data.encode("utf-8"))
    time.sleep(0.05)
    # return future


for year in range(2015, 2021):

    date_list = get_date_list(begin_date=f'{year}-01-01 00:00:00', end_date=f'{year}-12-31 23:48:00')

    print(f"Sending {len(date_list)} messages for year {year}")

    for inference_date in tqdm(date_list):
        timestamp = inference_date.strftime('%Y-%m-%dT%H:%M:%S')
        data = str({"timestamp": timestamp, "api_endpoint_url": url})
        publish_message(data)

    time.sleep(60)