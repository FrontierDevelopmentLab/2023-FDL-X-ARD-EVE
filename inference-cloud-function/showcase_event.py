import pandas as pd
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor
from google.cloud import pubsub_v1


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
    return future


date_list = get_date_list(begin_date='2011-01-01T01:00:00', end_date='2011-02-01 00:00:00')
futures = []
for inference_date in date_list:
    timestamp = inference_date.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Submitting: {timestamp} to pubsub")

    data = str({"timestamp": timestamp, "api_endpoint_url": url})
    future = publish_message(data)
    futures.append(future)


results = [future.result() for future in futures]
print(results)
print(f"Published {len(results)} messages to {topic_name}")