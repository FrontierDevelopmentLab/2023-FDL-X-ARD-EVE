import pandas as pd
import subprocess
from google.cloud import pubsub_v1
from tqdm import tqdm
import google.cloud.bigquery as bq
import json
import time

token = subprocess.run('gcloud auth print-identity-token', shell=True, capture_output=True, text=True).stdout.strip()
publisher = pubsub_v1.PublisherClient()
topic_name = "projects/us-fdl-x/topics/us-fdl-x-ard-terraform-pubsub-topic-orchestrator"
url = "https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve"


with open("inference_config.json", "r") as f:
    config = json.load(f)
cloud_config = config["gcp_config"]

bq_client = bq.Client(project=cloud_config["gcp_project"], location=cloud_config["gcp_region"])


def get_missing_times(start_date, end_date):
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:%M:%S")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:%M:%S")
    query = f"""
        SELECT Time
        FROM `{cloud_config["gcp_project"]}.{cloud_config["dataset"]}.{cloud_config["index_table"]}`
        WHERE Time >= '{start_date}' AND Time <= '{end_date}'
    """
    result = bq_client.query(query).result()
    result = result.to_dataframe()
    result = result["Time"].tolist()
    result = [x.strftime("%Y-%m-%dT%H:%M:%S") for x in result]
    all_times = set(result)

    query = f"""
        SELECT timestamp
        FROM `{cloud_config["gcp_project"]}.{cloud_config["dataset"]}.{cloud_config["inference_table"]}`
        WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
    """
    result = bq_client.query(query).result()
    result = result.to_dataframe()
    result = result["timestamp"].tolist()
    result = [x.strftime("%Y-%m-%dT%H:%M:%S") for x in result]
    inference_times = set(result)

    missing_times = all_times - inference_times
    missing_times = list(missing_times)
    missing_times.sort()

    # print("Missing Times:")
    # print("-"*50)
    # print(missing_times)

    print(f"No. of missing_times: {len(missing_times)}")
    print(f"No. of all_times: {len(all_times)}")
    print(f"No. of inference_times: {len(inference_times)}")

    return missing_times


def publish_message(data):
    publisher.publish(topic_name, data=data.encode("utf-8"))
    time.sleep(0.1)


ts_list = get_missing_times(start_date='2010-05-01 00:00:00', end_date='2020-12-31 23:48:00')

print(f"Sending {len(ts_list)} messages to virtual eve")

for timestamp in tqdm(ts_list):
    timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    data = json.dumps({"timestamp": timestamp, "api_endpoint_url": url})
    publish_message(data)