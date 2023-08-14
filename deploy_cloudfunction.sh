gcloud functions deploy virtual-eve \
    --gen2 \
    --region=us-central1 \
    --runtime=python311 \
    --source=inference-cloud-function \
    --entry-point=hello_http \
    --trigger-http