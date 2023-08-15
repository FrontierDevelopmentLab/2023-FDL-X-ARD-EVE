gcloud functions deploy virtual-eve \
    --gen2 \
    --region=us-central1 \
    --runtime=python311 \
    --source=inference-cloud-function \
    --entry-point=hello_http \
    --trigger-http \
    --memory=4GiB



curl -m 70 -X POST https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-H "Content-Type: application/json" \
-d '{
  "name": "Hello World",
  "time": "2010-05-13T01:00:00"
}' &

curl -m 70 -X POST https://us-central1-us-fdl-x.cloudfunctions.net/virtual-eve \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-H "Content-Type: application/json" \
-d '{
  "name": "Hello World",
  "time": "2010-05-13T01:48:00"
}' &