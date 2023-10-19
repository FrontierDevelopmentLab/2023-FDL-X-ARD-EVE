# The following script copies data from the gcp bucket to the local directory:

# Copy the data from the bucket to the local directory:

gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/AIA.zarr/2021 /mnt/sdomlv2_full/sdomlv2.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/AIA.zarr/2022 /mnt/sdomlv2_full/sdomlv2.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/AIA.zarr/2023 /mnt/sdomlv2_full/sdomlv2.zarr/

gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/HMI.zarr/2010 /mnt/sdomlv2_hmi/sdomlv2_hmi.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/HMI.zarr/2011 /mnt/sdomlv2_hmi/sdomlv2_hmi.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/HMI.zarr/2012 /mnt/sdomlv2_hmi/sdomlv2_hmi.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/HMI.zarr/2013 /mnt/sdomlv2_hmi/sdomlv2_hmi.zarr/
gsutil -m cp -r gs://us-fdlx-ard-sdomlv2a/HMI.zarr/2014 /mnt/sdomlv2_hmi/sdomlv2_hmi.zarr/

sudo chmod a+w /mnt/sdomlv2_full2
sudo mount -o discard,defaults /dev/sdd /mnt/sdomlv2_full2
