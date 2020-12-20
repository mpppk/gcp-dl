#! /bin/bash

export IMAGE_FAMILY="tf2-ent-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="dl"
export INSTANCE_TYPE="n1-standard-2"
export PROJECT="mpppk-workspace"
export STARTUP_SCRIPT="gs://mpppk-fashion-mnist/startup.sh"
export SHUTDOWN_SCRIPT="gs://mpppk-fashion-mnist/shutdown.sh"

gcloud compute instances create $INSTANCE_NAME \
 --project=$PROJECT \
 --zone=$ZONE \
 --machine-type=$INSTANCE_TYPE \
 --boot-disk-size=50GB \
 --image-family=$IMAGE_FAMILY \
 --image-project=deeplearning-platform-release \
 --accelerator=type=nvidia-tesla-t4,count=1 \
 --scopes=https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/devstorage.full_control \
 --metadata=startup-script-url=$STARTUP_SCRIPT,shutdown-script-url=$SHUTDOWN_SCRIPT \
 --preemptible

gcloud compute config-ssh
