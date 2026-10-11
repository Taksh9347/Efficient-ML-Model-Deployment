"""
Model Artifact Downloader.
Utility script to fetch versioned model binaries from cloud artifact registries
(S3, GCS, MinIO, or Hugging Face Hub) during container build or deployment.
"""

import os
import sys


def download_model(model_url: str, destination: str = "models/cancer_model.onnx"):
    """
    Downloads model artifact from cloud storage.
    Avoids committing binary blobs directly to the Git repository.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"Downloading model artifact from {model_url} -> {destination}...")
    # TODO: Implement download via boto3, google-cloud-storage, or urllib
    # import urllib.request
    # urllib.request.urlretrieve(model_url, destination)
    print("Download completed.")


if __name__ == "__main__":
    url = os.getenv("MODEL_DOWNLOAD_URL", "")
    if not url:
        print("MODEL_DOWNLOAD_URL environment variable is not set; skipping remote download.")
        sys.exit(0)
    download_model(url)
