from huggingface_hub import HfApi
import os

api = HfApi()

for root, dirs, files in os.walk('./bert_phishing_5k_benchmark'):
    for f in files:
        print("Preparing:", os.path.join(root, f))

api.upload_folder(
    folder_path='./bert_phishing_5k_benchmark',
    repo_id='IRBXrocket/phishnet-bert',
    repo_type='model'
)

print("Upload complete")