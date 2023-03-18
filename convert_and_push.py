from transformers import AutoTokenizer, T5ForConditionalGeneration
from huggingface_hub import HfApi

api = HfApi()

folder_path = 'experiments/harem-tgf-ptbr-flan-t5-base-22-1th-ep150-len256-b8-train/episode0'

print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('thegoodfellas/tgf-ptbr-flan-t5-base-22-1th', use_auth_token=True)

print('loading model')
model = T5ForConditionalGeneration.from_pretrained(folder_path, from_flax=True)

model.save_pretrained(folder_path)
tokenizer.save_pretrained(folder_path)

print('uploading...')
api.upload_folder(
  folder_path=folder_path,
  repo_id="thegoodfellas/tgf-flan-t5-base-final",
  commit_message="new version trained from pt"
)

