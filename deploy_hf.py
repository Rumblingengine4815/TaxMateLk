from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("HF_API_KEY")
if not token:
    raise ValueError("HF_API_KEY not found in .env")

api = HfApi(token=token)
repo_id = "Lorion4815/TaxMateLk"

print(f"Creating or updating space {repo_id}...")
try:
    create_repo(repo_id, repo_type="space", space_sdk="gradio", token=token, exist_ok=True)
except Exception as e:
    print(f"Repo might exist or error: {e}")

print("Setting secrets...")
api.add_space_secret(repo_id, "GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
api.add_space_secret(repo_id, "FINE_TUNED_MODEL_ID", "Lorion4815/taxmate-lk-merged")
api.add_space_secret(repo_id, "HF_TOKEN", token)

print("Uploading files...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[".git", ".venv", "__pycache__", ".pytest_cache", ".env", "*.sqlite3-journal", "deploy_hf.py", "uv.lock"],
)

print("Deployment successfully completed!")
