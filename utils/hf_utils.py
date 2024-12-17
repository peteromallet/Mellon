from huggingface_hub import scan_cache_dir
from config import config
import os
import json
from pathlib import Path

# TODO: find better strategy to find different kinds of models
def list_local_models():
    cache_dir = config.hf['cache_dir']
    cache_info = scan_cache_dir(cache_dir)
    local_models = [] #[model.repo_id for model in cache_info.repos]

    for repo in cache_info.repos:
        revision = list(repo.revisions)[-1] if repo.revisions else None

        if not revision:
            continue

        config_path = next((f for f in revision.files if f.file_name == 'model_index.json'), None)
        if not config_path:
            continue

        config_path = Path(config_path.file_path)

        if config_path.exists():
            with open(config_path, 'r') as f:
                model_info = json.load(f)
            if '_class_name' in model_info and 'Pipeline' in model_info['_class_name']:
                local_models.append(repo.repo_id)

    local_models.sort()
    return local_models

def get_repo_path(model_id):
    cache_dir = config.hf['cache_dir']
    cache_info = scan_cache_dir(cache_dir)
    for repo in cache_info.repos:
        if repo.repo_id == model_id:
            latest_revision = list(repo.revisions)[-1] if repo.revisions else None
            if latest_revision:
                return os.path.join(repo.repo_path, latest_revision.snapshot_path)

    return None

def is_local_files_only(model_id):
    return config.hf['online_status'] == 'Local files only' or (config.hf['online_status'] == 'Connect if needed' and model_id in list_local_models())
