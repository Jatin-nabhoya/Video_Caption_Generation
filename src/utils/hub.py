import json, os
from huggingface_hub import HfApi, create_repo, snapshot_download

def push_checkpoint(model, processor, state, repo_id, subfolder, local_root="/kaggle/working/ckpt"):
    """Save model, processor, and state, then upload them to a private HF model repo."""
    path = os.path.join(local_root, subfolder)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    json.dump(state, open(os.path.join(path, "state.json"), "w"), indent=2)
    create_repo(repo_id, private=True, exist_ok=True)
    HfApi().upload_folder(folder_path=path, repo_id=repo_id, path_in_repo=subfolder, commit_message=f"{subfolder} epoch {state.get('epoch')}")
    print(f"pushed -> {repo_id}/{subfolder}")

def try_resume(repo_id, subfolder):
    """Return a local checkpoint path and state, or (None, None)."""
    try:
        root = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
        path = os.path.join(root, subfolder)
        return path, json.load(open(os.path.join(path, "state.json")))
    except Exception as e:
        print(f"no checkpoint at {repo_id}/{subfolder} ({type(e).__name__}) -> starting fresh")
        return None, None
