from transformers import AutoProcessor, GitForCausalLM

def load_git(name_or_path="microsoft/git-base", num_frames=6):
    """Load GIT with one temporal embedding per frame."""
    processor = AutoProcessor.from_pretrained(name_or_path)
    model = GitForCausalLM.from_pretrained(name_or_path, num_image_with_embedding=num_frames)
    return processor, model
