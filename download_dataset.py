from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="Angelou0516/brats2023-ped-dataset",
    repo_type="dataset",
    local_dir="E:/SSL Thesis/brats_ped_dataset",
    local_dir_use_symlinks=False  # IMPORTANT on Windows
)

print("Dataset downloaded at:", local_path)