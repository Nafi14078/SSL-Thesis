from utils.preprocessing import process_brats

process_brats(
    brats_path=r"E:\SSL Thesis\data\raw",
    output_path=r"E:\SSL Thesis\processed",
    modality="flair"
)