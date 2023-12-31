from huggingface_hub import create_repo
import datasets

ORGANIZATION = "lyon-nlp/"
REPO_NAME = "sick-fr"
BASE_DATASET_REPO = "Lajavaness/SICK-fr"

base_dataset = datasets.load_dataset(BASE_DATASET_REPO)
base_dataset = base_dataset.rename_column("sentence_A", "sentence1")
base_dataset = base_dataset.rename_column("sentence_B", "sentence2")
base_dataset = base_dataset.rename_column("relatedness_score", "score")
base_dataset = base_dataset.rename_column("Unnamed: 0", "id")

create_repo(
    ORGANIZATION + REPO_NAME,
    repo_type="dataset"
)

base_dataset.push_to_hub(ORGANIZATION + REPO_NAME)
