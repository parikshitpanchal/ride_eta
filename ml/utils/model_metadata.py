import json
from pathlib import Path

def save_model_metadata(
    file_path: Path,
    numerical_feature_count: int,
    categorical_cardinalities: dict[str, int],
) -> None:

    metadata = {
        "numerical_feature_count": numerical_feature_count,
        "categorical_cardinalities": categorical_cardinalities,
    }

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )
        
def load_model_metadata(
    file_path: Path,
) -> dict:

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)