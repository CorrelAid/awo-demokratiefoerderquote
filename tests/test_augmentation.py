from lib.data_augmentation import extend_pipeline
from datasets import Dataset
from dotenv import load_dotenv
import os

load_dotenv()


def test_extend_pipeline():
    dataset = Dataset.from_dict(
        {
            "text": [
                "Förderprogram A(CatA: df, CatB: df): Dieses Programm fördert die Entwicklung von Technologien.",
                "Förderprogram B(CatA: df, CatB: df): Dieses Programm fördert die Entwicklung von Demokratie.",
            ],
            "label": [0, 1],
        },
    )
    api_key = os.getenv("OR_KEY")
    extended_dataset = extend_pipeline(
        dataset,
        2,
        "moonshotai/kimi-k2",
        "https://openrouter.ai/api/v1",
        api_key,
        logging=True,
        labeled=True,
    )
    print(extended_dataset["text"])
    print(extended_dataset["label"])
