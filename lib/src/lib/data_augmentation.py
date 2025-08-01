import re
import openai
from deepeval.models.base_model import DeepEvalBaseLLM
import dspy
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, concatenate_datasets, ClassLabel, Features, Value
from tqdm import tqdm
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from collections import Counter
import datasets


class CustomModel(DeepEvalBaseLLM):
    def __init__(self, model_name, base_url, temperature, api_key):
        self.model_name = model_name
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.temperature = temperature

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def extend_pipeline(
    dataset,
    num_rounds,
    model,
    base_url,
    api_key,
    labeled=True,
    logging=False,
    text_col="text",
    quality_threshold=0.7,
):
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    lm = dspy.LM(
        f"openai/{model}",
        api_key=api_key,
        base_url=base_url,
        temperature=1,
        cache=True,
    )

    dspy.configure(lm=lm)

    criteria = [
        "Verify that 'actual output' conveys the same core meaning and implications as 'input'",
        "Verify that 'actual output' also follows the input's structure of <title>: <description>"
        "Verify that 'actual output' is written in syntactically correct German",
    ]

    de_model = CustomModel(
        model_name=model, base_url=base_url, temperature=0, api_key=api_key
    )

    paraphrase_quality_metric = GEval(
        name="Paraphrase Quality",
        evaluation_steps=criteria,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=de_model,
        verbose_mode=False,
    )

    test_case = LLMTestCase(
        input="Ich mag keine Hunde.", actual_output="Hunde mag ich nicht."
    )

    assert paraphrase_quality_metric.measure(test_case, _show_indicator=False) > 0.5

    test_case = LLMTestCase(input="I do not like dogs.", actual_output="I like dogs.")
    assert paraphrase_quality_metric.measure(test_case, _show_indicator=False) < 0.5

    test_case = LLMTestCase(
        input="Mein Geständis: Ich mag keine Hunde.",
        actual_output="My confession is that I dislike dogs",
    )
    score = paraphrase_quality_metric.measure(test_case, _show_indicator=False)

    assert score <= 0.8

    class Paraphrase(dspy.Signature):
        """Paraphrases a German text while preserving its original structure.
        If the input follows the format '<title>: <text>', maintain this exact structure in the output.
        The paraphrase should:
        - Retain the same meaning as the original
        - Use different vocabulary and sentence structures where possible
        - Not add any additional colons at the beginning of the text
        """

        text = dspy.InputField(desc="Original German text to paraphrase")
        paraphrased_text = dspy.OutputField(
            desc="Paraphrased version in German that preserves the original structure"
        )

    program = dspy.Predict(Paraphrase)

    def extend_dataset_with_paraphrases(
        base_dataset,
        num_rounds=1,
        quality_threshold=quality_threshold,
        max_workers=50,
        program=program,
        paraphrase_quality_metric=paraphrase_quality_metric,
        labeled=labeled,
        text_col="text",
    ):
        all_datasets = [base_dataset]
        current_dataset = base_dataset

        # Only cast label column if the dataset is labeled
        if labeled:
            current_dataset = base_dataset.cast_column(
                "label", ClassLabel(names=["dfn", "dfy"])
            )

        def pp_one(text, threshold=quality_threshold):
            """
            Paraphrases `text` but preserves the parenthetical metadata.

            """
            m = re.match(r"^(.*?)\s*(\([^)]*\)):\s*(.*)$", text)
            if m:
                title, meta, desc = m.group(1), m.group(2), m.group(3)
                to_paraphrase = f"{title}: {desc}"
            else:
                # fallback if no meta block found
                to_paraphrase = text
                meta = ""

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    result = program(text=to_paraphrase)
                    paraphr = result.paraphrased_text

                    test_case = LLMTestCase(input=to_paraphrase, actual_output=paraphr)
                    score = paraphrase_quality_metric.measure(
                        test_case, _show_indicator=False
                    )

                    if score >= threshold:
                        if meta:
                            new_title, new_desc = paraphr.split(":", 1)
                            new_title = new_title.strip()
                            new_desc = new_desc.strip()
                            return f"{new_title} {meta}: {new_desc}"
                        else:
                            return paraphr

                except Exception as e:
                    if logging:
                        print(e)

                if logging:
                    print(f"Failed (attempt {attempt + 1}): score={score}\n{paraphr}\n")

            raise RuntimeError(
                f"Failed to generate quality paraphrase after {max_attempts} attempts"
            )

        for round_num in range(1, num_rounds + 1):
            if logging:
                print(f"Starting paraphrasing round {round_num} of {num_rounds}")

            # Use the text from the most recent dataset
            texts_to_paraphrase = current_dataset[text_col]

            # Only get labels if the dataset is labeled
            if labeled:
                current_labels = current_dataset["label"]

            successful_paraphrases = []
            successful_indices = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(pp_one, item) for item in texts_to_paraphrase
                ]
                for i, future in enumerate(
                    tqdm(futures, desc=f"Round {round_num} Paraphrasing")
                ):
                    try:
                        successful_paraphrases.append(future.result())
                        successful_indices.append(i)
                    except Exception as e:
                        print(f"Error: {e}")

            # Create dataset dict based on whether we have labels or not
            if labeled:
                successful_labels = [current_labels[i] for i in successful_indices]
                round_ds = Dataset.from_dict(
                    {text_col: successful_paraphrases, "label": successful_labels}
                )
            else:
                round_ds = Dataset.from_dict({text_col: successful_paraphrases})

            all_datasets.append(round_ds)

            # Update current_dataset to be the newly created dataset for the next round
            current_dataset = round_ds
            if logging:
                print(
                    f"Completed round {round_num}: Generated {len(round_ds)} paraphrases"
                )

        # Concatenate all datasets
        if labeled:
            target_features = Features(
                {
                    "text": Value(dtype="large_string"),
                    "label": ClassLabel(names=["dfn", "dfy"]),
                }
            )

            extended_dataset = concatenate_datasets(
                [
                    ds.select_columns(["label", "text"]).cast(target_features)
                    for ds in all_datasets
                ]
            )
            if logging:
                print(f"Original dataset size: {len(base_dataset)}")
                print(f"Final dataset size: {len(extended_dataset)}")
                print(f"Final label distribution: {Counter(extended_dataset['label'])}")
        else:
            target_features = Features(
                {
                    "text": Value(dtype="large_string"),
                }
            )
            extended_dataset = concatenate_datasets(
                [
                    ds.select_columns(["text"]).cast(target_features)
                    for ds in all_datasets
                ]
            )
            if logging:
                print(f"Original dataset size: {len(base_dataset)}")
                print(f"Final dataset size: {len(extended_dataset)}")

        return extended_dataset.shuffle(42)

    extended_dataset = extend_dataset_with_paraphrases(
        dataset, num_rounds=num_rounds, labeled=labeled, text_col=text_col
    )
    return extended_dataset
