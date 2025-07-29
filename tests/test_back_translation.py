from lib.data_augmentation import (
    back_translation_augmenter,
    load_and_quantize,
    ALL_PIVOTS,
)
import polars as pl
from rich.progress import Progress
import torch


def test_bt():
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.hip:", torch.version.hip)
    print("Device count           :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f" Device {i} name      :", torch.cuda.get_device_name(i))

    texts = [
        "Wenn Sie als kleines oder mittleres Unternehmen für Beteiligungen von privaten Kapitalbeteiligungsgesellschaften eine Ausfallgarantie benötigen, kann diese unter bestimmten Voraussetzungen von der Bürgschaftsbank Sachsen übernommen werden.",
        "Die Bürgschaftsbank Sachsen übernimmt Ausfallgarantien für Beteiligungen von privaten Kapitalbeteiligungsgesellschaften an kleinen und mittleren Unternehmen in Sachsen, wenn die Beteiligungen ohne Garantien nicht oder nicht zu angemessenen Bedingungen zustande kommen würden.",
    ]
    labels = [0, 1]

    quant_pipes = {p: load_and_quantize(p) for p in ALL_PIVOTS}

    with Progress() as progress:
        augmented_texts, augmented_labels = back_translation_augmenter(
            quant_pipes, texts, labels, progress=progress, factor=2
        )

    assert len(augmented_texts) == 4
    assert augmented_labels == 2 * labels

    with Progress() as progress:
        augmented_texts, augmented_labels = back_translation_augmenter(
            quant_pipes, texts, labels, progress=progress, factor=3
        )

    assert len(augmented_texts) == 8
    assert augmented_labels == 4 * labels

    with Progress() as progress:
        augmented_texts, augmented_labels = back_translation_augmenter(
            quant_pipes, texts, labels, progress=progress, factor=4
        )

    assert len(augmented_texts) == 16
    assert augmented_labels == 8 * labels

    with Progress() as progress:
        augmented_texts, augmented_labels = back_translation_augmenter(
            quant_pipes, texts, labels, progress=progress, factor=5
        )
    assert len(augmented_texts) == 32
    assert augmented_labels == 16 * labels

    print(augmented_texts)
