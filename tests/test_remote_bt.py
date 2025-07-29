import asyncio
import os
from dotenv import load_dotenv
from lib.data_augmentation import remote_back_translation_augmenter

load_dotenv()


def test_remote_bt():
    texts = [
        "Wenn Sie als kleines oder mittleres Unternehmen für Beteiligungen von privaten Kapitalbeteiligungsgesellschaften eine Ausfallgarantie benötigen"
    ]
    labels = [1]

    # Run the async function in an event loop
    loop = asyncio.get_event_loop()
    res_texts, res_labels = loop.run_until_complete(
        remote_back_translation_augmenter(texts, labels, 2)
    )

    assert len(res_texts) == 2
    assert len(res_labels) == 2
    assert res_labels == [1, 1]
    assert res_texts[0] != res_texts[1]


test_remote_bt()
