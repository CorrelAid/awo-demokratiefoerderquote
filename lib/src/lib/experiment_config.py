n_splits = 10
random_state = 42
test_size = 0.3

code_book_path = "codebook/codebook_compact_llm.md"

data_path = "data/labeled/25_07/to_classify.parquet"

gen_llm_url = "https://openrouter.ai/api/v1"

augment_model = "moonshotai/kimi-k2"
augment_base_url = "https://openrouter.ai/api/v1"

cpt_bert = "FundedBert"
# MODAL_WORKSPACE = "correlaid"
# environment = None
# prefix = MODAL_WORKSPACE + (f"-{environment}" if environment else "")
# TRANSLATOR_APP_NAME = "translator"
# TRANSLATOR_FUNCTION_NAME = "serve"
# TRANSLATOR_URL = f"https://{prefix}--{TRANSLATOR_FUNCTION_NAME}.modal.run"
