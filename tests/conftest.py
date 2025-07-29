# # conftest.py
# import os

# # 1) Disable the safetensors‐conversion background thread
# os.environ["TRANSFORMERS_NO_SAFETENSORS_CONVERSION"] = "1"

# # 2) Hide your GPU so that pipeline(...) picks CPU by default
# #    (torch.cuda.is_available() will be False)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# # 3) In case transformers was already imported earlier, monkey-patch
# try:
#     from transformers.utils import safetensors_conversion as _stc
#     _stc.auto_conversion = lambda *args, **kwargs: None
# except ImportError:
#     pass
