import torch

def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return False
    try:
        if not torch.backends.cudnn.enabled:
            raise RuntimeError("cuDNN is not enabled.")
    except Exception as e:
        print("cuDNN could not be loaded. Please check your cuDNN installation.")
        print(e)
        return False
    return True