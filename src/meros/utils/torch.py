def configure_torch():
    import cv2 as cv
    import torch

    torch.set_num_threads(4)
    cv.setNumThreads(2)

    assert torch.cuda.is_available(), "CUDA is required for video detection"
