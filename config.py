# config.py

class Config:
    DATA_ROOT = "data/mvtec_anomaly_detection"
    CATEGORY = "hazelnut"
    IMG_SIZE = 256

    EPOCHS = 100
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    FPR_LIMIT = 0.3

    WEIGHTS_DIR = "outputs/weights"
    MAPS_DIR = "outputs/anomaly_maps"
    RESULTS_DIR = "outputs/results"

    DEVICE = "cuda"