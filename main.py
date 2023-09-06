from Utils.tools import read_video, translate_preds
from Utils.config import KERAS_MODEL_PATH, TORCH_MODEL_PATH, VIDEO_TEST_PATH, OUTPUT_VIDEO_TEST_PATH, BOW_PATH

if __name__ == '__main__':
    preds = read_video(TORCH_MODEL_PATH)
    print(translate_preds(preds, BOW_PATH))