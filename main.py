from Utils.tools import read_video, translate_preds
from Utils.config import KERAS_MODEL_PATH, TORCH_MODEL_PATH, VIDEO_TEST_PATH, OUTPUT_VIDEO_TEST_PATH

if __name__ == '__main__':
    preds = read_video(TORCH_MODEL_PATH,
                       input_video=VIDEO_TEST_PATH,
                       output_video=OUTPUT_VIDEO_TEST_PATH)
    print(translate_preds(preds))