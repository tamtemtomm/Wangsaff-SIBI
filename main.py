print("SEDANG MENYIAPKAN!\nTunggu sebentar")
from new_tools import Tools, Window
import cv2
from config import KERAS_MODEL_PATH, TORCH_MODEL_PATH, VIDEO_TEST_PATH, OUTPUT_VIDEO_TEST_PATH, BOW_PATH

if __name__ == '__main__':
    window = Window(TORCH_MODEL_PATH,BOW_PATH)
    window.start()



