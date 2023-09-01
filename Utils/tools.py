import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import mediapipe as mp
import cv2

imgtest = 'Utils/test.jpg'

def read_img(im_path):
  im = cv2.imread(im_path)
  im_height, im_width, _ = im.shape

  mp_hands = mp.solutions.hands
  mp_drawing = mp.solutions.drawing_utils
  with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
    while True:
      results = hands.process(cv2.flip(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1))

      if not results.multi_hand_landmarks:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0)

      annotated_im = cv2.flip(im.copy(), 1)
      for hand_landmarks in results.multi_hand_landmarks:
      # pgelang Hand /  Pergelangan Tangan
        pgelangX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * im_width
        pgelangY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * im_height
        pgelangZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

        # jempol Finger / Ibu Jari
        jempol_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * im_width
        jempol_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * im_height
        jempol_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

        jempol_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * im_width
        jempol_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * im_height
        jempol_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

        jempol_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * im_width
        jempol_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * im_height
        jempol_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

        jempol_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * im_width
        jempol_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * im_height
        jempol_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

        # telunjuk Finger / Jari Telunjuk
        telunjuk_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * im_width
        telunjuk_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * im_height
        telunjuk_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

        telunjuk_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * im_width
        telunjuk_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * im_height
        telunjuk_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

        telunjuk_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * im_width
        telunjuk_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * im_height
        telunjuk_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

        telunjuk_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * im_width
        telunjuk_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * im_height
        telunjuk_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

        # tengah Finger / Jari Tengah
        tengah_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * im_width
        tengah_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * im_height
        tengah_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

        tengah_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * im_width
        tengah_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * im_height
        tengah_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

        tengah_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * im_width
        tengah_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * im_height
        tengah_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

        tengah_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * im_width
        tengah_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * im_height
        tengah_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

        # manis Finger / Jari Cincin
        manis_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * im_width
        manis_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * im_height
        manis_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

        manis_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * im_width
        manis_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * im_height
        manis_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

        manis_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * im_width
        manis_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * im_height
        manis_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

        manis_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * im_width
        manis_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * im_height
        manis_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

        # kelingking Finger / Jari Kelingking
        kelingking_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * im_width
        kelingking_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * im_height
        kelingking_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

        kelingking_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * im_width
        kelingking_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * im_height
        kelingking_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

        kelingking_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * im_width
        kelingking_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * im_height
        kelingking_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

        kelingking_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * im_width
        kelingking_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * im_height
        kelingking_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

        mp_drawing.draw_landmarks(annotated_im, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return (np.array([pgelangX, pgelangY, pgelangZ,
                        jempol_CmcX, jempol_CmcY, jempol_CmcZ,
                        jempol_McpX, jempol_McpY, jempol_McpZ,
                        jempol_IpX, jempol_IpY, jempol_IpZ,
                        jempol_TipX, jempol_TipY, jempol_TipZ,
                        telunjuk_McpX, telunjuk_McpY, telunjuk_McpZ,
                        telunjuk_PipX, telunjuk_PipY, telunjuk_PipZ,
                        telunjuk_DipX, telunjuk_DipY, telunjuk_DipZ,
                        telunjuk_TipX, telunjuk_TipY, telunjuk_TipZ,
                        tengah_McpX, tengah_McpY, tengah_McpZ,
                        tengah_PipX, tengah_PipY, tengah_PipZ,
                        tengah_DipX, tengah_DipY, tengah_DipZ,
                        tengah_TipX, tengah_TipY, tengah_TipZ,
                        manis_McpX, manis_McpY, manis_McpZ,
                        manis_PipX, manis_PipY, manis_PipZ,
                        manis_DipX, manis_DipY, manis_DipZ,
                        manis_TipX, manis_TipY, manis_TipZ,
                        kelingking_McpX, kelingking_McpY, kelingking_McpZ,
                        kelingking_PipX, kelingking_PipY, kelingking_PipZ,
                        kelingking_DipX, kelingking_DipY, kelingking_DipZ,
                        kelingking_TipX, kelingking_TipY, kelingking_TipZ
                        ]).reshape(-1, 1)), np.array(annotated_im)

def torch_output(features, model):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    features = features.squeeze(dim=1).unsqueeze(dim=0).unsqueeze(dim=0)
    
    out = model(features)
    pred = chr(torch.argmax(out, dim=1).item() + 65)
    return pred

def keras_output(features, model):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = features.reshape(1, -1, 1)
    
    predictions = model.predict(features)
    pred = chr(np.argmax(predictions, axis=1)[0] + 65)
    return pred