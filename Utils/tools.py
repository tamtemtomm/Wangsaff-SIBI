import os, time
import numpy as np
import torch
import tensorflow as tf
import mediapipe as mp
import cv2
from Utils.models import SIBIModelTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exctract_feature(im):
  if isinstance(im, str):
    assert os.path.isfile(im), f"Filepath Error: {im} file cannot be found" 
    im = cv2.imread(im)
    
  im_height, im_width, _ = im.shape

  mp_hands = mp.solutions.hands
  with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:
    while True:
      results = hands.process(cv2.flip(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1))

      if not results.multi_hand_landmarks:
        return None, None

      annot = []
      for hand_landmarks in results.multi_hand_landmarks:
        annot.append(hand_landmarks)
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
                        ]).reshape(-1, 1)), annot

def torch_output(features, model):
    features = torch.tensor(features, dtype=torch.float32)
    features = features.squeeze(dim=1).unsqueeze(dim=0).unsqueeze(dim=0)
    
    out = model(features.to(device))
    pred = chr(torch.argmax(out, dim=1).item() + 65)
    return pred

def keras_output(features, model):
    features = features.reshape(1, -1, 1)
    
    prediction = model.predict(features)
    pred = chr(np.argmax(prediction, axis=1)[0] + 65)
    return pred

def torch_load_model(model, model_path:str):
    model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
    return model

def keras_load_model(model_path:str):
    return tf.keras.models.load_model(model_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    if model_path[-3:] == '.h5':
        model = keras_load_model(model_path)
        mode = 'keras'
    elif model_path[-4:] == '.pth':
      model = torch_load_model(SIBIModelTorch(26).to(device), model_path)
      mode = 'torch'
  
def extract_video(frame, model_path:str,  width=720, height=480):
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands
  
  features, annot = exctract_feature(frame)
  
  if model_path[-3:] == '.h5':
      model = keras_load_model(model_path)
      mode = 'keras'
  elif model_path[-4:] == '.pth':
    model = torch_load_model(SIBIModelTorch(26).to(device), model_path)
    mode = 'torch'
  
  if (features is not None):
    if mode == 'keras':
      pred = keras_output(features, model)
    elif mode == 'torch':
      pred = torch_output(features, model)
      
  else : 
    pred = "Tidak ada hasil"
  
  xMax, yMax, xMin, yMin = [0,0,0,0]
  has = cv2.flip(frame,1)
  if(annot):
      for tang_l in annot:
          kord = tang_l.landmark
          yMax = float(str(max(kord, key= lambda lm: lm.y)).split('\n')[1].split(" ")[1]) * height
          xMax = float(str(max(kord, key= lambda lm: lm.x)).split('\n')[0].split(" ")[1]) * width
          yMin = float(str(min(kord, key= lambda lm: lm.y)).split('\n')[1].split(" ")[1]) * height
          xMin = float(str(min(kord, key= lambda lm: lm.x)).split('\n')[0].split(" ")[1]) * width
          mp_drawing.draw_landmarks(
              has,
              tang_l,
              mp_hands.HAND_CONNECTIONS)
  
  cv2.rectangle(has, (int(xMin)-40, int(yMin)-20), (int(xMax)-30, int(yMax)+10), (255,0,0), 4)
  cv2.rectangle(has, (int(xMin)-40, int(yMin)-50), (int(xMax)-30, int(yMin)-20),(255,0,0), -1)
  cv2.putText(has, f'Huruf: {pred if (features is not None and features[3][0] != 0 and features[39][0] !=0) else "Tidak diketahui"}', (int(xMin)-20,int(yMin)-30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
  
  return pred, has
            

def read_video(model_path:str, input_video=None, output_video=None, show=True):
  
  assert os.path.isfile(model_path), f"File Error : {model_path} cannot be found"
  
  preds = []
  timer = 0
  
  
  if input_video is not None:
    cap = cv2.VideoCapture(input_video)
    _, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    
    if output_video is not None:
      output = cv2.VideoWriter(output_video,
                               fourcc,
                               12,
                               (width, height))
  else : 
    width, height = 720, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)    
 
  while True:
    times = time.time()
    
    ret, frame = cap.read()
    pred, has = extract_video(frame, model_path, width, height)

    print(pred)
    preds.append(pred)
            
    fps = 1/(times-timer)
    
    cv2.putText(has, f'FPS: {float(fps):.2f}', (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)
    
    if output_video is not None:
      if ret :
        output.write(has)
      else : 
        output.release()
        break
      
    if show : 
      cv2.imshow("hasil", has)
      timer = times
      if(cv2.waitKey(1) == ord("q")):
          cv2.destroyAllWindows()
          if output_video : output.release()
          break
  
  return preds

def translate_preds(preds):
  translate_pred = [" "]
  word_freq = {chr(i+65):0 for i in range(26)}
  word_freq["Tidak ada hasil"] = 0
  prev = "Tidak ada hasil"
  
  for pred in preds:
    
    if pred != prev:
      word_freq[prev] = 0
    
    if pred == "Tidak ada hasil" : 
      word_freq[pred] += 1
      if prev == pred and word_freq[pred] >= 7 and translate_pred[-1] != " ": 
        translate_pred.append(" ")
        
    else :
      word_freq[pred] += 1
      if prev==pred and word_freq[pred] >= 7 and translate_pred[-1] != pred:
        translate_pred.append(pred)
    
    prev = pred
    
  return ''.join(translate_pred).lstrip()