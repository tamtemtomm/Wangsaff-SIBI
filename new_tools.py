import os, time
import numpy as np
import customtkinter as ctk
import torch
import mediapipe as mp
import threading
import cv2
from PIL import Image
import time
import pickle as pkl
from models import SIBIModelTorch
from Utils.correction import Correction, load_bow
from tools import translate_preds

class Tools:
    def __init__(self, model_path, bow_path, camId:int = 0) -> None:
        self.camera = cv2.VideoCapture(camId)

        self.frame = None
        self.frame_annot = None
        self.detected = None

        self.pred = None
        self.model_mode = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path.split(".")[-1] != "pth":
            raise IOError("tipe model tidak didukung")
        
        self.model_mode = "torch"
        self.model = self.torch_load_model(SIBIModelTorch(26).to(self.device), model_path)

        self.thread = threading.Thread(target=self.update, daemon=True)

    def torch_load_model(self, model, model_path:str):
        model.load_state_dict(torch.load(model_path))
        return model
    
    def start(self):
        self.cap, self.frame = self.camera.read()
        self.frame_annot = self.frame
        self.stop = False
        self.thread.start()

    def close(self):
        self.stop = True
        self.camera.release()
    
    def update(self):
        while True:
            if self.stop:
                break
            
            self.cap, self.frame = self.camera.read()

            if not self.cap:
                print("stopping")
                break

            self.pred, self.frame_annot, self.detected = self.extract_frame(self.frame, self.frame.shape[1], self.frame.shape[0])
            self.frame_annot = np.array(self.frame_annot)
            
    def output(self, features):
        predict = None

        if features is not None:
            if self.model_mode == "keras":
                features = features.reshape(1,-1,1)
                
                predict = self.model.predict(features)
                predict = chr(np.argmax(predict, axis=1)[0]+65)
            elif self.model_mode == "torch":
                features = torch.tensor(features, dtype=torch.float32)
                features = features.squeeze(dim=1).unsqueeze(dim=0).unsqueeze(dim=0)

                predict = self.model(features.to(self.device))
                predict = chr(torch.argmax(predict, dim=1).item() + 65)
        
        return predict


    def extract_frame(self, frame, width=720, height=480, confidence_extraction=0.1, confidence_tracking=0.5):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        detected = False

        features, annot = self.extract_feature(frame, confidence_extraction= confidence_extraction, confidence_tracking=confidence_tracking)

        pred = self.output(features)

        xMax, yMax, xMin, yMin = [0,0,0,0]
        has = cv2.flip(self.frame,1)
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
        
        if xMax+xMin+yMax+yMin >0:
            detected = True
            cv2.rectangle(has, (int(xMin)-20, int(yMin)-20), (int(xMax) + 20, int(yMax)+10), (255,0,0), 4)
            cv2.rectangle(has, (int(xMin), int(yMin)-50), (int(xMax), int(yMin)-20),(255,0,0), -1)
            cv2.putText(has, f'{self.pred if (features is not None and features[3][0] != 0 and features[39][0] !=0) else "Tidak diketahui"}', (int(xMin + (xMax-xMin)/2 -5),int(yMin)-30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255), 1, cv2.LINE_AA)

        return pred, has, detected    

    def extract_feature(self, frame, confidence_extraction = 0.1, confidence_tracking=0.5):
        frame_height, frame_width, _ = frame.shape
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence=confidence_extraction, min_tracking_confidence=confidence_tracking) as hands:
            while True:
                results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
                  
                if not results.multi_hand_landmarks:
                    return None, None
                
                annot = []
                for hand_landmarks in results.multi_hand_landmarks:
                    annot.append(hand_landmarks)
                    # pgelang Hand /  Pergelangan Tangan
                    pgelangX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame_width
                    pgelangY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame_height
                    pgelangZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                    # jempol Finger / Ibu Jari
                    jempol_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * frame_width
                    jempol_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * frame_height
                    jempol_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                    jempol_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * frame_width
                    jempol_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * frame_height
                    jempol_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                    jempol_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * frame_width
                    jempol_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * frame_height
                    jempol_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                    jempol_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width
                    jempol_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame_height
                    jempol_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                    # telunjuk Finger / Jari Telunjuk
                    telunjuk_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * frame_width
                    telunjuk_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * frame_height
                    telunjuk_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                    telunjuk_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * frame_width
                    telunjuk_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * frame_height
                    telunjuk_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                    telunjuk_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * frame_width
                    telunjuk_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * frame_height
                    telunjuk_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                    telunjuk_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width
                    telunjuk_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height
                    telunjuk_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                    # tengah Finger / Jari Tengah
                    tengah_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width
                    tengah_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height
                    tengah_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                    tengah_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * frame_width
                    tengah_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * frame_height
                    tengah_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                    tengah_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * frame_width
                    tengah_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * frame_height
                    tengah_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                    tengah_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_width
                    tengah_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_height
                    tengah_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                    # manis Finger / Jari Cincin
                    manis_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * frame_width
                    manis_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * frame_height
                    manis_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                    manis_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * frame_width
                    manis_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * frame_height
                    manis_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                    manis_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * frame_width
                    manis_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * frame_height
                    manis_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                    manis_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame_width
                    manis_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame_height
                    manis_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                    # kelingking Finger / Jari Kelingking
                    kelingking_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * frame_width
                    kelingking_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * frame_height
                    kelingking_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                    kelingking_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * frame_width
                    kelingking_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * frame_height
                    kelingking_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                    kelingking_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * frame_width
                    kelingking_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * frame_height
                    kelingking_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                    kelingking_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * frame_width
                    kelingking_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * frame_height
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
                
    

class Struct:
    def __init__(self, args) -> None:
        for key, val in args.items():
            if isinstance(key, (list, tuple)):
                setattr(self, key, [Struct(x) if isinstance(x, dict) else x for x in val])
            else: 
                setattr(self, key, Struct(val) if isinstance(val, dict) else val)
    
    def allvar(self):
        return vars(self)

class TextAutoCorrect:
    def __init__(self, model_path:str=r".\Utils\text_model.pkl"):
        self.WORDS = pkl.load(open(model_path, "rb"))

    def P(self, word): 
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N
    
    def correction(self, word):
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word):
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        return set(w for w in words if w in self.WORDS)
    
    def edits1(self, word):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
class Frame(ctk.CTkFrame):
    def __init__(self, main, text="Video", side="left"):
        ctk.CTkFrame.__init__(self, main)
        self.setup(text=text, side=side)

    def setup(self, text, side):
        self.out = ctk.CTkLabel(self, text=text, font= ("sans-serif", 30))
        self.out.pack(fill= "both", expand=True, padx=5, pady=5)
        self.out.configure(anchor="center")
        self.img_label = ctk.CTkLabel(self, text="")
        self.img_label.pack(side=side, fill= "both", expand="yes", padx=10, pady=10)
    
    def img_update(self, img):
        self.img_label.configure(image=img)
        self.img = img
    
class TextFrame(ctk.CTkFrame):
    def __init__(self, main, text="Video", side="left"):
        ctk.CTkFrame.__init__(self, main)
        self.text = ""
        self.setup(text=text, side=side)

    def setup(self, text, side):
        self.out = ctk.CTkLabel(self, text=text, font= ("sans-serif", 30))
        self.out.pack(fill= "both", expand=True, padx=5, pady=5)
        self.out.configure(anchor="center")
        self.text_label = ctk.CTkTextbox(self, state = "disabled", width=500, height=500)
        self.text_label.pack(side=side, fill= "both", expand="yes", padx=10, pady=10)
    
    def text_update(self, txt):
        self.text += txt or ""
        self.text_label.configure(state="normal")
        self.text_label.delete(0.0, ctk.END)
        self.text_label.insert(ctk.END, self.text)

class Window:
    def __init__(self, model_path:str, bow_path:str, camId:int = 0):
        self.main = ctk.CTk()
        self.wdinfo = Struct({
            "init_size" : [1080,720],
            "current_size" : [1080,720],
            "min_size" : [920,720],
            "im_size" : [500,500]
        })


        self.videoBox = Frame(self.main,"Camera")
        self.videoBox.place(relx=0, rely=0.5, x=5, anchor="w")

        self.textBox = TextFrame(self.main, "Terjemahan")
        self.textBox.place(relx=1, rely=0.5, x=-10, anchor="e")

        self.videoCapt = Tools(model_path,bow_path,camId)

        self.main.geometry(f"{self.wdinfo.init_size[0]}x{self.wdinfo.init_size[1]}+0+0")
        self.main.minsize(self.wdinfo.min_size[0], self.wdinfo.min_size[1])
        self.main.iconbitmap("logo.ico")
        self.main.title("SIBI Translator")

        self.autocorrect = TextAutoCorrect()
        self.run = None
        self.main.wm_protocol("WM_DELETE_WINDOW", self.close)
        self.mostpred = []
        self.wordbag = ""
        self.charbef = None
        self.countk = 0
        self.main.after(100, self.loop)
        

    def convert(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.wdinfo.im_size[0], self.wdinfo.im_size[1]), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=(self.wdinfo.im_size[0], self.wdinfo.im_size[1]))
        
        return img
    
    def loop(self):
        hspred = 0
        img = self.videoCapt.frame_annot
        pred = self.videoCapt.pred or ""
        img_res = self.convert(img)
        self.videoBox.img_update(img_res)

        if len(pred) >0:
            self.mostpred.append(ord(pred.lower()))

            if len(self.mostpred) >=30:
                hspred = chr(int(np.bincount(self.mostpred).argmax()))
                if hspred != self.charbef:
                    self.wordbag += hspred
                    print(self.wordbag)
                    self.charbef = hspred
                self.mostpred = []
            self.countk = 0
        else:
            self.countk += 1
            if self.countk >=60:
                if len(self.wordbag) >=1:
                    tpred = self.autocorrect.correction(self.wordbag)
                    self.textBox.text_update(tpred + " ")

                    self.wordbag = ""
                self.countk = 0
        
        if self.run:
            self.main.after(14, self.loop)

    def close(self):
        self.run = False
        self.videoCapt.close()
        self.main.destroy()

    def start(self):
        self.run = True
        self.videoCapt.start()

        self.img_annot_bef = self.videoCapt.frame_annot
        self.videoBox.img_update(self.convert(self.img_annot_bef))
        self.main.mainloop()




