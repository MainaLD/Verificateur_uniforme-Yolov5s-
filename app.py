import numpy as np
import av
import cv2
import time
import torch
import queue

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,WebRtcMode

font = cv2.FONT_HERSHEY_SIMPLEX #typologie fenêtre détection

couleurs = np.array([[0.0, 255.0,0.0],[0.0,0.0,255.0]]) #en BGR : ROUGE=NON / VERT=OK

labels = ['Casque_OK','Casque_NON','Gilet_OK','Gilet_NON']
# 0 = Casque_OK  /  1 = Casque_NO  /  2 = Gilet  /  3 = Gilet_NO
messages = ["Aucune détection", "Uniforme vérifié", "Uniforme non vérifié"]

# Chargement du modèle
path = 'Models/best_v5s_b64.pt'
modele_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False) #False / True si error 403

CONFIDENCE_THRESHOLD = 0.75 # Seuil de confidence de la pred

def live_detection():

    class VideoProcessor(VideoProcessorBase):

        def __init__(self) -> None:
            self.valeur_detection = False
        
        def Affichage_message(self, image):
            boxes = []
            classes_ids = []
            resultats = modele_yolo(image)
            resultats_pred = resultats.pred[0]
            
            if not resultats:
                return image, False

            else:
                for i in range(0,len(resultats_pred)) :
                    if resultats_pred[i,4] > CONFIDENCE_THRESHOLD :
                        x = int(resultats_pred[i,0])
                        y = int(resultats_pred[i,1])
                        w = int(resultats_pred[i,2])
                        h = int(resultats_pred[i,3])
                        boite = np.array([x, y, w, h])
                        boxes.append(boite)
                        classes_ids.append(int(resultats_pred[i,5]))

                # Pour l'affichage des cadres sur l'image de la video
                for boite, classe_id in zip(boxes,classes_ids):
                    print("L61",boite, classe_id)
                    couleur = couleurs[int(classe_id) % len(couleurs)]
                    cv2.rectangle(image, boite, couleur, 2)
                    cv2.rectangle(image, (boite[0], boite[1] - 20), (boite[0] + boite[2], boite[1]), couleur, -1)
                    cv2.putText(image, labels[classe_id], (boite[0], boite[1] - 5), font, .5, (0,0,0))
                
                return image, classes_ids

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            image_annotation, resultats = self.Affichage_message(image)
            print("result = ", resultats)

            if resultats == False:
                return av.VideoFrame.from_ndarray(image_annotation, format="bgr24")
            else:
                self.valeur_detection = resultats
                return av.VideoFrame.from_ndarray(image_annotation, format="bgr24")
    
    champs_message = st.empty()
    champs_message.info('En attente ...')

    stream = webrtc_streamer(
            key="mute_sample",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            )

    while True:
        time.sleep(1)
        champs_message.empty()

        if stream.video_processor:
            try:
                ancien_detection = stream.video_processor.valeur_detection
            except queue.Empty:
                ancien_detection = None

            if not ancien_detection :
                print("PAS DE DETECTION")
                champs_message.info(messages[0])
            elif 1 in ancien_detection or 3 in ancien_detection:
                print('INCOMPLET')
                champs_message.error(messages[2])
            else:
                print('COMPLET')
                champs_message.success(messages[1])


if __name__ == "__main__":
    st.title("Vérificateur d'uniforme 👷")
    st.write("Cette application à pour but de vérifier si la personne porte un uniforme complet : casque + gilet")
    verification_uniforme = live_detection()
    