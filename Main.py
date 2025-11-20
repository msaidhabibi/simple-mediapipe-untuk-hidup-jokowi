import cv2
import mediapipe as mp
import math
import pygame

def jarak(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

# Inisialisasi mixer pygame untuk suara
pygame.mixer.init()
pygame.mixer.music.load("mengepal.mp3")

is_playing = False

# Inisialisasi modul MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Konfigurasi deteksi tangan
hands = mp_hands.Hands()#(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# Gunakan webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera")
        break

    # flip horizontal
    img = cv2.flip(img, 1)
    # Konversi BGR (OpenCV) ke RGB (MediaPipe)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    mengepal = False
    # Jika ada tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik dan garis tangan
            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Contoh: ambil koordinat ujung jari telunjuk (index finger tip = landmark 8)
            h, w, c = img.shape
            
            # ambil koordinat ujung jari (tip) dan pangkal jari (pip)
            wrist = hand_landmarks.landmark[0]
            finger_tip = [4,8,12,16,20]
            
            distances = []
            for tip in finger_tip:
                d = jarak(hand_landmarks.landmark[tip], wrist, w, h)
                distances.append(d)
            # finger_pip = [2,6,10,14,18]
            if all(d < 100 for d in distances):
                mengepal = True
                cv2.putText(img, "Tangan Mengepal", (50, 50), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                # while mengepal:
                #     pygame.mixer.init()
                #     pygame.mixer.music.load("mengepal.mp3")
                #     pygame.mixer.music.play()
                #     mengepal = False
                
            # cek dengan toleran apakah semua jari mengepal
            # toleransi = 0.1
            # mengepal = 0
            # for tip, pip in zip(finger_tip,finger_pip):
            #     tip_y = hand_landmarks.landmark[tip].y 
            #     pip_y = hand_landmarks.landmark[pip].y 
                
            #     # jika ujung jari lebih rendah dari pangkal jari ==> dianggal mengepal
            #     if tip_y > pip_y: #toleransi:
            #         mengepal += 1
            
            # if mengepal == 5:
            #     cv2.putText(img, "Tangan Mengepal", (50, 50), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
    if mengepal and not is_playing:
        pygame.mixer.music.play(-1) # -1 loop terus menerus
        is_playing = True
    elif not mengepal and is_playing:
        pygame.mixer.music.stop()
        is_playing = False

    # Tampilkan hasil
    cv2.imshow("Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
