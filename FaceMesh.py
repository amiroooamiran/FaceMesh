# import opencv python
import cv2
# import mediapipe
import mediapipe as mp
# import time
import time

# open cam
cap = cv2.VideoCapture(0)


# model  for face mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces= 2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

prsTime = 0

while True:
    success, img = cap.read()

    # in the while FaceMesh models
    results = faceMesh.process(img)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    # fps control
    crTime = time.time()
    fps_rate = 1 / (crTime - prsTime)
    prsTime = crTime
                                                                                  # B ,  G ,  R
    cv2.putText(img, f'fps:{int(fps_rate)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("face mesh", img)
    cv2.waitKey(1)