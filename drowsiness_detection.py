import cv2
import dlib
from scipy.spatial import distance

def compute_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            COUNTER = 0

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
