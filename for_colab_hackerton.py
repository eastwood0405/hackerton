from ultralytics import YOLO
import cv2
import numpy as np
import time

class PhoneDetector:
    def __init__(self, model_path="./models/best.pt"):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.02, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 67:
                    return True, box.xyxy[0]
        return False, None

class FaceAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # ✅ 연속 미감지 프레임 카운터 & 이전 상태 저장
        self.no_face_counter = 0
        self.AWAY_THRESHOLD = 15   # 연속 15프레임(약 0.5초) 이상 얼굴 없을 때만 AWAY
        self.last_status = "STUDYING"

    def analyze(self, frame):
        # 원본은 건드리지 않고 복사본만 축소
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            small = cv2.resize(frame, (640, int(h * scale)))
        else:
            scale = 1.0
            small = frame.copy()

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )

        # ✅ 얼굴 미감지 시: 카운터 증가, 임계값 미달이면 이전 상태 유지
        if len(faces) == 0:
            self.no_face_counter += 1
            if self.no_face_counter < self.AWAY_THRESHOLD:
                return self.last_status, self.last_status == "STUDYING", frame
            self.last_status = "AWAY"
            return "AWAY", False, frame

        # 얼굴 감지 성공 시 카운터 리셋
        self.no_face_counter = 0

        # 좌표를 원본 크기로 역변환해서 원본 frame에 그리기
        for (x, y, w2, h2) in faces:
            ox, oy = int(x / scale), int(y / scale)
            ow, oh = int(w2 / scale), int(h2 / scale)
            cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (255, 255, 0), 2)
            cv2.putText(frame, "Face", (ox, oy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        (x, y, w2, h2) = faces[0]
        roi_gray = gray[y:y+h2, x:x+w2]

        # 얼굴 ROI 상단 절반만 탐색 (콧구멍/코 오인식 방지)
        roi_top_half = roi_gray[:h2//2, :]

        eyes = self.eye_cascade.detectMultiScale(
            roi_top_half, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            ox = int((x + ex) / scale)
            oy = int((y + ey) / scale)
            ow, oh = int(ew / scale), int(eh / scale)
            cv2.rectangle(frame, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 1)

        cv2.putText(frame, f"Eyes: {len(eyes)}", (int(x/scale), int((y+h2)/scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(eyes) < 2:
            self.last_status = "SLEEPING"
            return "SLEEPING", False, frame

        self.last_status = "STUDYING"
        return "STUDYING", True, frame


# 객체 생성
phone_engine = PhoneDetector("./models/best.pt")
face_engine = FaceAnalyzer()
cap = cv2.VideoCapture("test_video.mp4")

# ✅ 영상 저장 설정
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_video_output.mp4", fourcc, fps, (width, height))

print("FocusGuardian started - processing...")
away_start_time = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. 핸드폰 체크
    has_phone, phone_box = phone_engine.detect(frame)

    # 2. 얼굴 상태 체크 (frame에 직접 시각화 포함)
    status, is_studying, frame = face_engine.analyze(frame)

    # 3. 핸드폰 감지 시 빨간 박스
    if has_phone and phone_box is not None:
        x1, y1, x2, y2 = phone_box.tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, "PHONE", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        away_start_time = None

    # 4. 상태 텍스트 표시
    color_map = {"STUDYING": (0, 255, 0), "SLEEPING": (0, 165, 255), "AWAY": (0, 0, 255)}
    color = color_map.get(status, (255, 255, 255))
    cv2.putText(frame, f"Status: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # ✅ 처리된 프레임을 출력 영상에 저장
    out.write(frame)

    # 💡 [지시사항] 코랩 에러 방지를 위해 화면 표시 부분만 주석 처리했습니다.
    # cv2.imshow("Guardian", frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q') or key == 27:
    #     break

# cleanup
cap.release()
out.release()  # ✅ 저장 완료
# cv2.destroyAllWindows() # 💡 화면 창 닫기 주석 처리

print("Program exited successfully.")
print("저장 완료: test_video_output.mp4")