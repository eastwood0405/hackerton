# hackerton
해커톤 작품을 위한 리포2
FocusGuardian 설치 가이드


#1. 필요한 파일
FocusGuardian/
├── newnew.py              ← 메인 코드
├── test_video.mp4         ← 테스트 영상
└── models/
    └── face_landmarker.task  ← 첫 실행 시 자동 다운로드
    (best.pt 은 커스텀 모델 있으면 추가, 없으면 yolov8n.pt 자동 다운로드)
models/ 폴더는 비워둬도 되고, 첫 실행 시 자동으로 채워집니다./


#2. 필요한 환경
항목버전Python3.11.x (3.12 이상 ❌)mediapipe0.10.9ultralytics최신opencv-python최신numpy최신

#3. 설치 순서
#Step 1 — Python 3.11 설치
https://www.python.org/downloads/release/python-3119/
Windows installer (64-bit) 다운로드 후 설치

⚠️ "Add python.exe to PATH" 체크 해제 (기존 Python 보호)

#Step 2 — 프로젝트 폴더 준비

폴더 생성
mkdir D:\FocusGuardian
cd D:\FocusGuardian

파일 복사 (newnew.py, test_video.mp4)

#Step 3 — 가상환경 생성 및 활성화

3.11로 가상환경 생성
py -3.11 -m venv .venv

활성화 (Windows)
.venv\Scripts\activate

활성화 확인 — 앞에 (.venv) 표시되면 성공

#Step 4 — 라이브러리 설치
pip install mediapipe==0.10.9 ultralytics opencv-python numpy

#Step 5 — 실행
bashpython newnew.py
# 첫 실행 시 face_landmarker.task 자동 다운로드 (약 30MB)


#6. 매번 실행할 때 순서
bash# 1. 프로젝트 폴더로 이동
cd D:\FocusGuardian

 가상환경 활성화
.venv\Scripts\activate

 실행
python newnew.py
