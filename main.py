import cv2
import time
import math as m
import mediapipe as mp
import serial
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 아두이노 연결
arduino = None
try:
    arduino = serial.Serial('/dev/cu.usbmodem1101', 9600, timeout=1) 
    time.sleep(2)  # 아두이노 초기화 대기
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    arduino = None

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 카메라 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 경고음 설정
warning_file = "경고음.m4a"

# 그래프 데이터 초기화
fig = plt.figure(figsize=(14, 8))  # 전체 창 크기 설정
gs = fig.add_gridspec(10, 6)

# 웹캠 화면
ax1 = fig.add_subplot(gs[:6, :])  # 웹캠 화면 크게 표시
ax1.set_title("Webcam View")

# 그래프
ax2 = fig.add_subplot(gs[6:, 1:5])  # 아래쪽에 그래프 표시
ax2.set_title("Neck Angle and Arduino Data")

# Study Time 및 경고 횟수 텍스트
ax_timer = fig.add_subplot(gs[6:, 0])  # 그래프 왼쪽에 Study Time 표시
ax_timer.axis('off')  # 축 숨기기

# 데이터 초기화
angle_data = []
arduino_data = []
time_data = []
start_time = time.time()
warning_count = 0  # 경고음 횟수 초기화

# 경고 설정
angle_threshold = 40  # 목 각도 임계값
arduino_threshold = 30  # 아두이노 휨 센서 임계값 (30으로 변경)
alert_duration = 3.5  # 경고음 울리기 전 초과 지속 시간
alert_start_time_angle = None  # 목 각도의 경고 시작 시간
alert_start_time_arduino = None  # 아두이노의 경고 시작 시간

# 각도 계산 함수
def findAngle(x1, y1, x2, y2):
    try:
        theta = m.atan2(y2 - y1, x2 - x1)  # 기울기 기반 각도 계산
        degree = abs(theta * (180 / m.pi))  # 각도를 절대값으로 반환
    except Exception:
        degree = 0  # 계산 불가 시 0도 반환
    
    if degree <= 120:
        degree -= 100
        degree = abs(degree)
    elif degree < 130:
        degree = degree - 100 + 10
    else:
        degree = degree - 100 + 20

    return degree

# Study Time 표시 형식 함수
def format_time(elapsed_seconds):
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return f"{minutes:02}:{seconds:02}"

# 실시간 업데이트 함수
def update_graph(frame_idx):
    global angle_data, arduino_data, time_data, alert_start_time_angle, alert_start_time_arduino, warning_count

    # 아두이노 데이터 읽기
    arduino_value = 0
    if arduino:
        try:
            data = arduino.readline().decode('utf-8').strip()
            arduino_value = int(data) if data.isdigit() else 0
        except Exception:
            arduino_value = 0

    # 카메라 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    neck_angle = 0  # 초기값 설정
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 랜드마크 좌표 추출
        def get_coords(landmark):
            return landmark.x, landmark.y, landmark.visibility

        l_shldr = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_shldr = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_ear = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_EAR])
        r_ear = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])

        # 신뢰도 기준 필터링
        if r_shldr[2] > 0.5 and r_ear[2] > 0.5:
            neck_angle = findAngle(r_shldr[0], r_shldr[1], r_ear[0], r_ear[1])
        elif l_shldr[2] > 0.5 and l_ear[2] > 0.5:
            neck_angle = findAngle(l_shldr[0], l_shldr[1], l_ear[0], l_ear[1])

    # 각도 및 시간 데이터 추가
    elapsed_time = time.time() - start_time
    angle_data.append(neck_angle)
    arduino_data.append(arduino_value)
    time_data.append(elapsed_time)

    # 최근 1분 데이터 유지
    time_limit = 60
    while time_data and (elapsed_time - time_data[0] > time_limit):
        time_data.pop(0)
        angle_data.pop(0)
        arduino_data.pop(0)

    # 경고음 로직 (목 각도와 아두이노 값 둘 다 동시에 충족되어야 경고음 울림)
    if neck_angle > angle_threshold:
        if alert_start_time_angle is None:
            alert_start_time_angle = time.time()
        elif time.time() - alert_start_time_angle >= alert_duration:
            # 목 각도가 40도를 초과하고 3.5초 이상 유지됨
            pass  # 이 시점에서 추가 경고음은 발생하지 않음
    else:
        alert_start_time_angle = None  # 목 각도가 40도 이하로 떨어지면 초기화

    if arduino_value > arduino_threshold:
        if alert_start_time_arduino is None:
            alert_start_time_arduino = time.time()
        elif time.time() - alert_start_time_arduino >= alert_duration:
            # 아두이노 휨 센서 값이 30을 초과하고 3.5초 이상 유지됨
            pass  # 이 시점에서 추가 경고음은 발생하지 않음
    else:
        alert_start_time_arduino = None  # 아두이노 값이 30 이하로 떨어지면 초기화

    # 두 조건이 모두 충족되었을 때 경고음 울리기
    if alert_start_time_angle is not None and alert_start_time_arduino is not None:
        if (time.time() - alert_start_time_angle >= alert_duration) and (time.time() - alert_start_time_arduino >= alert_duration):
            subprocess.call(["afplay", warning_file])  # 경고음 재생
            warning_count += 1  # 경고 횟수 증가
            alert_start_time_angle = None  # 초기화
            alert_start_time_arduino = None  # 초기화

    # 그래프 업데이트
    ax2.clear()
    ax2.plot(time_data, angle_data, label="Neck Angle (°)", color='blue')
    ax2.plot(time_data, arduino_data, label="Arduino Data", color='red', linestyle='--')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper left")

    # Study Time 및 Warning Count 업데이트
    timer_text = f"Study Time\n{format_time(elapsed_time)}\n\nWarnings: {warning_count}"
    ax_timer.clear()
    ax_timer.text(0.25, 0.5, timer_text, ha='center', va='center', fontsize=16, fontweight='bold')
    ax_timer.axis('off')

    # 웹캠 화면 표시
    ax1.clear()
    ax1.imshow(frame_rgb)
    ax1.axis('off')

# 애니메이션 초기화
ani = FuncAnimation(fig, update_graph, interval=100)

# 그래프와 웹캠 화면 출력
plt.show()

# 자원 해제
cap.release()
if arduino:
    arduino.close()
