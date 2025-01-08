import os
import cv2
import sys
from ultralytics import YOLO

# OpenMP 에러 해결을 위한 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLOv8 모델 로드
def load_yolov8_model():
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano 모델 사용
    return model

# YOLOv8 모델 훈련
def train_model(data_yaml, epochs=50, img_size=640):
    print("YOLOv8 모델 훈련 시작...")
    model = load_yolov8_model()
    try:
        model.train(
            data=data_yaml,  # 데이터셋 yaml 경로
            epochs=epochs,
            imgsz=img_size,
            project="D:/AI3/study/abc/images/test",
            name="yolov8_trained",
            device="cpu"  # GPU 사용 가능하면 'cuda'로 변경
        )
        print("모델 훈련이 완료되었습니다.")
    except Exception as e:
        print(f"모델 훈련 중 오류 발생: {e}")

# YOLOv8 모델 추론
def run_inference(model_path, input_path, output_dir):
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)

    print(f"추론 중: {input_path}")
    results = model.predict(
        source=input_path,  # 입력 이미지 또는 비디오 경로
        save=True,          # 결과 저장 활성화
        save_dir=output_dir # 결과 저장 디렉토리
    )
    print(f"추론 결과가 저장되었습니다: {output_dir}")

# YOLOv8 실시간 웹캠 탐지 및 데이터 저장
def run_webcam_detection():
    print("Starting YOLOv8 webcam detection...")

    # YOLOv8 모델 로드
    model = load_yolov8_model()
    model.conf = 0.5  # 신뢰도 임계값 설정

    # 저장 경로 설정
    base_dir = input("저장할 디렉토리 경로를 입력하세요 (예: D:/AI3/study/abc/dataset): ").strip()
    if not base_dir:
        base_dir = "D:/AI3/study/abc/dataset"  # 기본 저장 디렉토리
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"훈련 데이터는 {train_dir}에, 검증 데이터는 {val_dir}에 저장됩니다.")

    # YAML 파일 생성
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(f"train: {train_dir}\n")
        yaml_file.write(f"val: {val_dir}\n")
        yaml_file.write(f"nc: 1\n")
        yaml_file.write(f"names: ['person']\n")

    # 웹캠 열기
    cap = cv2.VideoCapture(0)  # 0번 카메라 (기본 웹캠)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        sys.exit()

    print("객체 탐지 중... 캡처하려면 's' 키를 누르세요. 훈련하려면 't' 키를 누르세요. 추론하려면 'i' 키를 누르세요. 종료하려면 'q' 키를 누르세요.")

    frame_count = 0  # 이미지 저장 인덱스

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # YOLOv8로 탐지 수행
        results = model.predict(source=frame, save=False, verbose=False)

        # 검출된 객체 처리
        detections = results[0].boxes.data.cpu().numpy()  # [[x1, y1, x2, y2, confidence, class], ...]
        label_data = []
        person_count = 0  # 탐지된 사람 수

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf >= 0.5:  # 사람 클래스 (ID: 0) 및 신뢰도 조건
                person_count += 1
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"Person {person_count}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 라벨 데이터 준비 (YOLO 형식: class x_center y_center width height)
                w, h = frame.shape[1], frame.shape[0]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                label_data.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # 탐지된 사람 수 표시
        count_text = f"Total Persons: {person_count}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 화면 표시
        cv2.imshow('YOLOv8 Multi-Person Detection', frame)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # 's' 키를 눌렀을 때 데이터 저장
            save_dir = train_dir if frame_count % 5 != 0 else val_dir  # 5:1 비율로 검증 데이터 분리
            img_path = os.path.join(save_dir, f"image_{frame_count}.jpg")
            label_path = os.path.join(save_dir, f"image_{frame_count}.txt")
            cv2.imwrite(img_path, frame)
            print(f"이미지 저장 완료: {img_path}")

            # 라벨 저장
            with open(label_path, "w") as f:
                f.write("\n".join(label_data))
            print(f"라벨 저장 완료: {label_path}")

            frame_count += 1

        elif key == ord('t'):  # 't' 키를 눌렀을 때 훈련 시작
            train_model(yaml_path, epochs=50)

        elif key == ord('i'):  # 'i' 키를 눌렀을 때 추론 시작
            input_path = input("추론할 이미지 또는 비디오 경로를 입력하세요: ").strip()
            output_dir = os.path.join(base_dir, "inference_results")
            os.makedirs(output_dir, exist_ok=True)
            run_inference("D:/AI3/study/abc/images/test/yolov8_trained/weights/best.pt", input_path, output_dir)

        elif key == ord('q'):  # 'q' 키를 눌렀을 때 종료
            print("프로그램을 종료합니다.")
            break

    # 웹캠 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 실행 흐름
run_webcam_detection()
