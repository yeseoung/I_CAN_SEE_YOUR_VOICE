import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

# 설정
actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ',
    'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ',
    'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
    ]

seq_length = 10

# macOS 한글 폰트
fontpath = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font = ImageFont.truetype(fontpath, 40)


def initialize_detector_and_model():
    """MediaPipe 홀리스틱 모델과 TFLite 모델 초기화"""

    detector = hm.HolisticDetector(
        min_detection_confidence=0.3
    )

    interpreter = tf.lite.Interpreter(
        model_path="models/multi_hand_gesture_classifier.tflite"
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("========== TFLite Model ==========")
    print("Input shape :", input_details[0]['shape'])
    print("Input dtype :", input_details[0]['dtype'])
    print("Output shape:", output_details[0]['shape'])
    print("Output dtype:", output_details[0]['dtype'])
    print("==================================")

    return detector, interpreter


def process_hand_landmarks(right_hand_lmList):
    """손 랜드마크 처리 및 벡터 정규화"""

    joint = np.zeros((21, 2), dtype=np.float32)

    for j, lm in enumerate(right_hand_lmList.landmark):
        if j >= 21:
            break

        joint[j] = [lm.x, lm.y]

    vector, angle_label = Vector_Normalization(joint)

    result = np.concatenate([
        vector.flatten(),
        angle_label.flatten()
    ])

    return result.astype(np.float32)


def predict_action(interpreter, input_data):
    """TFLite 모델을 사용하여 동작 예측"""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    y_pred = interpreter.get_tensor(
        output_details[0]['index']
    )

    return y_pred[0]


def draw_text_on_image(img, text):
    """OpenCV 이미지에 한글 텍스트 출력"""

    img_pil = Image.fromarray(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(img_pil)

    draw.text(
        (10, 30),
        text,
        font=font,
        fill=(255, 255, 255)
    )

    return cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR
    )


def main():

    detector, interpreter = initialize_detector_and_model()

    # MacBook 내장 카메라
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    seq = []
    action_seq = []

    last_action = None
    current_action = '?'
    current_conf = 0.0

    print("카메라 시작")
    print("ESC를 누르면 종료됩니다.")

    while cap.isOpened():

        ret, img = cap.read()

        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전
        img = cv2.flip(img, 1)

        # MediaPipe Holistic
        img = detector.findHolistic(
            img,
            draw=True
        )

        _, right_hand_lmList = detector.findRighthandLandmark(img)

        # 손이 검출된 경우
        if right_hand_lmList is not None:

            d = process_hand_landmarks(
                right_hand_lmList
            )

            seq.append(d)

            if len(seq) > seq_length:
                seq = seq[-seq_length:]

            # 10프레임 이상 모이면 추론
            if len(seq) >= seq_length:

                input_data = np.expand_dims(
                    np.array(
                        seq[-seq_length:],
                        dtype=np.float32
                    ),
                    axis=0
                )

                try:

                    y_pred = predict_action(
                        interpreter,
                        input_data
                    )

                    i_pred = int(
                        np.argmax(y_pred)
                    )

                    conf = float(
                        y_pred[i_pred]
                    )

                    current_conf = conf

                    if conf >= 0.9:

                        action = actions[i_pred]

                        action_seq.append(action)

                        if len(action_seq) > 5:
                            action_seq = action_seq[-5:]

                        # 같은 동작 3회 연속 확인
                        if len(action_seq) >= 3:

                            if (
                                action_seq[-1]
                                == action_seq[-2]
                                == action_seq[-3]
                            ):

                                this_action = action

                                if last_action != this_action:
                                    last_action = this_action

                                    print(
                                        f"인식: {this_action} "
                                        f"({conf:.2f})"
                                    )

                                current_action = this_action

                    else:
                        current_action = '?'

                except Exception as e:

                    print(
                        "모델 추론 오류:",
                        e
                    )

        # 화면에 결과 표시
        display_text = f"{current_action} ({current_conf:.2f})"

        img = draw_text_on_image(
            img,
            display_text
        )

        cv2.imshow(
            'Korean Sign Language Recognition',
            img
        )

        # ESC 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

