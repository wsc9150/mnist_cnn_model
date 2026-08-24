# ============================================================
# 웹캠 숫자 인식기 - PyTorch 버전
# ============================================================

import os

import streamlit as st

import numpy as np
import cv2

from PIL import Image, ImageOps

from scipy.ndimage import center_of_mass, shift

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 모델 설정
# ------------------------------------------------------------
MODEL_DIR = "saved_models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# CNN 모델
# ------------------------------------------------------------
class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------------------------
# 최신 모델 찾기
# ------------------------------------------------------------
def get_latest_model():

    if not os.path.exists(MODEL_DIR):
        return None

    models = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pth")
    ]

    if not models:
        return None

    models.sort(reverse=True)

    return os.path.join(
        MODEL_DIR,
        models[0]
    )


# ------------------------------------------------------------
# 모델 로드
# ------------------------------------------------------------
model_path = get_latest_model()

model = None

if model_path:

    model = MNISTCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()


# ------------------------------------------------------------
# 전처리
# ------------------------------------------------------------
def preprocess_image(image_arr):

    # --------------------------------------------------------
    # RGB → Grayscale인 경우 그대로 사용
    # --------------------------------------------------------
    gray = image_arr

    # 대비 강화
    norm = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    # Gaussian Blur
    blurred = cv2.GaussianBlur(
        norm,
        (5, 5),
        0
    )

    # 흰 종이 + 검정 펜
    # MNIST는 검정 배경 + 흰색 숫자이므로 반전한다.
    inverted = cv2.bitwise_not(
        blurred
    )

    # Otsu 이진화
    _, binary = cv2.threshold(
        inverted,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 숫자가 없는 경우
    if np.sum(binary > 0) < 20:
        return None

    # 중심 계산
    cy, cx = center_of_mass(binary)

    shift_y = int(
        round(binary.shape[0] // 2 - cy)
    )

    shift_x = int(
        round(binary.shape[1] // 2 - cx)
    )

    shifted = shift(
        binary,
        shift=(shift_y, shift_x),
        mode="constant",
        cval=0
    )

    # 28x28
    resized = cv2.resize(
        shifted.astype("uint8"),
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # 0~1
    input_arr = (
        resized.astype("float32") / 255.0
    )

    # PyTorch:
    # [Batch, Channel, Height, Width]
    tensor = torch.from_numpy(
        input_arr
    ).unsqueeze(0).unsqueeze(0)

    return resized, tensor


# ------------------------------------------------------------
# 예측
# ------------------------------------------------------------
def predict(tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.softmax(
            output,
            dim=1
        )[0]

        pred_class = int(
            probability.argmax().item()
        )

    return (
        pred_class,
        probability.cpu().numpy()
    )


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="웹캠 숫자 인식기",
    layout="centered"
)

st.title(
    "웹캠 숫자 인식기 (MNIST + PyTorch)"
)

st.markdown(
    "흰 종이에 검은색 펜으로 0~9 숫자를 "
    "작성한 후 웹캠으로 촬영합니다."
)


if model is None:

    st.warning(
        "PyTorch 모델(.pth)이 없습니다. "
        "먼저 모델을 학습해주세요."
    )

else:

    image_data = st.camera_input(
        "숫자가 보이도록 웹캠으로 촬영"
    )

    if image_data is not None:

        image = Image.open(
            image_data
        ).convert("RGB")

        st.image(
            image,
            caption="입력 이미지",
            width=500
        )

        # RGB → Grayscale
        gray = ImageOps.grayscale(
            image
        )

        gray_np = np.array(gray)

        processed = preprocess_image(
            gray_np
        )

        if processed is None:

            st.error(
                "숫자 영역을 찾지 못했습니다."
            )

        else:

            resized, tensor = processed

            pred_class, probability = (
                predict(tensor)
            )

            # ----------------------------------------------
            # 예측 결과
            # ----------------------------------------------
            st.subheader(
                f"예측된 숫자: **{pred_class}**"
            )

            st.bar_chart(
                probability
            )

            # ----------------------------------------------
            # 전처리 이미지
            # ----------------------------------------------
            st.subheader(
                "전처리된 MNIST 입력 이미지"
            )

            st.image(
                resized,
                width=150,
                clamp=True
            )
