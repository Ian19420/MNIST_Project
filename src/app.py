import os
#設定環境變數 告訴TensorFlow不要去檢查GPU 直接用CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import pandas as pd

#頁面設定
st.set_page_config(
    page_title="手寫數字辨識系統",
    layout="wide",
)

#CSS設定
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

#載入模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'mnist_model.keras')

@st.cache_resource #加入快取
def load_my_model():
    return tf.keras.models.load_model(model_path, compile=False)


model = load_my_model()

#主畫面
st.title("AI手寫數字辨識系統")
st.markdown("#### 請在左側畫布寫下 **0~9** 的數字，右側將即時顯示 AI 推論結果。")
st.divider()

#檢查模型
if model is None:
    st.error("錯誤：找不到模型檔案！")
    st.stop()

#核心介面：左右分欄
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("手寫輸入區")
    #畫布
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=25,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )
    #按鈕
    predict_btn = st.button('開始辨識', type="primary")

with col2:
    st.subheader("AI 辨識結果")

    #canvas有東西且按鈕被按下 就執行
    if canvas_result.image_data is not None and predict_btn:

        #影像處理區塊
        #轉灰階
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img_array = np.array(img)
        #反轉顏色
        if img_array.mean() > 127:
            img_array = 255 - img_array
        # 找出邊框
        rows, cols = np.where(img_array > 20)
        if len(rows) > 0:
            #裁切
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)
            crop_img = img_array[min_r:max_r + 1, min_c:max_c + 1]
            #縮放並保持比例
            h, w = crop_img.shape
            scale = 20.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            #置中貼回 28x28
            final_img = np.zeros((28, 28), dtype=np.uint8)
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
            final_img[top:top + new_h, left:left + new_w] = resized_img
            #二值化增強
            _, final_img = cv2.threshold(final_img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            final_img = np.zeros((28, 28), dtype=np.uint8)

        #預測區塊
        img_input = final_img.astype('float32') / 255.0
        img_input = img_input.reshape(1, 28, 28, 1)

        prediction = model.predict(img_input)
        final_number = np.argmax(prediction)
        confidence = np.max(prediction)

        #顯示結果區塊
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="預測數字", value=str(final_number))
        with res_col2:
            st.metric(label="信心水準", value=f"{confidence * 100:.1f}%", delta_color="normal")

        st.write("信心度指標：")
        st.progress(float(confidence))

        st.write("各數字機率分佈：")
        chart_data = pd.DataFrame(
            prediction[0],
            index=[str(i) for i in range(10)],
            columns=['機率']
        )
        st.bar_chart(chart_data)

        with st.expander("查看模型輸入影像"):
            st.write("這是進入 CNN 模型的影像：")
            st.image(final_img, width=150, caption="28x28 Preprocessed Image")

    elif not predict_btn:
        st.info("請在左側畫圖，並按下「開始辨識」")


st.divider()
st.subheader("模型效能評估報告(Model Evaluation)")
report_csv_path = os.path.join(current_dir, '..', 'models', 'classification_report.csv')

with st.expander("點擊展開詳細數據"):
    #檢查檔案在不在
    if os.path.exists(report_csv_path):
        #讀取CSV
        df_report = pd.read_csv(report_csv_path, index_col=0)
        st.write("本模型在測試集上的分類詳細數據：")

        #顯示表格
        st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)

        #額外顯示準確度
        if 'accuracy' in df_report.index:
            acc = df_report.loc['accuracy', 'f1-score']
            st.metric("整體準確度 (Accuracy)", f"{acc:.2%}")