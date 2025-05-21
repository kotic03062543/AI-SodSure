import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import io

# โหลดโมเดล
@st.cache_resource
def load_sodsure_model():
    model = load_model('meat_model.h5')
    return model

model = load_sodsure_model()
class_names = ['สด', 'เน่า']

# หน้า UI
st.title("🥩 SodSure AI - ตรวจสอบความสดของเนื้อ")
st.write("อัปโหลดรูปเนื้อ แล้ว AI จะช่วยทำนายว่าเนื้อสดหรือเน่า พร้อมความมั่นใจเป็นเปอร์เซ็นต์!")
uploaded_file = st.file_uploader("📤 อัปโหลดรูปเนื้อ (JPEG/JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # แสดงรูป
    img = Image.open(uploaded_file)
    st.image(img, caption='รูปที่คุณอัปโหลด', use_column_width=True)

    # เตรียมภาพ
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # พยากรณ์
    prediction = model.predict(img_array)
    pred = float(prediction.squeeze())  # โอกาสว่าเน่า
    freshness_percent = (1 - pred) * 100
    predicted_class = class_names[int(pred < 0.5)]

    # แสดงผล
    st.subheader("🔍 ผลการวิเคราะห์")
    st.write(f"💡 **ความสดโดยประมาณ: {freshness_percent:.2f}%**")
    if freshness_percent >= 50:
        st.success(f"✅ คาดว่าเนื้อ **'สด'** (ความมั่นใจ {freshness_percent:.2f}%)")
    else:
        st.error(f"⚠️ คาดว่าเนื้อ **'เน่า'** (ความมั่นใจ {100 - freshness_percent:.2f}%)")
