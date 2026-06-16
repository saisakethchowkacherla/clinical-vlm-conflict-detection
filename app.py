import streamlit as st

st.set_page_config(page_title="Clinical VLM Conflict Detection")

st.title("🏥 Clinical VLM Conflict Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Chest X-ray", width=300)

patient = st.text_input("Patient Name")

image_pred = st.selectbox(
    "Image Only Prediction",
    ["COVID-19", "Normal", "Pneumonia", "Lung Opacity"]
)

clinical_pred = st.selectbox(
    "Image + Clinical Data Prediction",
    ["COVID-19", "Normal", "Pneumonia", "Lung Opacity"]
)

if st.button("Analyze"):

    st.subheader("Analysis Results")

    st.write("Patient:", patient)
    st.write("Image Prediction:", image_pred)
    st.write("Multimodal Prediction:", clinical_pred)

    if image_pred != clinical_pred:
        st.error("⚠️ CONFLICT DETECTED")
        st.write("Additional clinical data changed the diagnosis.")
    else:
        st.success("✅ NO CONFLICT")
        st.write("Predictions remain consistent.")