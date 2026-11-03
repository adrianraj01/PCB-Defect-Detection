
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 PCB Defect Detection System")
st.write("Upload a reference PCB image and a test PCB image to detect defects.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference PCB")
    reference_file = st.file_uploader(
        "Upload defect-free PCB image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

with col2:
    st.subheader("Test PCB")
    test_file = st.file_uploader(
        "Upload PCB image to inspect",
        type=["jpg", "jpeg", "png"],
        key="test"
    )

if reference_file and test_file:

    ref_bytes = np.frombuffer(reference_file.read(), np.uint8)
    test_bytes = np.frombuffer(test_file.read(), np.uint8)

    reference = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
    test = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    if reference is None or test is None:
        st.error("Could not read one of the images.")
    elif reference.shape != test.shape:
        st.error("Both images must have the same dimensions.")
    else:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        difference = cv2.absdiff(ref_gray, test_gray)

        _, threshold = cv2.threshold(
            difference, 30, 255, cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)
        threshold = cv2.morphologyEx(
            threshold, cv2.MORPH_OPEN, kernel
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = test.copy()
        defect_count = 0

        for contour in contours:
            if cv2.contourArea(contour) > 20:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    output, (x, y), (x+w, y+h),
                    (0, 0, 255), 2
                )
                defect_count += 1

        st.subheader("Detection Results")

        c1, c2, c3 = st.columns(3)

        c1.image(reference, channels="BGR",
                 caption="Reference PCB")
        c2.image(test, channels="BGR",
                 caption="Test PCB")
        c3.image(output, channels="BGR",
                 caption="Detected Differences")

        st.metric("Potential Defect Regions", defect_count)

        if defect_count > 0:
            st.warning("Potential differences detected!")
        else:
            st.success("No significant differences detected.")
