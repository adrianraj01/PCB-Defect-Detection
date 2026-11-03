import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 PCB Defect Detection System")
st.write(
    "Upload a reference PCB image and a test PCB image "
    "to detect and label potential defects."
)

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
    ref_bytes = np.frombuffer(reference_file.getvalue(), np.uint8)
    test_bytes = np.frombuffer(test_file.getvalue(), np.uint8)

    reference = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
    test = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    if reference is None or test is None:
        st.error("Could not read one of the images.")

    elif reference.shape != test.shape:
        st.error(
            "Both images must have the same dimensions. "
            "Please resize the test image to match the reference."
        )

    else:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        # Align the test image to the reference using feature matching
        orb = cv2.ORB_create(nfeatures=3000)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        kp_test, des_test = orb.detectAndCompute(test_gray, None)

        aligned_test = test_gray.copy()

        if des_ref is not None and des_test is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = matcher.knnMatch(des_test, des_ref, k=2)

            good_matches = [
                m for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]

            if len(good_matches) >= 10:
                points_test = np.float32(
                    [kp_test[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                points_ref = np.float32(
                    [kp_ref[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                matrix, _ = cv2.findHomography(
                    points_test,
                    points_ref,
                    cv2.RANSAC,
                    5.0
                )

                if matrix is not None:
                    aligned_test = cv2.warpPerspective(
                        test_gray,
                        matrix,
                        (ref_gray.shape[1], ref_gray.shape[0])
                    )

        # Blur slightly to reduce noise from tiny pixel changes
        ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        test_blur = cv2.GaussianBlur(aligned_test, (5, 5), 0)

        difference = cv2.absdiff(ref_blur, test_blur)

        # Ignore small brightness differences
        _, threshold = cv2.threshold(
            difference,
            45,
            255,
            cv2.THRESH_BINARY
        )

        # Remove tiny noise and connect nearby difference pixels
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((5, 5), np.uint8)

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            open_kernel
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = test.copy()
        defect_count = 0
        crack_count = 0
        component_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore small difference regions
            if area > 150:
                x, y, w, h = cv2.boundingRect(contour)

                # Estimate a label from the shape of the difference
                aspect_ratio = max(w, h) / max(1, min(w, h))

                if aspect_ratio >= 4:
                    error_type = "Possible crack"
                    crack_count += 1
                else:
                    error_type = "Possible component difference"
                    component_count += 1

                # Draw the detection box
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                # Add the estimated error label
                cv2.putText(
                    output,
                    error_type,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                defect_count += 1

        st.subheader("Detection Results")

        c1, c2, c3 = st.columns(3)

        c1.image(
            reference,
            channels="BGR",
            caption="Reference PCB"
        )

        c2.image(
            test,
            channels="BGR",
            caption="Test PCB"
        )

        c3.image(
            output,
            channels="BGR",
            caption="Detected Differences with Labels"
        )

        st.metric("Potential Defect Regions", defect_count)

        summary1, summary2 = st.columns(2)
        summary1.metric("Possible Crack-like Regions", crack_count)
        summary2.metric("Possible Component Differences", component_count)

        if defect_count > 0:
            st.warning(
                "Potential differences detected. Review the highlighted "
                "regions; labels are estimates and may include false positives."
            )
        else:
            st.success("No significant differences detected.")