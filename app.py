"""Hospital Multi-Model AI Risk Screening & Brain Stroke Detection System.

Run with:  streamlit run app.py
"""
import time

import pandas as pd
import streamlit as st

import auth
import stroke_model
import vitals_model

st.set_page_config(page_title="Hospital AI System", page_icon="🏥", layout="wide")

DISCLAIMER = (
    "Decision support only. This tool does not replace the diagnosis of a doctor or a radiologist. "
    "Always confirm results with a qualified clinician."
)
MAX_FAILED_LOGINS = 5
LOCK_SECONDS = 30

# ---------------------------------------------------------------- session state
st.session_state.setdefault("user", None)
st.session_state.setdefault("failed_logins", 0)
st.session_state.setdefault("locked_until", 0.0)


# ---------------------------------------------------------------- login (FR-1.1, FR-1.2)
def login_view():
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.markdown("## 🏥 Hospital AI System")
        st.caption("Sign in to continue. Authorized hospital staff only.")
        if auth.user_count() == 0:
            st.warning("No accounts exist yet. Create one first by running:  `python create_user.py`")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. staff_sara")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            now = time.time()
            if now < st.session_state.locked_until:
                st.error(f"Too many failed attempts. Try again in {int(st.session_state.locked_until - now)} seconds.")
                return
            user = auth.verify_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.failed_logins = 0
                st.rerun()
            st.session_state.failed_logins += 1
            if st.session_state.failed_logins >= MAX_FAILED_LOGINS:
                st.session_state.locked_until = now + LOCK_SECONDS
                st.session_state.failed_logins = 0
            st.error("Incorrect username or password.")


# ---------------------------------------------------------------- Tab 1: vital signs
def vitals_tab():
    st.subheader("Patient Vital-Signs Risk Screening")
    st.caption("Enter the patient's current vital signs. The model flags whether the pattern looks Normal or Abnormal.")

    if not vitals_model.model_available():
        st.info("The vital-signs model file (`models/vitals_model.joblib`) is not installed yet.")
        return

    with st.form("vitals_form"):
        cols = st.columns(2)
        values = {}
        for i, (key, label, lo, hi, default, step) in enumerate(vitals_model.FIELDS):
            with cols[i % 2]:
                values[key] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)
        predict_clicked = st.form_submit_button("Predict", type="primary")

    if predict_clicked:
        errors = vitals_model.validate(values)
        if errors:
            for e in errors:
                st.error(e)
        else:
            bundle = vitals_model.load_bundle()
            label, proba, contrib = vitals_model.predict(bundle, values)
            st.session_state["vitals_result"] = {"label": label, "proba": proba, "contrib": contrib, "values": values}

    res = st.session_state.get("vitals_result")
    if res:
        st.divider()
        left, right = st.columns([1, 1.4])
        with left:
            if res["label"] == "Abnormal":
                st.error(f"### Predicted: ABNORMAL (higher risk)\nProbability of abnormal pattern: **{res['proba']:.0%}**")
            else:
                st.success(f"### Predicted: NORMAL (lower risk)\nProbability of abnormal pattern: **{res['proba']:.0%}**")
            st.caption(DISCLAIMER)
        with right:
            contrib = res["contrib"]
            if contrib is None:
                st.info("Explanation is not available for this model.")
            else:
                st.markdown("**Which vitals influenced this result (SHAP)**")
                top = contrib.iloc[0]
                direction = "toward Abnormal" if top["Push toward Abnormal"] > 0 else "toward Normal"
                st.write(f"The strongest factor was **{top['Vital sign']}** ({top['Value']:g}), which pushed the result {direction}.")
                st.bar_chart(contrib.set_index("Vital sign")["Push toward Abnormal"])
                st.caption("Positive bars push toward Abnormal. Negative bars push toward Normal.")


# ---------------------------------------------------------------- Tab 2: stroke detection
def stroke_tab():
    st.subheader("Brain Stroke Detection from CT Scans")
    st.caption("Upload one axial non-contrast brain CT slice. The model boxes every suspected lesion and labels it Hemorrhage or Ischemia.")

    if not stroke_model.model_available():
        st.info("The stroke model file (`models/best.pt`) is not installed yet.")
        return

    left, right = st.columns([1, 1.2])
    with left:
        uploaded = st.file_uploader("Upload CT slice (PNG, JPG or DICOM)", type=["png", "jpg", "jpeg", "dcm"])
        threshold = st.slider(
            "Confidence threshold", 0.05, 0.90, stroke_model.DEFAULT_CONF, 0.05,
            help="Lower = catches more strokes but gives more false alarms. Default 0.20 was chosen for high sensitivity.",
        )
        analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.session_state.pop("stroke_result", None)
    if analyze and uploaded is not None:
        try:
            image = stroke_model.read_slice(uploaded)
            model = stroke_model.load_model()
            with st.spinner("Analyzing..."):
                detections = stroke_model.detect(model, image, conf=0.05)   # keep low-confidence boxes, filter by slider below
            st.session_state["stroke_result"] = {"name": uploaded.name, "image": image, "detections": detections}
        except ValueError as e:
            st.session_state.pop("stroke_result", None)
            st.error(str(e))

    res = st.session_state.get("stroke_result")
    if res is None:
        with right:
            st.info("Upload a slice and press **Analyze**.")
    else:
        shown = [d for d in res["detections"] if d["conf"] >= threshold]
        label, conf, kind = stroke_model.slice_result(shown)
        with right:
            st.image(stroke_model.draw_boxes(res["image"], shown), caption=res["name"])
        with left:
            if kind == "none":
                st.success(f"### {label}")
            else:
                st.error(f"### {label}\nConfidence: **{conf:.0%}**")
            st.write(f"Boxes shown: **{len(shown)}** at threshold {threshold:.2f}")
            if shown:
                table = pd.DataFrame(
                    [{"Class": d["class"], "Confidence": round(d["conf"], 3)} for d in shown]
                )
                st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption(DISCLAIMER)
            st.caption("The model analyses one 2D slice at a time, not a full 3D scan.")


# ---------------------------------------------------------------- main
if st.session_state.user is None:
    login_view()
    st.stop()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown("## 🏥 Hospital AI System")
with top_right:
    st.write(f"👤 {st.session_state.user['username']}")
    if st.button("Log out"):
        for key in ("user", "vitals_result", "stroke_result"):
            st.session_state.pop(key, None)
        st.rerun()

tab1, tab2 = st.tabs(["Vital-Signs Risk Screening", "Brain Stroke Detection"])
with tab1:
    vitals_tab()
with tab2:
    stroke_tab()
st.divider()
st.caption(DISCLAIMER)
