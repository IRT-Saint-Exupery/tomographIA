"""
Streamlit labelling app.

Launch with ``tomographia-label-app`` (or ``streamlit run <path to this file>``).
"""
import streamlit as st

from tomographia.labelling import (
    LabelingConfig,
    LabelingSession,
    apply_threshold_labels,
    df_recap,
    load_label_df,
)

st.set_page_config(page_title="tomographIA patch labeller", layout="wide")
st.markdown("# Patch labeller")

with st.sidebar:
    data_path = st.text_input("Data folder (CSV + .tif slices)", "")
    identifier = st.text_input("CSV identifier", "")
    label_col = st.text_input("Label column", "defect")
    threshold = st.number_input("Auto-label 0 below intensity", 0.0, 1e6, 50.0, step=1.0)
    scale = st.number_input("Display scale", 0.01, 100.0, 1.0, step=0.1)
    offset = st.number_input("Display offset", -10000.0, 10000.0, 0.0, step=10.0)
    show_prediction = st.toggle("Show model prediction (CSV 'prediction' column)", value=False)

    if st.button("Start / reset session", disabled=not (data_path and identifier)):
        cfg = LabelingConfig(data_path=data_path, identifier=identifier, label_col=label_col,
                             threshold=threshold, scale=scale, offset=offset)
        df = load_label_df(cfg.data_path, cfg.identifier)
        apply_threshold_labels(df, cfg.label_col, cfg.intensity_col, cfg.threshold)
        st.session_state.sess = LabelingSession(df, cfg)
        st.session_state.iter = st.session_state.sess.iter_patches()
        st.session_state.current = None

    if "sess" in st.session_state:
        st.write(df_recap(st.session_state.sess.df, st.session_state.sess.label_col))


def _next_patch():
    st.session_state.current = next(st.session_state.iter, None)


if "sess" not in st.session_state:
    st.info("Fill in the data folder and CSV identifier, then start a session.")
    st.stop()

sess = st.session_state.sess
if st.session_state.get("current") is None:
    _next_patch()
patch = st.session_state.current
if patch is None:
    st.success("No more unlabelled patches.")
    if st.button("Save CSV"):
        sess.save()
        st.success("Saved.")
    st.stop()

fig, _ = sess.figure_for_patch(sess.load_slice(patch), patch)
st.pyplot(fig)
st.markdown(f"**Patch** `{patch.id_global}` — intensity {patch.intensity:.1f}")
if show_prediction:
    st.markdown("No prediction available." if patch.prediction is None
                else f"**Model P(defect):** {patch.prediction * 100:.1f} %")

st.markdown("`0` no defect · `1` defect · `2` unsure · `s` save · `n` skip · `k` stop")
with st.form("label", clear_on_submit=True):
    key = st.text_input("Label", max_chars=3).strip().lower()
    submitted = st.form_submit_button("Apply (Enter)")

if submitted:
    if key in {"0", "1", "2"}:
        sess.assign_label(patch, int(key))
        st.session_state.current = None
        st.rerun()
    elif key == "s":
        sess.save()
        st.success("Saved.")
    elif key == "n":
        st.session_state.current = None
        st.rerun()
    elif key == "k":
        sess.save()
        for k in ("sess", "iter", "current"):
            st.session_state.pop(k, None)
        st.rerun()
    else:
        st.warning("Use 0/1/2, s, n or k.")
