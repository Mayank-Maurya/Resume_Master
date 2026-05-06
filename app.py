import base64
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv

from agent.compile import OUTPUT_DIR, DEFAULT_TIMEOUT_SECS
from agent.extract import extract_pdf_text
from agent.flatten import flatten_tex_zip
from agent.pipeline import run

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


class StreamlitLogHandler(logging.Handler):
    """Mirror agent log records into a Streamlit code block, live."""

    def __init__(self, buffer: list[str], placeholder):
        super().__init__()
        self.buffer = buffer
        self.placeholder = placeholder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
            self.placeholder.code(
                "\n".join(self.buffer[-300:]), language="log"
            )
        except Exception:
            self.handleError(record)


st.set_page_config(page_title="Resume Master", layout="wide")
st.title("Resume Master")
st.caption("Tailor your LaTeX resume to a job description with an ATS-focused agent.")

with st.container(border=True):
    cols = st.columns([4, 1])
    cols[0].markdown(f"**Output directory** &nbsp; `{OUTPUT_DIR}`")
    cols[1].markdown(f"**Compile timeout** &nbsp; `{DEFAULT_TIMEOUT_SECS}s`")
    existing = sorted(OUTPUT_DIR.glob("resume.*")) if OUTPUT_DIR.exists() else []
    if existing:
        with st.expander(f"Files currently in output/ ({len(existing)})", expanded=False):
            for f in existing:
                st.code(f"{f.name}  ({f.stat().st_size:,} bytes)", language="text")

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("OPENROUTER_API_KEY not set. Copy `.env.example` to `.env` and add your key from https://openrouter.ai/keys.")
    st.stop()

with st.sidebar:
    st.header("Inputs")
    tex_file = st.file_uploader(
        "Resume LaTeX (.tex or .zip with multiple .tex files)",
        type=["tex", "zip"],
        help="A single .tex, OR a zip containing a main .tex with \\input{...} / \\include{...} references to other .tex files.",
    )
    jd_kind = st.radio("Job Description format", ["PDF", "Text"], horizontal=True)
    if jd_kind == "PDF":
        jd_file = st.file_uploader("Job Description (PDF)", type=["pdf"])
        jd_text = None
    else:
        jd_file = None
        jd_text = st.text_area("Paste Job Description", height=240)
    run_button = st.button("Tailor Resume", type="primary", use_container_width=True)


def _read_jd() -> str | None:
    if jd_kind == "PDF":
        if not jd_file:
            return None
        with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(jd_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            return extract_pdf_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return (jd_text or "").strip() or None


def _attach_ui_log_handler(buffer: list[str], placeholder) -> StreamlitLogHandler:
    handler = StreamlitLogHandler(buffer, placeholder)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
        )
    )
    for name in ("agent", "app"):
        logging.getLogger(name).addHandler(handler)
    return handler


def _detach_ui_log_handler(handler: StreamlitLogHandler) -> None:
    for name in ("agent", "app"):
        logging.getLogger(name).removeHandler(handler)


if run_button:
    if not tex_file:
        st.error("Please upload your resume .tex or .zip file.")
        st.stop()

    jd = _read_jd()
    if not jd:
        st.error("Please provide the job description.")
        st.stop()

    upload_name = tex_file.name.lower()

    status = st.status("Working...", expanded=True)
    final: dict | None = None
    with status:
        st.markdown("**Live logs**")
        log_placeholder = st.empty()
        log_buffer: list[str] = []
        ui_handler = _attach_ui_log_handler(log_buffer, log_placeholder)

        try:
            if upload_name.endswith(".zip"):
                logger.info("Upload is a .zip (%d bytes)", len(tex_file.getvalue()))
                try:
                    resume_tex, included, main_rel = flatten_tex_zip(tex_file.getvalue())
                except (ValueError, KeyError) as e:
                    logger.error("Zip flatten failed: %s", e)
                    st.error(f"Could not flatten zip: {e}")
                    st.stop()
                st.info(f"Detected main file: `{main_rel}` — flattened {len(included)} .tex files.")
                with st.expander("Files included (in resolution order)"):
                    for f in included:
                        st.code(f, language="text")
            else:
                logger.info("Upload is a single .tex (%d bytes)", len(tex_file.getvalue()))
                resume_tex = tex_file.getvalue().decode("utf-8")

            logger.info("Resume source ready: %d chars; JD: %d chars", len(resume_tex), len(jd))

            for event in run(resume_tex, jd):
                stage = event["stage"]
                if stage == "tailoring":
                    st.write(f"Tailoring resume (attempt {event['attempt']})...")
                elif stage == "compiling":
                    st.write(f"Compiling LaTeX (attempt {event['attempt']})...")
                elif stage == "compile_failed":
                    st.warning(f"Compile failed on attempt {event['attempt']}, retrying...")
                    with st.expander(f"Compile log (attempt {event['attempt']})"):
                        st.code(event["log"][-2000:])
                elif stage == "done":
                    st.success("PDF compiled successfully.")
                    final = event
                elif stage == "error":
                    st.error(event["message"])
                    final = event
        finally:
            _detach_ui_log_handler(ui_handler)

    st.session_state["final"] = final

final = st.session_state.get("final")
if final and final.get("stage") == "done":
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Preview")
        b64 = base64.b64encode(final["pdf"]).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="900px" style="border:1px solid #ccc"></iframe>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Downloads")
        st.download_button(
            "Download PDF",
            data=final["pdf"],
            file_name="resume_tailored.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "Download .tex",
            data=final["tex"],
            file_name="resume_tailored.tex",
            mime="text/x-tex",
            use_container_width=True,
        )
        with st.expander("Tailored .tex source"):
            st.code(final["tex"], language="latex")
elif final and final.get("stage") == "error":
    st.error(final["message"])
    if final.get("last_log"):
        with st.expander("Last compile log"):
            st.code(final["last_log"][-3000:])
    if final.get("last_tex"):
        with st.expander("Last attempted .tex"):
            st.code(final["last_tex"], language="latex")
