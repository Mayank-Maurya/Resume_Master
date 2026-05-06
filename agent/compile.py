import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("RESUME_OUTPUT_DIR", "output")).resolve()
DEFAULT_TIMEOUT_SECS = int(os.environ.get("COMPILE_TIMEOUT_SECS", "300"))


class CompileError(Exception):
    def __init__(self, log: str):
        super().__init__("LaTeX compilation failed")
        self.log = log


_TEX_FALLBACK_DIRS = (
    "/Library/TeX/texbin",
    "/usr/local/texlive/2026basic/bin/universal-darwin",
    "/Library/TeX/local/bin",
)


def _which_compiler(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in _TEX_FALLBACK_DIRS:
        cand = Path(d) / name
        if cand.is_file():
            return str(cand)
    return None


def _find_compiler() -> str | None:
    preferred = os.environ.get("LATEX_COMPILER")
    if preferred:
        if path := _which_compiler(preferred):
            return path
    for binary in ("xelatex", "pdflatex", "tectonic"):
        if path := _which_compiler(binary):
            return path
    return None


def compile_tex(tex_source: str) -> bytes:
    compiler = _find_compiler()
    if not compiler:
        raise RuntimeError(
            "No LaTeX compiler found. Install tectonic with `brew install tectonic`, "
            "or a TeX distribution like `brew install --cask mactex-no-gui`."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_file = OUTPUT_DIR / "resume.tex"
    pdf_file = OUTPUT_DIR / "resume.pdf"
    log_file = OUTPUT_DIR / "resume.log"

    for stale in (pdf_file, log_file, OUTPUT_DIR / "resume.aux", OUTPUT_DIR / "resume.out"):
        if stale.exists():
            stale.unlink()
            logger.debug("Removed stale %s", stale.name)

    tex_file.write_text(tex_source, encoding="utf-8")

    logger.info(
        "Compiling LaTeX with %s (%d chars, output_dir=%s, timeout=%ds)",
        compiler, len(tex_source), OUTPUT_DIR, DEFAULT_TIMEOUT_SECS,
    )

    compiler_name = Path(compiler).name
    if compiler_name == "tectonic":
        cmd = [compiler, "--keep-logs", "-o", str(OUTPUT_DIR), str(tex_file)]
    else:
        cmd = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory", str(OUTPUT_DIR),
            str(tex_file),
        ]
    logger.debug("Compile command: %s", " ".join(cmd))

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=OUTPUT_DIR,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        logger.error(
            "Compile timed out after %.1fs (limit %ds). "
            "First runs of tectonic download package bundles and can take 1-2 min. "
            "Set COMPILE_TIMEOUT_SECS=600 to allow more time.",
            elapsed, DEFAULT_TIMEOUT_SECS,
        )
        raise CompileError(
            f"Compile timed out after {DEFAULT_TIMEOUT_SECS}s. "
            f"If this is the first run, tectonic is downloading the LaTeX package "
            f"bundle (can take 1-2 min). Increase COMPILE_TIMEOUT_SECS in .env and retry.\n\n"
            f"stderr tail:\n{(e.stderr or b'').decode(errors='ignore')[-2000:]}"
        )
    elapsed = time.perf_counter() - start

    if result.returncode != 0 or not pdf_file.exists():
        log_text = (result.stderr or "") + "\n" + (result.stdout or "")
        if log_file.exists():
            log_text += "\n\n--- LOG ---\n" + log_file.read_text(errors="ignore")
        tail = log_text[-4000:]
        logger.error(
            "Compile failed in %.2fs (exit %s, pdf_exists=%s); log tail:\n%s",
            elapsed, result.returncode, pdf_file.exists(), tail[-600:],
        )
        raise CompileError(tail)

    pdf_bytes = pdf_file.read_bytes()
    logger.info(
        "Compile succeeded in %.2fs: %d-byte PDF at %s",
        elapsed, len(pdf_bytes), pdf_file,
    )
    return pdf_bytes
