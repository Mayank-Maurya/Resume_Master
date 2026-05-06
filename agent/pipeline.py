import logging
from typing import Iterator

from .compile import CompileError, compile_tex
from .tailor import tailor_resume

logger = logging.getLogger(__name__)


def run(resume_tex: str, job_description: str) -> Iterator[dict]:
    """Stream pipeline events.

    New design: LLM only rewrites bullet text; LaTeX structure is never touched.
    Compile failures become rare (only if the original .tex had issues), so no retry.
    """
    logger.info(
        "Pipeline start: resume=%d chars, jd=%d chars",
        len(resume_tex), len(job_description),
    )

    yield {"stage": "tailoring", "attempt": 1}
    try:
        new_tex = tailor_resume(
            resume_tex=resume_tex,
            job_description=job_description,
        )
    except Exception as e:
        logger.exception("Tailoring failed")
        yield {
            "stage": "error",
            "message": f"Tailoring failed: {e}",
            "last_log": str(e),
            "last_tex": None,
        }
        return

    yield {"stage": "compiling", "attempt": 1}
    try:
        pdf_bytes = compile_tex(new_tex)
    except CompileError as e:
        logger.error(
            "Compile failed. Since structure was preserved, the original .tex likely "
            "had pre-existing issues OR a bullet replacement contained malformed LaTeX."
        )
        yield {
            "stage": "error",
            "message": (
                "Compile failed. Bullet substitutions preserve the LaTeX structure, "
                "so this usually means the original resume.tex had compile issues, "
                "or a tailored bullet contained malformed LaTeX. See the log."
            ),
            "last_log": e.log,
            "last_tex": new_tex,
        }
        return

    logger.info("Pipeline done: %d-byte PDF", len(pdf_bytes))
    yield {"stage": "done", "tex": new_tex, "pdf": pdf_bytes}
