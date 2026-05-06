import logging
from typing import Iterator

from .compile import CompileError, compile_tex
from .tailor import tailor_resume

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def run(resume_tex: str, job_description: str) -> Iterator[dict]:
    """Stream pipeline events. Final event is either 'done' or 'error'."""
    logger.info(
        "Pipeline start: resume=%d chars, jd=%d chars, max_retries=%d",
        len(resume_tex), len(job_description), MAX_RETRIES,
    )
    previous_attempt: str | None = None
    compile_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Attempt %d/%d: tailoring", attempt, MAX_RETRIES)
        yield {"stage": "tailoring", "attempt": attempt}
        edited_tex = tailor_resume(
            resume_tex=resume_tex,
            job_description=job_description,
            previous_attempt=previous_attempt,
            compile_error=compile_error,
        )

        logger.info("Attempt %d/%d: compiling (%d chars)", attempt, MAX_RETRIES, len(edited_tex))
        yield {"stage": "compiling", "attempt": attempt}
        try:
            pdf_bytes = compile_tex(edited_tex)
            logger.info(
                "Pipeline done on attempt %d: %d-byte PDF", attempt, len(pdf_bytes)
            )
            yield {"stage": "done", "tex": edited_tex, "pdf": pdf_bytes}
            return
        except CompileError as e:
            previous_attempt = edited_tex
            compile_error = e.log
            logger.warning("Attempt %d failed; will retry with compile log fed back", attempt)
            yield {"stage": "compile_failed", "attempt": attempt, "log": e.log}

    logger.error("Pipeline failed after %d attempts", MAX_RETRIES)
    yield {
        "stage": "error",
        "message": f"Failed to produce a compilable PDF after {MAX_RETRIES} attempts.",
        "last_log": compile_error,
        "last_tex": previous_attempt,
    }
