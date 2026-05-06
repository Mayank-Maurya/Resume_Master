import json
import logging
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-3-27b-it:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 8192
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are an ATS resume content editor.

You will receive:
- A job description.
- A JSON object whose keys are bullet IDs and values are the original bullet text (PLAIN ENGLISH PROSE, no markup).

Rewrite each bullet to maximize keyword overlap with the job description, while staying strictly truthful to the original. Output the rewritten bullets as a JSON object with the SAME keys.

Hard rules:
- PLAIN TEXT ONLY. No LaTeX commands (no \\textbf, \\texttt, \\item, \\section, \\href, no backslashes). No Markdown (no **bold**, no `code`). No HTML.
- TRUTHFUL: never invent metrics, technologies, employers, projects, dates, or accomplishments. If the original says "20% latency improvement", do not change the number. If the original lists Node.js, do not invent Python.
- Keep each rewritten bullet within ~20% of the original length.
- Use the JD's vocabulary where the candidate genuinely has the skill.
- Lead each bullet with a strong action verb.

Output format:
A single JSON object whose values are PLAIN STRINGS. No prose, no markdown fences, no commentary. Example:
{"0": "rewritten bullet 0", "1": "rewritten bullet 1", ...}"""


_FORMAT_CMDS = (
    "textbf", "textit", "texttt", "emph", "underline",
    "textsc", "textsl", "textsf", "textrm", "textmd", "textup",
)
_STRIP_RE = re.compile(r"\\(?:" + "|".join(_FORMAT_CMDS) + r")\*?\{([^{}]*)\}")
_TWO_ARG_RE = re.compile(r"\\(?:href|textcolor|colorbox)\{[^{}]*\}\{([^{}]*)\}")
_LATEX_UNESCAPE = (
    ("\\&", "&"), ("\\%", "%"), ("\\$", "$"),
    ("\\#", "#"), ("\\_", "_"),
    ("~", " "),
)
_LATEX_ESCAPE = (
    ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
    ("#", "\\#"), ("_", "\\_"),
)


def strip_inline_latex(text: str) -> str:
    """Convert a LaTeX-flavored bullet to plain prose for the LLM."""
    out = text
    for _ in range(8):
        new = _STRIP_RE.sub(r"\1", out)
        new = _TWO_ARG_RE.sub(r"\1", new)
        if new == out:
            break
        out = new
    for esc, plain in _LATEX_UNESCAPE:
        out = out.replace(esc, plain)
    return " ".join(out.split())


def escape_for_latex(text: str) -> str:
    """Re-apply LaTeX escapes so plain LLM output substitutes safely back into .tex."""
    for find, sub in _LATEX_ESCAPE:
        text = text.replace(find, sub)
    return text


@dataclass
class BulletSpan:
    start: int
    end: int
    text: str


_ITEM_RE = re.compile(r"\\item\s*(?:\[[^\]]*\])?\s*", re.MULTILINE)
_BULLET_END_RE = re.compile(
    r"\\(?:item\b|end\{itemize\}|end\{enumerate\}|section\b|subsection\b|paragraph\b)"
)


def find_bullet_spans(tex: str) -> list[BulletSpan]:
    """Return the position+text of every \\item body in the document."""
    spans: list[BulletSpan] = []
    for m in _ITEM_RE.finditer(tex):
        text_start = m.end()
        tail = _BULLET_END_RE.search(tex, text_start)
        text_end = tail.start() if tail else len(tex)
        raw = tex[text_start:text_end]
        stripped = raw.rstrip()
        if stripped.strip():
            spans.append(BulletSpan(text_start, text_start + len(stripped), stripped))
    return spans


def apply_substitutions(
    tex: str, spans: list[BulletSpan], new_texts: list[str]
) -> tuple[str, int, int]:
    """Replace span text in tex (right-to-left so positions stay valid).

    Returns (new_tex, applied_count, skipped_count).
    """
    out = tex
    applied = 0
    skipped = 0
    for span, new in sorted(
        zip(spans, new_texts), key=lambda p: p[0].start, reverse=True
    ):
        if not _is_safe_replacement(new):
            logger.warning(
                "Skipping unsafe replacement (unbalanced braces): %r", new[:80]
            )
            skipped += 1
            continue
        out = out[: span.start] + new + out[span.end :]
        applied += 1
    return out, applied, skipped


def _is_safe_replacement(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return text.count("{") == text.count("}")


def tailor_resume(
    resume_tex: str,
    job_description: str,
    previous_attempt: str | None = None,
    compile_error: str | None = None,
) -> str:
    """Tailor bullets in resume_tex to job_description; LaTeX structure is untouched."""
    spans = find_bullet_spans(resume_tex)
    if not spans:
        raise ValueError(
            "No \\item bullets found in resume. This pipeline edits bullet text only; "
            "your resume template must use \\item inside itemize/enumerate environments."
        )
    logger.info("Found %d bullets to tailor", len(spans))

    bullets_plain = [strip_inline_latex(s.text) for s in spans]
    for i, (s, plain) in enumerate(zip(spans, bullets_plain)):
        logger.info(
            "Extracted bullet [%d] (pos %d-%d):\n    RAW:    %s\n    PLAIN:  %s",
            i, s.start, s.end, s.text, plain,
        )

    tailored_plain = _tailor_bullets_via_llm(bullets_plain, job_description)

    tailored_escaped: list[str] = []
    for i in range(len(bullets_plain)):
        if i < len(tailored_plain) and tailored_plain[i] and tailored_plain[i] != bullets_plain[i]:
            escaped = escape_for_latex(tailored_plain[i])
            tailored_escaped.append(escaped)
            logger.info(
                "Tailored bullet [%d]:\n    BEFORE: %s\n    AFTER:  %s",
                i, bullets_plain[i], tailored_plain[i],
            )
        else:
            tailored_escaped.append(spans[i].text)
            logger.info("Bullet [%d]: kept original (LLM returned blank/unchanged)", i)

    tailored = tailored_escaped

    if len(tailored) != len(bullets):
        logger.warning(
            "LLM returned %d bullets, expected %d. Filling missing with originals.",
            len(tailored), len(bullets),
        )
        for i in range(len(bullets)):
            if i >= len(tailored) or not tailored[i]:
                tailored.append(bullets[i]) if i >= len(tailored) else (tailored.__setitem__(i, bullets[i]))

    new_tex, applied, skipped = apply_substitutions(resume_tex, spans, tailored)
    logger.info(
        "Substituted %d/%d bullets (skipped %d unsafe); resume length %d -> %d chars",
        applied, len(spans), skipped, len(resume_tex), len(new_tex),
    )
    return new_tex


def _tailor_bullets_via_llm(bullets: list[str], jd: str) -> list[str]:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost:8501"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Resume Master"),
        },
    )
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

    payload = {str(i): b for i, b in enumerate(bullets)}
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    user_msg = (
        f"<job_description>\n{jd}\n</job_description>\n\n"
        f"<original_bullets>\n{payload_json}\n</original_bullets>\n\n"
        "Rewrite each bullet per the rules. Output JSON with the same keys."
    )

    logger.info(
        "Calling model=%s for %d bullets (jd=%d chars, prompt~=%d chars)",
        model, len(bullets), len(jd), len(user_msg),
    )
    logger.debug("Full JSON payload to LLM:\n%s", payload_json)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    elapsed = time.perf_counter() - start

    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    print(f"{text}")
    if usage is not None:
        logger.info(
            "Model responded in %.2fs: %d chars (prompt=%s, completion=%s, total=%s tokens)",
            elapsed, len(text),
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            getattr(usage, "total_tokens", "?"),
        )
    else:
        logger.info("Model responded in %.2fs: %d chars", elapsed, len(text))

    parsed = _parse_json_object(text)
    out: list[str] = []
    for i in range(len(bullets)):
        v = parsed.get(str(i))
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        else:
            logger.warning(
                "LLM did not return a valid string for bullet %d; using original.", i
            )
            out.append(bullets[i])
    return out


def _parse_json_object(text: str) -> dict:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    logger.error("Could not parse JSON from LLM response; head: %r", text[:300])
    raise ValueError(
        "LLM did not return parseable JSON. Try a stronger model (set OPENROUTER_MODEL "
        "to e.g. anthropic/claude-sonnet-4.5 or google/gemini-2.5-flash)."
    )
