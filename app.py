# backend/app.py

import os
import io
import json
import re
import uuid
import urllib.parse
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import markdown
import numpy as np
from dotenv import load_dotenv

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    current_app,
    g,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# local model client & embeddings
from model_client import chat as model_chat, generate_embeddings, embed_query

# pdf / image libs
import pypdf
import fitz  # PyMuPDF
from PIL import Image

load_dotenv()

# ---------------- basic config ----------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
EXAMPLE_PATH = os.getenv(
    "EXAMPLE_FILE",
    str(BASE_DIR.parent / "OS_Unit_1_(Notes)[1].pdf"),
)

SESSION_COOKIE_NAME = "rag_session_id"

# ---------- Flask + DB ----------
app = Flask(__name__, template_folder="templates")

# SQLite DB in backend directory
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///rag_study_buddy.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# OCR (easyocr) optional
try:
    import easyocr

    OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
    _ocr_reader = easyocr.Reader(OCR_LANGS, gpu=False)
except Exception:
    _ocr_reader = None

# YouTube transcript support (optional)
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except Exception:
    YouTubeTranscriptApi = None

    class TranscriptsDisabled(Exception):
        pass

    class NoTranscriptFound(Exception):
        pass


# ============ DB MODELS ============


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class UserSession(db.Model):
    __tablename__ = "user_session"
    id = db.Column(db.String(64), primary_key=True)  # cookie id
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)


class UploadedDocument(db.Model):
    __tablename__ = "uploaded_document"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), db.ForeignKey("user_session.id"))
    stored_filename = db.Column(db.String(255), nullable=False)  # e.g. dfc9b0...pdf
    original_name = db.Column(db.String(255), nullable=False)  # e.g. OS_Unit_1.pdf
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    __tablename__ = "chat_message"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), db.ForeignKey("user_session.id"))
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'ai'
    source = db.Column(db.String(30), nullable=False)  # 'doc_chat', 'study_chat', ...
    text = db.Column(db.Text, nullable=False)
    filename = db.Column(db.String(255), nullable=True)  # if related to a doc
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============ SESSION + AUTH HELPERS ============


@app.before_request
def load_or_create_session():
    """Give each browser a simple session id via cookie + attach user if logged in."""
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    new = False
    if not sid:
        sid = uuid.uuid4().hex
        new = True

    g.session_id = sid
    g.new_session = new
    g.user = None

    # get or create session row
    us = db.session.get(UserSession, sid)
    if not us:
        us = UserSession(id=sid)
        db.session.add(us)
        db.session.commit()

    # if session linked to a user, set g.user
    if us.user_id:
        user = db.session.get(User, us.user_id)
        g.user = user


@app.after_request
def set_session_cookie(response):
    try:
        if getattr(g, "new_session", False):
            response.set_cookie(
                SESSION_COOKIE_NAME,
                g.session_id,
                max_age=60 * 60 * 24 * 30,  # 30 days
                httponly=True,
            )
    except RuntimeError:
        pass
    return response


def _save_chat(
    session_id: str,
    role: str,
    source: str,
    text: str,
    filename: Optional[str],
):
    """Small helper to save chat messages."""
    try:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            source=source,
            text=text,
            filename=filename,
        )
        db.session.add(msg)
        db.session.commit()
    except Exception as e:
        current_app.logger.warning("Failed to save chat message: %s", e)


# ============ AUTH ROUTES ============


@app.route("/auth", methods=["GET", "POST"])
def auth_page():
    """
    Login / Sign-up / Guest in one page.

    Form fields (POST):
      - mode: 'login' | 'signup' | 'guest'
      - name (signup)
      - email
      - password
    """
    if request.method == "GET":
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template(
            "auth.html",
            example_name=ex_name,
            active="auth",
        )

    mode = request.form.get("mode", "login")
    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()
    name = (request.form.get("name") or "").strip() or "Guest"
    session_id = getattr(g, "session_id", None)

    error = None

    if mode == "signup":
        if not email or not password:
            error = "Email and password required."
        elif User.query.filter_by(email=email).first():
            error = "User already exists."
        else:
            user = User(
                email=email,
                name=name,
                password_hash=generate_password_hash(password),
            )
            db.session.add(user)
            db.session.commit()

            if session_id:
                us = db.session.get(UserSession, session_id)
                if us:
                    us.user_id = user.id
                    db.session.commit()
            return redirect(url_for("index"))

    elif mode == "login":
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            error = "Invalid email or password."
        else:
            if session_id:
                us = db.session.get(UserSession, session_id)
                if us:
                    us.user_id = user.id
                    db.session.commit()
            return redirect(url_for("index"))

    elif mode == "guest":
        # Just continue with anonymous session
        return redirect(url_for("index"))

    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template(
        "auth.html",
        example_name=ex_name,
        active="auth",
        error=error,
    )


@app.route("/logout")
def logout():
    session_id = getattr(g, "session_id", None)
    if session_id:
        us = db.session.get(UserSession, session_id)
        if us:
            us.user_id = None
            db.session.commit()

    resp = redirect(url_for("index"))
    resp.set_cookie(SESSION_COOKIE_NAME, "", expires=0)
    return resp


# ============ PDF / IMAGE / OCR HELPERS ============


def extract_text_from_pdf(path: str):
    try:
        reader = pypdf.PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except Exception:
        return []


def extract_images_from_pdf(path: str, max_images_per_page: int = 3):
    images = []
    try:
        doc = fitz.open(path)
    except Exception:
        return images
    for i in range(len(doc)):
        page = doc[i]
        img_list = page.get_images(full=True)
        count = 0
        for im in img_list:
            if count >= max_images_per_page:
                break
            xref = im[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
                count += 1
            except Exception:
                continue
    return images


def ocr_image_local(img: Image.Image) -> str:
    if _ocr_reader is None:
        return ""
    try:
        arr = np.array(img)
        texts = _ocr_reader.readtext(arr, detail=0)
        return "\n".join(texts)
    except Exception:
        return ""


def chunk_texts(pages, chunk_size: int = 1000, chunk_overlap: int = 200):
    chunks = []
    for p in pages:
        s = p.strip()
        if not s:
            continue
        i = 0
        while i < len(s):
            chunks.append(s[i : i + chunk_size])
            i += chunk_size - chunk_overlap
    return chunks


def build_chunks_for_file(path: str, max_images_ocr: int = 2):
    ext = Path(path).suffix.lower()
    pages = []
    if ext == ".pdf":
        pages = extract_text_from_pdf(path)
        imgs = extract_images_from_pdf(path)
        for img in imgs[:max_images_ocr]:
            t = ocr_image_local(img)
            if t and t.strip():
                pages.append(t)
    elif ext in {".txt", ".md"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                pages = [f.read()]
        except Exception:
            pages = []
    else:
        try:
            img = Image.open(path).convert("RGB")
            t = ocr_image_local(img)
            if t:
                pages.append(t)
        except Exception:
            pages = []
    chunks = chunk_texts(pages)
    return chunks


# ============ RETRIEVAL ============


def similarity_search(chunks, query, top_k: int = 6):
    chunk_embs = generate_embeddings(chunks)
    q_emb = embed_query(query)
    vectors = np.array(chunk_embs)
    q = np.array(q_emb)
    norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(q) + 1e-12)
    sims = vectors.dot(q) / norms
    idx = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in idx]


# ============ QUESTION GENERATION PROMPTS ============


def build_instruction(mode: str, num_q: int, include_answers: bool) -> str:
    if mode == "mcq":
        if include_answers:
            inst = (
                f"Create {num_q} multiple-choice questions (A-D) from the context. "
                "Return JSON array: [{\"question\":\"...\",\"choices\":"
                "{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"},\"answer\":\"A\"}]."
            )
        else:
            inst = (
                f"Create {num_q} multiple-choice questions (A-D) from the context. "
                "Return JSON array: [{\"question\":\"...\",\"choices\":"
                "{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"}}]."
            )
    elif mode == "short":
        inst = f"Create {num_q} short-answer questions (1-3 sentences)."
        if include_answers:
            inst += " Include 'answer' field with brief answers."
        inst += " Return JSON array: [{\"question\":\"...\",\"answer\":\"...\"}]."
    else:  # long answer
        inst = f"Create {num_q} long-answer questions (detailed)."
        if include_answers:
            inst += " Include 'answer' field with comprehensive answers."
        inst += " Return JSON array: [{\"question\":\"...\",\"answer\":\"...\"}]."
    return inst


# ============ YOUTUBE HELPERS ============

YOUTUBE_REGEX = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=[\w\-]+|youtu\.be/[\w\-]+)",
    re.IGNORECASE,
)


def extract_youtube_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = YOUTUBE_REGEX.search(text)
    return m.group(0) if m else None


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None

    if "youtu.be/" in url:
        return url.rstrip("/").split("/")[-1].split("?")[0]

    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    return None


def language_to_iso(lang: str) -> List[str]:
    if not lang:
        return ["en"]

    name = lang.strip().lower()
    mapping = {
        "english": ["en"],
        "hindi": ["hi", "en"],
        "hinglish": ["hi", "en"],
        "spanish": ["es", "en"],
        "french": ["fr", "en"],
        "german": ["de", "en"],
        "portuguese": ["pt", "en"],
        "bengali": ["bn", "en"],
        "tamil": ["ta", "en"],
        "telugu": ["te", "en"],
        "marathi": ["mr", "en"],
        "gujarati": ["gu", "en"],
        "urdu": ["ur", "en"],
    }
    return mapping.get(name, ["en"])


def get_youtube_transcript_text(video_id: str, language: str = "English") -> str:
    """Fetch plain-text transcript for a YouTube video id."""
    try:
        import youtube_transcript_api as yta_mod
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
        )
    except Exception as e:
        raise RuntimeError(
            "youtube-transcript-api is not installed in this environment. "
            "Install with: pip install youtube-transcript-api"
        ) from e

    api_callable = None
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        api_callable = YouTubeTranscriptApi.get_transcript
    elif hasattr(yta_mod, "YouTubeTranscriptApi") and hasattr(
        yta_mod.YouTubeTranscriptApi, "get_transcript"
    ):
        api_callable = yta_mod.YouTubeTranscriptApi.get_transcript
    elif hasattr(yta_mod, "get_transcript"):
        api_callable = yta_mod.get_transcript

    if api_callable is None:
        raise RuntimeError(
            "youtube-transcript-api is imported, but no get_transcript() function "
            "could be found."
        )

    langs = language_to_iso(language)

    try:
        transcript = api_callable(video_id, languages=langs)
    except (TranscriptsDisabled, NoTranscriptFound):
        if "en" not in langs:
            transcript = api_callable(video_id, languages=["en"])
        else:
            raise RuntimeError(
                "This YouTube video has subtitles disabled or no transcript is available."
            )
    except Exception as e:
        raise RuntimeError(f"Error retrieving transcript from YouTube: {e}") from e

    return " ".join(seg.get("text", "") for seg in transcript if seg.get("text"))


# ============ IMPORTANT QUESTION SHORTLISTING ============


def shortlist_important_questions(
    raw_text: str, num_q: int = 20, language: str = "English"
):
    prompt = f"""
You are an expert exam paper analyst.

You will receive questions mixed from MULTIPLE past exam papers.
Your job is to shortlist the MOST IMPORTANT exam questions for students.

Context (questions from many papers, unstructured):
{raw_text}

Task:
- Select the TOP {num_q} questions that are high-yield for exams.
- Prefer questions that:
  - cover core concepts
  - are frequently repeated or appear in similar form
  - cover large parts of the syllabus
- If similar questions appear multiple times, merge them and keep ONE good phrasing.
- Ignore marking schemes, section headings, page numbers, and random text.

Output format:
Return ONLY valid JSON, no backticks, no extra text.
JSON structure:

[
  {{
    "question": "rephrased important question in clear exam style",
    "reason": "short explanation why this question is important / repeated / high-weight"
  }},
  ...
]

Language:
- Write all questions and reasons in: {language}
"""

    raw = model_chat(prompt, max_tokens=1500, temperature=0.3)

    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            try:
                return json.loads(m.group(1).replace("'", '"'))
            except Exception:
                return {"raw": raw}
    else:
        return {"raw": raw}


# ============ CENTRALIZED FILE CHAT HANDLER ============


def _handle_chat_for_path(resolved_path: str, question: str, language: str):
    if not resolved_path or not question:
        return 400, {"status": "error", "error": "missing parameters"}

    if not os.path.isabs(resolved_path):
        resolved_path = str(UPLOAD_DIR / resolved_path)

    if not os.path.exists(resolved_path):
        current_app.logger.warning("File not found at %s", resolved_path)
        return 404, {"status": "error", "error": f"file not found: {resolved_path}"}

    try:
        chunks = build_chunks_for_file(str(resolved_path))
        if not chunks:
            return 400, {"status": "error", "error": "no extractable text from file"}

        top_ctx = similarity_search(chunks, question, top_k=6)
        context = "\n\n---\n\n".join(top_ctx)

        prompt = f"""
You are an assistant for question answering over documents.

Use ONLY the given context to answer the user's question.
Always answer in {language}. Do not change the language.

Context:
{context}

Question:
{question}

Return only JSON.
"""
        raw = model_chat(prompt)

        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if m:
            try:
                ans = json.loads(m.group(1))
            except Exception:
                try:
                    ans = json.loads(m.group(1).replace("'", '"'))
                except Exception:
                    ans = {"raw": raw}
        else:
            ans = {"raw": raw}

        return 200, {"status": "ok", "answer": ans}
    except Exception as e:
        current_app.logger.exception("Error in chat handler")
        return 500, {"status": "error", "error": str(e)}


# ============ ROUTES: PAGES (HOME / STUDY / RESULT) ============


@app.route("/", methods=["GET"])
def index():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template(
        "index.html",
        example_path=EXAMPLE_PATH,
        example_name=ex_name,
        active="home",
    )


@app.route("/study", methods=["GET"])
def study_page():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template("study.html", example_name=ex_name, active="study")


@app.route("/study_result", methods=["POST"])
def study_result_page():
    """
    YouTube + notes study page.

    - If YouTube link is given, try transcript.
    - If transcript fails or is missing, still explain the topic using URL + notes.
    """
    youtube_url = (request.form.get("youtube_url") or "").strip()
    notes = (request.form.get("notes") or "").strip()
    language = (request.form.get("language") or "English").strip()

    if not youtube_url and not notes:
        return redirect(url_for("study_page"))

    yt_url = extract_youtube_url(youtube_url) if youtube_url else None
    video_id = extract_youtube_id(yt_url) if yt_url else None

    transcript_text = ""
    transcript_error = ""

    if video_id:
        try:
            transcript_text = get_youtube_transcript_text(video_id, language=language)
        except Exception as e:
            transcript_error = str(e)
            current_app.logger.warning("Transcript error: %s", e)

    if yt_url:
        transcript_snippet = (
            transcript_text[:2000] if transcript_text else "[no transcript available]"
        )

        prompt = f"""
You are an expert teacher and video explainer.

You do NOT have access to the YouTube website or the actual video,
only this information:

- YouTube URL: {youtube_url}
- Parsed video id: {video_id or 'could not parse'}
- Extra notes / text from student (may be empty):
{notes or '[none]'}

- Automatic transcript (may be empty or missing):
{transcript_snippet}

If the transcript is missing or looks like an error, DO NOT pretend you watched the video.
In that case, infer the most likely TOPIC from the URL text and student's notes,
and then explain that topic generally from your own knowledge.

In {language}, give the student a full understanding with these sections:

1. Video topic & main idea
2. Detailed explanation of concepts (step-by-step)
3. Key points / takeaways (bullets)
4. Important definitions & formulas (if relevant)
5. Intuitive / real-life examples (if possible)
6. 5–10 exam or interview questions with short answers
"""
        try:
            first_answer = model_chat(prompt, max_tokens=1800, temperature=0.35)
        except Exception as e:
            first_answer = f"Error while generating answer: {e}"

        original_message = youtube_url + ("\n\n" + notes if notes else "")
    else:
        study_prompt = f"""
You are a friendly, expert study assistant for students.

- Explain concepts clearly and step-by-step.
- When solving problems, show all important steps.
- Use simple language and examples when needed.
- Always reply in {language}.

Student's notes / text:
{notes}
"""
        try:
            first_answer = model_chat(study_prompt, max_tokens=1200, temperature=0.35)
        except Exception as e:
            first_answer = f"Error while generating answer: {e}"

        original_message = notes

    session_id = getattr(g, "session_id", None)
    if session_id and original_message:
        _save_chat(session_id, "user", "study_result", original_message, None)
    if session_id and first_answer:
        _save_chat(session_id, "ai", "study_result", first_answer, None)

    first_answer_html = markdown.markdown(
        first_answer,
        extensions=["fenced_code", "tables"],
    )

    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template(
        "study_result.html",
        original_message=original_message,
        first_answer_html=first_answer_html,
        language=language,
        example_name=ex_name,
        active="study",
    )


# ============ FILE UPLOAD + QUESTION GENERATION ============


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))
    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return "Not allowed file type", 400
    uid = uuid.uuid4().hex
    saved = UPLOAD_DIR / f"{uid}{ext}"
    f.save(saved)

    session_id = getattr(g, "session_id", None)
    if session_id:
        try:
            doc_row = UploadedDocument(
                session_id=session_id,
                stored_filename=saved.name,
                original_name=f.filename or saved.name,
            )
            db.session.add(doc_row)
            db.session.commit()
        except Exception as e:
            current_app.logger.warning("Failed to save upload row: %s", e)

    return redirect(url_for("process_file", filename=saved.name))


@app.route("/process/<filename>", methods=["GET", "POST"])
def process_file(filename):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return "Not found", 404
    if request.method == "GET":
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template("ask.html", filename=filename, example_name=ex_name)

    mode = request.form.get("mode", "mcq")
    num_q = int(request.form.get("num_questions", "4"))
    include_answers = request.form.get("include_answers") == "on"
    language = request.form.get("language", "English")
    try:
        chunks = build_chunks_for_file(str(path))
        if not chunks:
            return "No extractable text", 400
        context = "\n\n---\n\n".join(chunks[:8])
        inst = build_instruction(mode, num_q, include_answers)
        prompt = (
            f"You are a question generator. {inst}\nContext:\n{context}\n"
            f"Output language: {language}\nReturn only JSON."
        )
        raw = model_chat(prompt)
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if m:
            try:
                qdata = json.loads(m.group(1))
            except Exception:
                try:
                    qdata = json.loads(m.group(1).replace("'", '"'))
                except Exception:
                    qdata = {"raw": raw}
        else:
            qdata = {"raw": raw}
        ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
        return render_template(
            "results.html",
            filename=filename,
            questions=qdata,
            language=language,
            example_name=ex_name,
        )
    except Exception as e:
        current_app.logger.exception("Error in process_file")
        return f"Error: {e}", 500


# ============ CHAT OVER UPLOADED FILE(S) ============


@app.route("/chat/<filename>", methods=["POST"])
def chat_file(filename):
    question = request.form.get("question", "").strip()
    language = request.form.get("language", "English")
    session_id = getattr(g, "session_id", None)

    if session_id and question:
        _save_chat(session_id, "user", "doc_chat", question, filename)

    status, payload = _handle_chat_for_path(filename, question, language)

    if session_id and status == 200 and payload.get("status") == "ok":
        ans_obj = payload.get("answer")
        if isinstance(ans_obj, str):
            text_to_save = ans_obj
        else:
            text_to_save = json.dumps(ans_obj, ensure_ascii=False)[:4000]
        _save_chat(session_id, "ai", "doc_chat", text_to_save, filename)

    return jsonify(payload), status


@app.route("/chat", methods=["POST"])
def chat_generic():
    question = (request.form.get("question") or "").strip()
    language = (request.form.get("language") or "English").strip()
    file_url = request.form.get("file_url")
    filename = request.form.get("filename")
    session_id = getattr(g, "session_id", None)

    if not question:
        return jsonify({"status": "error", "error": "missing question"}), 400

    if file_url:
        resolved_path = file_url
    elif filename:
        resolved_path = str(UPLOAD_DIR / filename)
    else:
        return (
            jsonify({"status": "error", "error": "missing filename or file_url"}),
            400,
        )

    if session_id:
        _save_chat(session_id, "user", "doc_chat_generic", question, filename)

    status, payload = _handle_chat_for_path(resolved_path, question, language)

    if session_id and status == 200 and payload.get("status") == "ok":
        ans_obj = payload.get("answer")
        if isinstance(ans_obj, str):
            text_to_save = ans_obj
        else:
            text_to_save = json.dumps(ans_obj, ensure_ascii=False)[:4000]
        _save_chat(session_id, "ai", "doc_chat_generic", text_to_save, filename)

    return jsonify(payload), status


# ============ STUDY CHAT API (separate from study_result page) ============


@app.route("/study_chat", methods=["POST"])
def study_chat():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        language = (data.get("language") or "English").strip()
    else:
        message = (request.form.get("message") or "").strip()
        language = (request.form.get("language") or "English").strip()

    if not message:
        return jsonify({"status": "error", "error": "missing message"}), 400

    session_id = getattr(g, "session_id", None)
    if session_id:
        _save_chat(session_id, "user", "study_chat", message, None)

    yt_url = extract_youtube_url(message)
    if yt_url:
        vid = extract_youtube_id(yt_url)
        transcript_text = ""
        transcript_error = ""

        if not vid:
            transcript_error = "could not parse YouTube video ID from the URL"
        else:
            try:
                transcript_text = get_youtube_transcript_text(vid, language=language)
            except Exception as e:
                transcript_error = str(e)

        transcript_snippet = (
            transcript_text[:2000] if transcript_text else "[no transcript available]"
        )

        prompt = f"""
You are an expert teacher and explainer for YouTube videos.

You do NOT have access to the YouTube website or actual video,
only this information:

- YouTube URL: {yt_url}
- Parsed video id: {vid or 'could not parse'}
- Student's request / question:
{message}

- Automatic transcript (may be empty or missing):
{transcript_snippet}

If the transcript is missing or looks like an error, DO NOT pretend you watched the video.
In that case, infer the most likely TOPIC from the URL text and student's message,
and then explain that topic generally from your own knowledge.

In {language}, respond with:

1. Short overview of what this video is (or is likely) about.
2. Detailed explanation of the core concepts, step-by-step.
3. Key points or takeaways in bullets.
4. Important definitions / formulas (if relevant).
5. 3–5 exam or interview style questions with short answers.
"""
        try:
            info = model_chat(prompt, max_tokens=1800, temperature=0.35)
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

        if session_id:
            _save_chat(session_id, "ai", "study_chat", info, None)

        return jsonify(
            {
                "status": "ok",
                "mode": "youtube_info",
                "language": language,
                "youtube_url": yt_url,
                "video_id": vid,
                "info": info,
                "transcript_used": bool(transcript_text),
                "transcript_error": transcript_error or None,
            }
        ), 200

    study_prompt = f"""
You are a friendly, expert study assistant for students.

- Explain concepts clearly and step-by-step.
- When solving problems, show all important steps.
- Use simple language and examples when needed.
- Always reply in {language}.

Student message:
{message}
"""
    try:
        answer = model_chat(study_prompt, max_tokens=1000, temperature=0.3)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    if session_id:
        _save_chat(session_id, "ai", "study_chat", answer, None)

    return jsonify(
        {
            "status": "ok",
            "mode": "chat",
            "language": language,
            "answer": answer,
        }
    ), 200


# ============ MULTIPLE QUESTION PAPERS → IMPORTANT QUESTIONS ============


@app.route("/important_questions", methods=["POST"])
def important_questions():
    files = request.files.getlist("papers")
    if not files:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "No files uploaded. Use 'papers' as field name with multiple files.",
                }
            ),
            400,
        )

    language = (request.form.get("language") or "English").strip()
    try:
        num_q = int(request.form.get("num_questions", "20"))
    except ValueError:
        num_q = 20

    all_chunks = []

    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue

        uid = uuid.uuid4().hex
        saved_path = UPLOAD_DIR / f"{uid}{ext}"
        f.save(saved_path)

        try:
            chunks = build_chunks_for_file(str(saved_path))
            all_chunks.extend(chunks)
        except Exception as e:
            current_app.logger.warning("Error processing file %s: %s", saved_path, e)

    if not all_chunks:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Could not extract text from any uploaded question paper.",
                }
            ),
            400,
        )

    max_chunks = 40
    combined_text = "\n\n---\n\n".join(all_chunks[:max_chunks])

    try:
        shortlisted = shortlist_important_questions(
            combined_text, num_q=num_q, language=language
        )
    except Exception as e:
        current_app.logger.exception("Error during important question shortlisting")
        return jsonify({"status": "error", "error": str(e)}), 500

    return (
        jsonify(
            {
                "status": "ok",
                "language": language,
                "requested_num_questions": num_q,
                "important_questions": shortlisted,
            }
        ),
        200,
    )


@app.route("/important", methods=["GET"])
def important_questions_page():
    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""
    return render_template(
        "important_questions.html",
        example_name=ex_name,
        active="important",
    )


# ============ HISTORY PAGE ============


@app.route("/history", methods=["GET"])
def history():
    session_id = getattr(g, "session_id", None)

    if not session_id:
        docs = []
        messages = []
    else:
        docs = (
            UploadedDocument.query.filter_by(session_id=session_id)
            .order_by(UploadedDocument.uploaded_at.desc())
            .limit(20)
            .all()
        )
        messages = (
            ChatMessage.query.filter_by(session_id=session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(40)
            .all()
        )

    ex_name = Path(EXAMPLE_PATH).name if EXAMPLE_PATH else ""

    return render_template(
        "history.html",
        docs=docs,
        messages=messages,
        example_name=ex_name,
        active="history",
    )


# ============ EXAMPLE + UPLOAD SERVING ============


@app.route("/use_example", methods=["GET", "POST"])
def use_example():
    ex = Path(EXAMPLE_PATH)
    if not ex.exists():
        return (
            jsonify(
                {"error": "example file not found on disk", "path": str(EXAMPLE_PATH)}
            ),
            404,
        )
    dest = UPLOAD_DIR / ex.name
    if not dest.exists():
        import shutil

        shutil.copy(ex, dest)
    return redirect(url_for("process_file", filename=dest.name))


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# ============ MAIN ============

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run("0.0.0.0", port=8000, debug=True)
