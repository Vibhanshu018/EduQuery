import os, io, json, base64, re, uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

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
    session,
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

# ----------------- Configuration -----------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

EXAMPLE_PATH = os.getenv("EXAMPLE_FILE", "/mnt/data/OS_Unit_1_(Notes)[1].pdf")

# ----------------- Flask + Database Setup -----------------
app = Flask(__name__, template_folder="templates")

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///rag_study_buddy.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ----------------- Models -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=False)
    name = db.Column(db.String, nullable=False)
    password_hash = db.Column(db.String, nullable=False)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserSession(db.Model):
    id = db.Column(db.String, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class UploadedDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String, db.ForeignKey('user_session.id'), nullable=False)
    filename = db.Column(db.String, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UploadedDocument {self.session_id} {self.filename}>"

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String, db.ForeignKey('user_session.id'), nullable=False)
    sender = db.Column(db.String, nullable=False)
    message = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ----------------- Initialize Database (only if not exists) -----------------
with app.app_context():
    db.create_all()

# ----------------- RAG + OCR + Embedding Imports -----------------
from model_client import chat as model_chat, generate_embeddings, embed_query
import fitz, pypdf
from PIL import Image
import numpy as np

try:
    import easyocr
    _ocr_reader = easyocr.Reader(os.getenv("OCR_LANGS", "en").split(","), gpu=False)
except:
    _ocr_reader = None

# ----------------- Helper Functions -----------------
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    pages = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            pages.append(text if text else "")
    except:
        return []
    return pages

def extract_images_from_pdf(pdf_path: str, max_images_per_page=2) -> List[Image.Image]:
    images = []
    try:
        doc = fitz.open(pdf_path)
    except:
        return images

    for page in doc:
        imgs = page.get_images(full=True)
        for i, img_item in enumerate(imgs):
            if i >= max_images_per_page:
                break
            try:
                xref = img_item[0]
                pix = fitz.Pixmap(doc, xref)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except:
                continue
    return images

def ocr_image_local(img: Image.Image) -> str:
    if not _ocr_reader:
        return ""
    try:
        arr = np.array(img)
        texts = _ocr_reader.readtext(arr, detail=0)
        return "\n".join(texts)
    except:
        return ""

def chunk_texts(pages: List[str], chunk_size=1000, overlap=200) -> List[str]:
    chunks = []
    for text in pages:
        text = text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
    return chunks

def build_chunks_for_file(file_path: str) -> List[str]:
    ext = Path(file_path).suffix.lower()
    pages = []

    if ext == ".pdf":
        texts = extract_text_from_pdf(file_path)
        pages.extend(texts)
        images = extract_images_from_pdf(file_path, max_images_per_page=1)
        for img in images[:2]:
            ocr_text = ocr_image_local(img)
            if ocr_text:
                pages.append(ocr_text)

    elif ext in {".txt", ".md"}:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pages.append(f.read())
        except:
            return []

    return chunk_texts(pages)

def similarity_search(chunks: List[str], query: str, top_k=5) -> List[str]:
    if not chunks:
        return []
    try:
        db_vecs = np.array(generate_embeddings(chunks))
        q_vec = np.array(embed_query(query))
        sims = db_vecs.dot(q_vec) / (np.linalg.norm(db_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-12)
        idxs = np.argsort(-sims)[:top_k]
        return [chunks[i] for i in idxs]
    except:
        return []

# --------------- Routes for Frontend + Backend APIs -----------------

@app.before_request
def load_or_create_session():
    sid = request.cookies.get("rag_session_id")
    new_session = False

    if not sid:
        sid = uuid.uuid4().hex
        new_session = True

    g.session_id = sid
    g.new_session = new_session

    if not UserSession.query.get(sid):
        us = UserSession(id=sid)
        db.session.add(us)
        db.session.commit()

    user_id = session.get("user_id")
    g.user = User.query.get(user_id) if user_id else None

    us = UserSession.query.get(sid)
    if g.user and us.user_id != g.user.id:
        us.user_id = g.user.id
        db.session.commit()
    elif not g.user and us.user_id:
        us.user_id = None
        db.session.commit()

@app.after_request
def set_session_cookie(response):
    if getattr(g, "new_session", False):
        response.set_cookie(
            "rag_session_id",
            g.session_id,
            max_age=60 * 60 * 24 * 30,
            httponly=True,
        )
    return response

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", active="home", user=getattr(g, "user", None))

@app.route("/study", methods=["GET"])
def study_page():
    return render_template("study.html", active="study", user=getattr(g, "user", None))

@app.route("/history", methods=["GET"])
def history():
    sid = getattr(g, "session_id", None)
    docs = UploadedDocument.query.filter_by(session_id=sid).order_by(UploadedDocument.uploaded_at.desc()).limit(20).all()
    messages = ChatMessage.query.filter_by(session_id=sid).order_by(ChatMessage.timestamp.desc()).limit(40).all()

    return render_template(
        "history.html",
        docs=docs,
        messages=messages[::-1],
        active="history",
        user=getattr(g, "user", None),
    )

@app.route("/auth", methods=["GET"])
def auth_page():
    return render_template("auth.html", active="auth", user=getattr(g, "user", None))

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return render_template("auth.html", error="Invalid email or password", user=None)

    session["user_id"] = user.id
    return redirect(url_for("index"))

@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if User.query.filter_by(email=email).first():
        return render_template("auth.html", error="Email already registered", user=None)

    password_hash = generate_password_hash(password)
    user = User(name=name, email=email, password_hash=password_hash)

    db.session.add(user)
    db.session.commit()

    session["user_id"] = user.id
    return redirect(url_for("index"))

@app.route("/logout", methods=["GET"])
def logout():
    session.pop("user_id", None)
    return redirect(url_for("index"))

@app.route("/guest", methods=["GET"])
def continue_as_guest():
    session.pop("user_id", None)
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type {ext}"}), 400

    fname = uuid.uuid4().hex + ext
    save_path = UPLOAD_DIR / fname
    file.save(save_path)

    sid = getattr(g, "session_id", None)
    if sid:
        doc = UploadedDocument(session_id=sid, filename=fname)
        db.session.add(doc)
        db.session.commit()

    return redirect(url_for("study_page"))

@app.route("/api/chat", methods=["POST"])
def chat_api():
    sid = getattr(g, "session_id", None)
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    past = UploadedDocument.query.filter_by(session_id=sid).order_by(UploadedDocument.uploaded_at.desc()).limit(5).all()
    contexts = []
    for d in past:
        c = build_chunks_for_file(str(UPLOAD_DIR / d.filename))
        if c:
            contexts.extend(similarity_search(c, query, 3))

    ctx_text = "\n\n".join(contexts[:6])
    prompt = f"Answer using context:\n{ctx_text}\n\nQ: {query}"
    answer = model_chat(prompt)

    if sid:
        db.session.add(ChatMessage(session_id=sid, sender="user", message=query))
        db.session.add(ChatMessage(session_id=sid, sender="assistant", message=answer))
        db.session.commit()

    return jsonify({"answer": answer})

@app.route("/api/gen-questions/<filename>", methods=["GET"])
def gen_questions_api(filename):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404

    chunks = build_chunks_for_file(str(path))
    if not chunks:
        return jsonify({"error": "No text extracted"}), 400

    ai_prompt = "Generate short, long, MCQ questions from the text."
    response = model_chat("\n\n".join(chunks) + "\n" + ai_prompt)

    try:
        qdata = json.loads(re.search(r"(\{.*\}|\[.*\])", response).group(1))
    except:
        qdata = {"raw": response}

    return jsonify({"questions": qdata})

# ---------------- Vercel Entry Point -----------------
def main(event=None, context=None):
    return app

if __name__ == "__main__":
    app.run()
