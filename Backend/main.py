import os
import json
import io
import secrets
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Backend.database import (
    create_tables,
    get_db_connection,
    hash_password,
    verify_password,
)
from Backend.embeddings import initialize_embedding_model
from Backend.resume import process_resume
from Backend.job import create_job
from Backend.matching import get_top_candidates
from Backend.llm import DEFAULT_MODEL, SUPPORTED_MODELS, generate_interview_kit_llm
from Backend.pdf_parser import extract_document_text


app = FastAPI(
    title="AI Recruitment Portal API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    create_tables()
    initialize_embedding_model()


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Recruitment Portal API is active"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}



# -------------------------
# Auth Models & Endpoints
# -------------------------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: Optional[str] = "recruiter"


class LoginRequest(BaseModel):
    email: str
    password: str


def _get_current_user_id(
    authorization: Optional[str] = None,
    x_user_id: Optional[str] = None
) -> Optional[int]:
    """Resolve current user ID from session token or X-User-Id header."""
    if authorization and "Bearer " in authorization:
        token = authorization.split("Bearer ")[1].strip()
        connection = get_db_connection()
        try:
            session = connection.execute(
                "SELECT user_id FROM user_sessions WHERE token = ?", (token,)
            ).fetchone()
            if session:
                return session["user_id"]
        finally:
            connection.close()

    if x_user_id:
        try:
            return int(x_user_id)
        except (ValueError, TypeError):
            pass

    return None


@app.post("/auth/register")
def register_user(request: RegisterRequest):
    name = request.name.strip()
    email = request.email.strip().lower()
    password = request.password

    if not name:
        raise HTTPException(status_code=400, detail="Full name is required.")
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    connection = get_db_connection()
    try:
        existing = connection.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists. Please sign in instead.",
            )

        pwd_hash, salt = hash_password(password)
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, salt, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, email, pwd_hash, salt, request.role or "recruiter"),
        )
        user_id = cursor.lastrowid

        token = f"token_{secrets.token_urlsafe(32)}"
        cursor.execute(
            "INSERT INTO user_sessions (token, user_id) VALUES (?, ?)",
            (token, user_id)
        )
        connection.commit()

        return {
            "success": True,
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": user_id,
                "name": name,
                "email": email,
                "role": request.role or "recruiter",
            },
        }
    finally:
        connection.close()


@app.post("/auth/login")
def login_user(request: LoginRequest):
    email = request.email.strip().lower()
    password = request.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    connection = get_db_connection()
    try:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if not user or not verify_password(password, user["password_hash"], user["salt"]):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password. Please register first if you do not have an account.",
            )

        token = f"token_{secrets.token_urlsafe(32)}"
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO user_sessions (token, user_id) VALUES (?, ?)",
            (token, user["id"])
        )
        connection.commit()

        return {
            "token": token,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
            },
        }
    finally:
        connection.close()


# -------------------------
# Request Models
# -------------------------

class JobRequest(BaseModel):
    title: str
    description: str


class InterviewKitRequest(BaseModel):
    job_title: str
    job_description: str
    candidate_name: str
    candidate_resume: str
    candidate_experience: float
    model: Optional[str] = "llama-3.3-70b-versatile"


class CandidateStatusRequest(BaseModel):
    status: Literal[
        "New", "Under Review", "Shortlisted", "Interview Scheduled", "Selected", "Rejected"
    ]


def _parse_json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except json.JSONDecodeError:
        return fallback


def _candidate_response(row, match=None):
    row_keys = row.keys() if hasattr(row, 'keys') else []
    phone = row["phone"] if "phone" in row_keys else ""
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"] or "",
        "phone": phone or "",
        "skills": _parse_json(row["skills"], []),
        "experience": row["experience"] or 0,
        "status": row["status"],
        "summary": row["summary"] or "",
        "resume_file": row["resume_filename"],
        "date_added": row["uploaded_at"],
        "match_score": match["match_score"] if match else None,
    }


def _job_response(row, candidate_count=0):
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "skills": _parse_json(row["skills"], []),
        "keywords": _parse_json(row["keywords"], []),
        "candidate_count": candidate_count,
    }


# -------------------------
# Home & Status
# -------------------------

@app.get("/")
def home():
    return {
        "message": "AI Recruitment Portal Backend is running",
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }


@app.get("/api/groq/status")
def groq_status():
    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    masked_key = ""
    if api_key_set:
        raw = os.getenv("GROQ_API_KEY", "")
        masked_key = raw[:6] + "..." + raw[-4:] if len(raw) > 10 else "***"

    return {
        "configured": api_key_set,
        "masked_key": masked_key,
        "default_model": DEFAULT_MODEL,
        "supported_models": SUPPORTED_MODELS,
    }


# -------------------------
# AI Interview Kit Generation
# -------------------------

@app.post("/interview/generate")
def generate_interview_kit(request: InterviewKitRequest):
    """
    Generate customized interview questions using Groq calibrated
    against candidate resume, job description, and years of experience.
    """
    try:
        result = generate_interview_kit_llm(
            job_title=request.job_title,
            job_description=request.job_description,
            candidate_name=request.candidate_name,
            candidate_resume=request.candidate_resume,
            candidate_experience=request.candidate_experience,
            model=request.model or DEFAULT_MODEL
        )
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# -------------------------
# Upload Resumes
# -------------------------

@app.post("/resumes/upload")
async def upload_resumes(
    files: list[UploadFile] = File(...),
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    successful = []
    failed = []

    for file in files:
        try:
            result = process_resume(file, user_id=user_id)
            successful.append(result)

        except Exception as error:
            failed.append({
                "filename": file.filename,
                "error": str(error)
            })

    return {
        "total_uploaded": len(files),
        "successful_resumes": successful,
        "failed_resumes": failed
    }


# -------------------------
# Candidates
# -------------------------

@app.get("/candidates")
def list_candidates(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        rows = connection.execute(
            "SELECT * FROM candidates WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    else:
        rows = connection.execute("SELECT * FROM candidates ORDER BY id DESC").fetchall()
    candidates = []
    for row in rows:
        match = connection.execute(
            "SELECT * FROM candidate_matches WHERE candidate_id = ? ORDER BY matched_at DESC, id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        candidates.append(_candidate_response(row, match))
    connection.close()
    return {"data": candidates}


@app.get("/candidates/{candidate_id}")
def get_candidate(
    candidate_id: int,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        row = connection.execute(
            "SELECT * FROM candidates WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (candidate_id, user_id),
        ).fetchone()
    else:
        row = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if row is None:
        connection.close()
        raise HTTPException(status_code=404, detail="Candidate not found")
    match = connection.execute(
        "SELECT * FROM candidate_matches WHERE candidate_id = ? ORDER BY matched_at DESC, id DESC LIMIT 1",
        (candidate_id,),
    ).fetchone()
    connection.close()
    return {"data": _candidate_response(row, match)}


@app.patch("/candidates/{candidate_id}/status")
def update_candidate_status(
    candidate_id: int,
    request: CandidateStatusRequest,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        cursor = connection.execute(
            "UPDATE candidates SET status = ? WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (request.status, candidate_id, user_id),
        )
    else:
        cursor = connection.execute(
            "UPDATE candidates SET status = ? WHERE id = ?", (request.status, candidate_id)
        )
    if cursor.rowcount == 0:
        connection.close()
        raise HTTPException(status_code=404, detail="Candidate not found")
    connection.commit()
    row = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    match = connection.execute(
        "SELECT * FROM candidate_matches WHERE candidate_id = ? ORDER BY matched_at DESC, id DESC LIMIT 1",
        (candidate_id,),
    ).fetchone()
    connection.close()
    return {"data": _candidate_response(row, match)}


@app.get("/candidates/{candidate_id}/match")
def get_candidate_match(
    candidate_id: int,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        candidate = connection.execute(
            "SELECT * FROM candidates WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (candidate_id, user_id),
        ).fetchone()
    else:
        candidate = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if candidate is None:
        connection.close()
        raise HTTPException(status_code=404, detail="Candidate not found")
    match = connection.execute(
        "SELECT * FROM candidate_matches WHERE candidate_id = ? ORDER BY matched_at DESC, id DESC LIMIT 1",
        (candidate_id,),
    ).fetchone()
    if match is None:
        connection.close()
        raise HTTPException(status_code=404, detail="No match result is available for this candidate")
    job = connection.execute("SELECT skills FROM jobs WHERE id = ?", (match["job_id"],)).fetchone()
    job_skills = _parse_json(job["skills"] if job else None, [])
    candidate_skills = _parse_json(candidate["skills"], [])
    candidate_skill_index = {skill.lower(): skill for skill in candidate_skills}
    matched_skills = [skill for skill in job_skills if skill.lower() in candidate_skill_index]
    missing_skills = [skill for skill in job_skills if skill.lower() not in candidate_skill_index]
    connection.close()
    return {
        "data": {
            "overall_score": match["match_score"],
            "skill_match": match["skill_score"],
            "semantic_match": match["semantic_score"],
            "keyword_match": match["keyword_score"],
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }
    }


@app.delete("/candidates/{candidate_id}")
def delete_candidate(
    candidate_id: int,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    """Delete a candidate record, match history, and uploaded resume asset."""
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    try:
        if user_id:
            candidate = connection.execute(
                "SELECT * FROM candidates WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
                (candidate_id, user_id),
            ).fetchone()
        else:
            candidate = connection.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found")

        resume_filename = candidate["resume_filename"]
        if resume_filename:
            file_path = os.path.join("uploads/resumes", resume_filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        connection.execute("DELETE FROM candidate_matches WHERE candidate_id = ?", (candidate_id,))
        connection.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        connection.commit()
        return {
            "success": True,
            "message": f"Candidate #{candidate_id} and associated resume deleted successfully."
        }
    finally:
        connection.close()


# -------------------------
# Create Job
# -------------------------

@app.post("/jobs")
def create_new_job(
    job: JobRequest,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    if not job.title.strip() or not job.description.strip():
        raise HTTPException(status_code=422, detail="Job title and description are required")
    user_id = _get_current_user_id(authorization, x_user_id)
    try:
        result = create_job(job.title, job.description, user_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"data": result}


@app.post("/jobs/from-file")
async def create_job_from_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    """Create a job from a PDF, DOCX, or UTF-8 text job-description file."""
    raw_filename = file.filename or "job-description.pdf"
    clean_filename = os.path.basename(raw_filename.strip().strip("\"'"))
    if not clean_filename:
        clean_filename = "job-description.pdf"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded job description file is empty.")

    description = extract_document_text(clean_filename, content)

    if not description:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract readable text from '{clean_filename}'. Please ensure the file contains selectable text, or paste the text directly."
        )

    derived_title = os.path.splitext(clean_filename)[0].replace("_", " ").replace("-", " ")
    user_id = _get_current_user_id(authorization, x_user_id)
    try:
        result = create_job(title.strip() or derived_title, description, user_id=user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"data": result}


@app.get("/jobs")
def list_jobs(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        rows = connection.execute(
            "SELECT * FROM jobs WHERE user_id = ? OR user_id IS NULL ORDER BY id DESC", (user_id,)
        ).fetchall()
    else:
        rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    jobs = []
    for row in rows:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM candidate_matches WHERE job_id = ?", (row["id"],)
        ).fetchone()["count"]
        jobs.append(_job_response(row, count))
    connection.close()
    return {"data": jobs}


@app.get("/jobs/{job_id}")
def get_job(
    job_id: int,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ? AND (user_id = ? OR user_id IS NULL)", (job_id, user_id)
        ).fetchone()
    else:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        connection.close()
        raise HTTPException(status_code=404, detail="Job not found")
    count = connection.execute(
        "SELECT COUNT(*) AS count FROM candidate_matches WHERE job_id = ?", (job_id,)
    ).fetchone()["count"]
    connection.close()
    return {"data": _job_response(row, count)}


# -------------------------
# Match Candidates
# -------------------------

@app.get("/matching/{job_id}")
def match_candidates(
    job_id: int,
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    try:
        matches = get_top_candidates(job_id, user_id=user_id)
        return {
            "job_id": job_id,
            "matches": matches
        }

    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


# -------------------------
# Dashboard
# -------------------------

@app.get("/dashboard")
def get_dashboard(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
):
    user_id = _get_current_user_id(authorization, x_user_id)
    connection = get_db_connection()
    if user_id:
        stats = {
            "total_candidates": connection.execute(
                "SELECT COUNT(*) AS count FROM candidates WHERE user_id = ?", (user_id,)
            ).fetchone()["count"],
            "total_jobs": connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE user_id = ? OR user_id IS NULL", (user_id,)
            ).fetchone()["count"],
            "shortlisted_candidates": connection.execute(
                "SELECT COUNT(*) AS count FROM candidates WHERE status = 'Shortlisted' AND user_id = ?", (user_id,)
            ).fetchone()["count"],
            "interviews_scheduled": connection.execute(
                "SELECT COUNT(*) AS count FROM candidates WHERE status = 'Interview Scheduled' AND user_id = ?", (user_id,)
            ).fetchone()["count"],
        }
        candidate_rows = connection.execute(
            "SELECT * FROM candidates WHERE user_id = ? ORDER BY id DESC LIMIT 4", (user_id,)
        ).fetchall()
        job_rows = connection.execute(
            "SELECT * FROM jobs WHERE user_id = ? OR user_id IS NULL ORDER BY id DESC LIMIT 3", (user_id,)
        ).fetchall()
    else:
        stats = {
            "total_candidates": connection.execute("SELECT COUNT(*) AS count FROM candidates").fetchone()["count"],
            "total_jobs": connection.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()["count"],
            "shortlisted_candidates": connection.execute(
                "SELECT COUNT(*) AS count FROM candidates WHERE status = 'Shortlisted'"
            ).fetchone()["count"],
            "interviews_scheduled": connection.execute(
                "SELECT COUNT(*) AS count FROM candidates WHERE status = 'Interview Scheduled'"
            ).fetchone()["count"],
        }
        candidate_rows = connection.execute("SELECT * FROM candidates ORDER BY id DESC LIMIT 4").fetchall()
        job_rows = connection.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 3").fetchall()

    recent_candidates = []
    for row in candidate_rows:
        match = connection.execute(
            "SELECT * FROM candidate_matches WHERE candidate_id = ? ORDER BY matched_at DESC, id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        recent_candidates.append(_candidate_response(row, match))

    recent_jobs = []
    for row in job_rows:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM candidate_matches WHERE job_id = ?", (row["id"],)
        ).fetchone()["count"]
        recent_jobs.append(_job_response(row, count))
    connection.close()
    return {"data": {"stats": stats, "recent_jobs": recent_jobs, "recent_candidates": recent_candidates}}
