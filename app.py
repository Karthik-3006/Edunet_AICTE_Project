from flask import Flask, render_template, request, jsonify, session
import uuid
from questions import get_questions
from granite_api import (
    evaluate_answer, generate_suggested_answer,
    analyze_resume, analyze_skills, generate_roadmap,
    extract_skills_from_resume, parse_resume_profile,
)
from file_parser import extract_text, allowed_file, ACCEPTED_MIME_TYPES

app = Flask(__name__)
app.secret_key = "ai_interview_trainer_secret_key_2024"


# ── Portal / Landing pages ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("portal.html", active_page="home")


@app.route("/get_started")
def get_started():
    return render_template("get_started.html", active_page="home")


@app.route("/start_now")
def start_now():
    return render_template("start_now.html", active_page="home")


@app.route("/guidelines")
def guidelines():
    return render_template("guidelines.html", active_page="guidelines")


@app.route("/faqs")
def faqs():
    return render_template("faqs.html", active_page="faqs")


@app.route("/interview_setup")
def interview_setup():
    return render_template("interview_setup.html", active_page="home")


@app.route("/resume_analyzer")
def resume_analyzer():
    return render_template("resume_analyzer.html", active_page="resume")


@app.route("/skill_analyzer")
def skill_analyzer():
    return render_template("skill_analyzer.html", active_page="skills")


# ── File Upload (multi-format) ────────────────────────────────────────────────

@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    Accept a binary file upload (PDF, DOCX, DOC, TXT, images) and return
    the extracted plain text so the frontend can populate the resume textarea.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "Empty file."}), 400

    if not allowed_file(f.filename):
        return jsonify({
            "error": (
                "Unsupported file type. Please upload a PDF, Word document (.docx), "
                "plain text (.txt), or image (PNG, JPG, JPEG, WEBP, BMP, TIFF)."
            )
        }), 400

    file_bytes = f.read()
    text = extract_text(file_bytes, f.filename)

    if text.startswith("[Error]"):
        return jsonify({"error": text[len("[Error] "):]}), 422

    if not text.strip():
        return jsonify({"error": "No text could be extracted from this file. Please paste your resume text manually."}), 422

    return jsonify({"text": text, "filename": f.filename})


# ── AI Analysis APIs ──────────────────────────────────────────────────────────

@app.route("/parse_resume", methods=["POST"])
def parse_resume_api():
    """Parse a resume and return auto-filled profile fields (name, role, experience, skills)."""
    data        = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    if not resume_text:
        return jsonify({"error": "Resume text is required."}), 400
    result = parse_resume_profile(resume_text)
    return jsonify(result)


@app.route("/analyze_resume", methods=["POST"])
def analyze_resume_api():
    """Analyze a resume against a target role using IBM Granite."""
    data        = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    job_role    = data.get("job_role", "").strip()
    experience  = data.get("experience", "Entry Level (0-2 years)").strip()

    if not resume_text:
        return jsonify({"error": "Resume text is required."}), 400
    if not job_role:
        return jsonify({"error": "Job role is required."}), 400

    result = analyze_resume(resume_text, job_role, experience)
    return jsonify(result)


@app.route("/analyze_skills", methods=["POST"])
def analyze_skills_api():
    """Analyze a skill set for market competitiveness using IBM Granite."""
    data        = request.get_json()
    skills      = data.get("skills", [])
    target_role = data.get("target_role", "").strip()
    experience  = data.get("experience", "Entry Level (0-2 years)").strip()

    if not skills:
        return jsonify({"error": "At least one skill is required."}), 400

    result = analyze_skills(skills, target_role, experience)
    return jsonify(result)


@app.route("/generate_roadmap", methods=["POST"])
def generate_roadmap_api():
    """Generate a personalised learning roadmap using IBM Granite."""
    data   = request.get_json()
    domain = data.get("domain", "").strip()
    level  = data.get("level", "Complete Beginner").strip()
    goal   = data.get("goal", "").strip()

    if not domain:
        return jsonify({"error": "Domain is required."}), 400

    result = generate_roadmap(domain, level, goal)
    return jsonify(result)


# ── Interview flow ────────────────────────────────────────────────────────────

@app.route("/start_interview", methods=["POST"])
def start_interview():
    """Start a mock interview session (manual profile entry)."""
    data           = request.get_json()
    name           = data.get("name", "").strip()
    job_role       = data.get("job_role", "").strip()
    experience     = data.get("experience", "").strip()
    interview_type = data.get("interview_type", "Mixed").strip()

    if not all([name, job_role, experience]):
        return jsonify({"error": "Please fill in all required fields."}), 400

    questions = get_questions(job_role, experience, interview_type)
    _save_session(name, job_role, experience, interview_type, questions)

    return jsonify({
        "status":          "started",
        "total_questions": len(questions),
        "question":        questions[0],
        "question_number": 1,
    })


@app.route("/start_interview_with_resume", methods=["POST"])
def start_interview_with_resume():
    """
    Start a mock interview session with resume upload.
    Questions are generated directly from the resume content and target role/industry.
    """
    data           = request.get_json()
    name           = data.get("name", "").strip()
    job_role       = data.get("job_role", "").strip()
    experience     = data.get("experience", "").strip()
    interview_type = data.get("interview_type", "Mixed").strip()
    resume_text    = data.get("resume_text", "").strip()
    industry       = data.get("industry", "").strip()

    if not resume_text:
        return jsonify({"error": "Resume text is required."}), 400

    # Parse profile from resume if fields not explicitly provided
    if not name or not job_role or not experience:
        profile = parse_resume_profile(resume_text)
        name           = name           or profile.get("name", "Candidate")
        job_role       = job_role       or profile.get("job_role", "Software Engineer")
        experience     = experience     or profile.get("experience", "Entry Level (0-2 years)")
        interview_type = interview_type or profile.get("interview_type", "Mixed")

    # Extract skills list for the session summary display
    resume_skills = extract_skills_from_resume(resume_text, job_role)

    # Generate questions grounded in the actual resume + target role/industry
    questions = get_questions(
        job_role, experience, interview_type,
        resume_text=resume_text,
        industry=industry or job_role,
    )
    _save_session(name, job_role, experience, interview_type, questions,
                  resume_text=resume_text, resume_skills=resume_skills)

    return jsonify({
        "status":          "started",
        "total_questions": len(questions),
        "question":        questions[0],
        "question_number": 1,
        "detected_skills": resume_skills,
    })


def _save_session(name, job_role, experience, interview_type, questions,
                  resume_text="", resume_skills=None):
    session["interview"] = {
        "id":             str(uuid.uuid4()),
        "name":           name,
        "job_role":       job_role,
        "experience":     experience,
        "interview_type": interview_type,
        "questions":      questions,
        "current_index":  0,
        "answers":        [],
        "scores":         [],
        "resume_text":    resume_text,
        "resume_skills":  resume_skills or [],
    }


@app.route("/interview")
def interview():
    """Interview page — served only when a session is active."""
    if not session.get("interview"):
        return render_template("interview_setup.html", active_page="home")
    return render_template("interview.html")


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    """Evaluate the current answer and return feedback + next question."""
    interview = session.get("interview")
    if not interview:
        return jsonify({"error": "No active interview session found."}), 400

    data   = request.get_json()
    answer = data.get("answer", "").strip()
    if not answer:
        return jsonify({"error": "Answer cannot be empty."}), 400

    current_index    = interview["current_index"]
    questions        = interview["questions"]
    current_question = questions[current_index]

    evaluation = evaluate_answer(
        question   = current_question,
        answer     = answer,
        job_role   = interview["job_role"],
        experience = interview["experience"],
    )
    suggested = generate_suggested_answer(
        question   = current_question,
        job_role   = interview["job_role"],
        experience = interview["experience"],
    )

    interview["answers"].append({"question": current_question, "answer": answer})
    interview["scores"].append(evaluation.get("score", 0))
    interview["current_index"] += 1
    session["interview"] = interview

    next_index = interview["current_index"]
    is_last    = next_index >= len(questions)

    response = {
        "evaluation":       evaluation,
        "suggested_answer": suggested,
        "question_number":  current_index + 1,
        "is_last":          is_last,
    }
    if not is_last:
        response["next_question"]        = questions[next_index]
        response["next_question_number"] = next_index + 1

    return jsonify(response)


@app.route("/summary")
def summary():
    """Final summary page."""
    interview = session.get("interview")
    if not interview:
        return render_template("portal.html", active_page="home")

    scores  = interview.get("scores", [])
    overall = round(sum(scores) / len(scores), 1) if scores else 0

    return render_template(
        "summary.html",
        interview     = interview,
        overall_score = overall,
    )


@app.route("/reset")
def reset():
    session.pop("interview", None)
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
