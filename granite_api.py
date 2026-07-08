"""
granite_api.py
--------------
IBM Granite integration via the ibm-watsonx-ai SDK.

Credentials are loaded from a .env file (or real environment variables):
    WATSONX_API_KEY    – IBM Cloud API key
    WATSONX_PROJECT_ID – watsonx.ai project ID
    WATSONX_URL        – regional endpoint  (default: us-south)

If credentials are missing or the API call fails, every public function
gracefully falls back to a sensible stub response so the app stays usable
during development.
"""

import json
import logging
import os
import random
import re

from dotenv import load_dotenv

load_dotenv()  # reads .env next to app.py

logger = logging.getLogger(__name__)

# ── Credential constants ──────────────────────────────────────────────────────
_API_KEY    = os.getenv("WATSONX_API_KEY", "")
_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# ── Lazy singleton — model is created once on first use ──────────────────────
_model = None


def _get_model():
    """
    Return a cached ModelInference instance.
    Returns None when credentials are absent so callers fall back to stubs.
    """
    global _model
    if _model is not None:
        return _model

    _placeholder = {"", "your_ibm_cloud_api_key_here", "your_watsonx_project_id_here"}
    if not _API_KEY or not _PROJECT_ID or _API_KEY in _placeholder or _PROJECT_ID in _placeholder:
        logger.warning(
            "IBM Granite credentials not configured. "
            "Edit .env and set real WATSONX_API_KEY and WATSONX_PROJECT_ID values."
        )
        return None

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        credentials = Credentials(url=_URL, api_key=_API_KEY)
        client      = APIClient(credentials)

        _model = ModelInference(
            model_id   = ModelTypes.GRANITE_13B_CHAT_V2,
            api_client = client,
            project_id = _PROJECT_ID,
            params     = {
                Params.MAX_NEW_TOKENS: 600,
                Params.TEMPERATURE:    0.7,
                Params.TOP_P:          0.9,
                Params.STOP_SEQUENCES: ["```"],
            },
        )
        logger.info("IBM Granite model initialised successfully.")
        return _model

    except Exception as exc:
        logger.error("Failed to initialise IBM Granite model: %s", exc)
        return None


# ── Low-level call ────────────────────────────────────────────────────────────

def _call_granite(prompt: str) -> str:
    """
    Send *prompt* to IBM Granite and return the raw response text.
    Returns an empty string on any error so callers can fall back.
    """
    model = _get_model()
    if model is None:
        return ""
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else ""
    except Exception as exc:
        logger.error("IBM Granite API call failed: %s", exc)
        return ""


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """
    Try to parse a JSON object from *text*.
    Handles both clean JSON and JSON embedded inside markdown code fences.
    Returns None if parsing fails.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Stub helpers (used when Granite is unavailable) ──────────────────────────

_STRENGTHS_POOL = [
    "Clear and structured communication",
    "Good use of relevant examples",
    "Demonstrated solid technical knowledge",
    "Showed a methodical problem-solving approach",
    "Mentioned industry best practices",
    "Highlighted teamwork and collaboration",
    "Linked the answer to business value",
]

_WEAKNESSES_POOL = [
    "Could elaborate more on technical details",
    "Lacked specific metrics or measurable outcomes",
    "Answer could be more concise",
    "Missing discussion of edge cases or failure modes",
    "Could connect the answer better to real-world scenarios",
    "Consider using the STAR method for behavioural questions",
    "More depth on trade-offs would strengthen the answer",
]


def _score_answer_locally(question: str, answer: str) -> int:
    """
    Heuristic scoring used when Granite is unavailable.
    Penalises: very short answers, obvious filler phrases, off-topic or 'I don't know' replies.
    Rewards:   length, presence of technical keywords, structured responses.
    """
    ans = answer.strip().lower()

    # ── Hard penalties ────────────────────────────────────────────
    dont_know_phrases = [
        "i don't know", "i do not know", "idk", "no idea", "not sure",
        "i have no idea", "i cannot answer", "i can't answer",
        "don't know", "no clue",
    ]
    if any(p in ans for p in dont_know_phrases):
        return random.randint(0, 2)

    wrong_indicators = [
        "nothing", "skip", "pass", "n/a", "na", "none", "no answer",
        "abc", "xyz", "test", "hello", "hi", "ok", "okay",
    ]
    if ans in wrong_indicators or len(ans) < 15:
        return random.randint(1, 3)

    # ── Length scoring baseline ───────────────────────────────────
    word_count = len(answer.split())

    if word_count < 10:
        base = random.randint(1, 3)
    elif word_count < 25:
        base = random.randint(3, 5)
    elif word_count < 60:
        base = random.randint(5, 7)
    elif word_count < 120:
        base = random.randint(6, 8)
    else:
        base = random.randint(7, 9)

    # ── Positive keyword bonus (max +1) ──────────────────────────
    positive_kw = [
        "example", "for instance", "such as", "because", "therefore",
        "however", "result", "achieved", "implemented", "designed",
        "improved", "reduced", "increased", "team", "project", "solution",
        "star", "situation", "task", "action", "used", "built", "deployed",
    ]
    bonus = 1 if any(kw in ans for kw in positive_kw) else 0

    return min(10, base + bonus)


def _stub_evaluation(job_role: str, question: str = "", answer: str = "") -> dict:
    """
    Fallback evaluation when Granite is unavailable.
    Score is derived from actual answer content — NOT random high scores.
    """
    score = _score_answer_locally(question, answer)

    if score <= 2:
        strengths  = ["Attempted to respond"]
        weaknesses = ["Answer is incorrect or too vague", "No relevant technical content provided",
                      "Does not address the question asked"]
        feedback   = (f"This answer scored {score}/10. It does not sufficiently address the question. "
                      "Please revisit the topic and try again with specific examples and correct information.")
    elif score <= 4:
        strengths  = ["Made an attempt to answer the question"]
        weaknesses = ["Answer lacks technical depth", "Missing specific examples or evidence",
                      "Does not fully address what was asked"]
        feedback   = (f"This answer scored {score}/10. The response is too brief or lacks the depth expected "
                      f"for a {job_role} position. Add specific examples, correct terminology, and structure.")
    elif score <= 6:
        strengths  = random.sample(_STRENGTHS_POOL[:4], 2)
        weaknesses = random.sample(_WEAKNESSES_POOL[:5], 2)
        feedback   = (f"This answer scored {score}/10. A decent attempt but it needs more technical detail "
                      "and concrete examples. Using the STAR method and quantifying outcomes would improve it.")
    elif score <= 8:
        strengths  = random.sample(_STRENGTHS_POOL, 2)
        weaknesses = random.sample(_WEAKNESSES_POOL[2:], 2)
        feedback   = (f"Good answer scoring {score}/10. You demonstrated relevant knowledge for a {job_role} role. "
                      "Adding measurable outcomes and edge-case awareness would push this to an excellent response.")
    else:
        strengths  = random.sample(_STRENGTHS_POOL, 3)
        weaknesses = ["Consider adding even more specific metrics to further strengthen the answer"]
        feedback   = (f"Excellent answer scoring {score}/10. You clearly understand the topic at the {job_role} level "
                      "and communicated it effectively.")

    return {
        "score":     score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "feedback":  feedback,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_answer(question: str, answer: str, job_role: str, experience: str) -> dict:
    """
    Evaluate *answer* against *question* using IBM Granite.
    Falls back to heuristic scoring (not random high scores) when Granite is unavailable.

    Returns dict with keys: score (int 0-10), strengths (list), weaknesses (list), feedback (str)
    """
    prompt = f"""<|system|>
You are a strict, senior technical interviewer conducting a real job interview for a {job_role} position.
Your job is to evaluate the candidate's answer CRITICALLY and HONESTLY.

SCORING RULES — follow these exactly:
- Score 0-2: Answer is wrong, blank, "I don't know", or completely off-topic.
- Score 3-4: Answer is very vague, too short, or shows minimal understanding.
- Score 5-6: Partial answer — some correct points but missing key details, depth, or examples.
- Score 7-8: Good answer — correct, reasonably detailed, with an example or explanation.
- Score 9-10: Excellent — thorough, technically accurate, uses examples, metrics, and best practices.

Do NOT give a high score just because the candidate wrote something. Be strict.
Respond ONLY with a valid JSON object — no extra text, no markdown fences.
<|user|>
Role: {job_role}
Experience Level: {experience}

Interview Question:
{question}

Candidate's Answer:
{answer}

Evaluate strictly. If the answer is wrong or does not address the question, score it 0-3.
Return ONLY this JSON:
{{
  "score": <integer 0-10>,
  "strengths": ["<what they did well, or 'No notable strengths' if poor>"],
  "weaknesses": ["<specific thing that is wrong or missing>", "<another gap>"],
  "feedback": "<2-3 sentences: what score they got, why, and what the correct approach should be>"
}}
<|assistant|>
"""

    raw = _call_granite(prompt)

    if raw:
        parsed = _extract_json(raw)
        if parsed:
            try:
                score = max(0, min(10, int(parsed.get("score", 0))))
                strengths  = list(parsed.get("strengths",  [])) or ["Attempted to respond"]
                weaknesses = list(parsed.get("weaknesses", [])) or ["Answer needs significant improvement"]
                return {
                    "score":     score,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "feedback":  str(parsed.get("feedback", "")),
                }
            except (ValueError, TypeError):
                pass
        logger.warning("Could not parse Granite evaluation response; using heuristic scoring.")

    # Fallback uses answer content — never blindly high
    return _stub_evaluation(job_role, question, answer)


def generate_suggested_answer(question: str, job_role: str, experience: str) -> str:
    """
    Ask IBM Granite to produce a model answer for *question*.

    Returns a plain-text string.
    """
    prompt = f"""<|system|>
You are a top-performing {job_role} with {experience} of experience being interviewed.
Give a concise, high-quality model answer. Use the STAR method where appropriate.
Keep the answer to 4-6 sentences. Return plain text only — no bullet points, no JSON.
<|user|>
Interview Question: {question}
<|assistant|>
"""

    raw = _call_granite(prompt)

    if raw and len(raw) > 40:
        return raw

    # Fallback stub
    return (
        f"A strong answer for a {job_role} at {experience} level would clearly explain "
        "the core concept, provide a concrete example from past experience using the STAR "
        "(Situation, Task, Action, Result) method, and link the outcome to measurable business value. "
        "Quantify results where possible (e.g. 'reduced load time by 40%') and highlight "
        "collaboration or leadership aspects relevant to the role."
    )


def generate_questions(job_role: str, experience: str, interview_type: str, count: int = 7,
                        resume_text: str = "", industry: str = "") -> list[str]:
    """
    Ask IBM Granite to generate *count* tailored interview questions.

    When resume_text is provided the questions are grounded in the candidate's
    actual background and their target industry/role.

    Returns a list of question strings.
    Falls back to the static question bank (questions.py) on any failure.
    """
    resume_section = ""
    if resume_text:
        resume_section = f"\nCandidate Resume (use this to personalise every question):\n{resume_text[:3000]}\n"
    industry_section = f"\nTarget Industry / Business Domain: {industry}" if industry else ""

    prompt = f"""<|system|>
You are an expert interviewer creating a personalised interview question set.
Return ONLY a valid JSON array of exactly {count} question strings — no extra text.
<|user|>
Generate {count} interview questions for:
- Role: {job_role}
- Experience level: {experience}
- Interview type: {interview_type} (HR = behavioural/soft-skills, Technical = role-specific depth, Mixed = both){industry_section}{resume_section}
Rules:
- Make every question specific to this candidate's background, skills, and the stated role.
- Reference technologies, projects, or skills visible in the resume where possible.
- For Technical or Mixed, include scenario-based and system-design questions directly relevant to their stack.
- No repeated questions. No numbering. Return a JSON array only.

Example format:
["Question one?", "Question two?", "Question three?"]
<|assistant|>
"""

    raw = _call_granite(prompt)

    if raw:
        # Strip markdown fences
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
        # Find the first [...] array
        match = re.search(r"\[.*\]", raw_clean, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
                questions = [str(q).strip() for q in questions if str(q).strip()]
                if len(questions) >= count:
                    return questions[:count]
                if questions:
                    logger.warning(
                        "Granite returned %d questions (wanted %d); padding with static bank.",
                        len(questions), count
                    )
                    return questions  # caller (questions.py) will pad
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse Granite question list; falling back to static bank.")

    return []  # empty → caller falls back to static bank


# ── Resume Analysis ───────────────────────────────────────────────────────────

def analyze_resume(resume_text: str, job_role: str, experience: str) -> dict:
    """
    Analyze a resume against a target job role.
    Returns score, strengths, missing_skills, keywords, suggestions, summary.
    Falls back to a curated stub when Granite is unavailable.
    """
    prompt = f"""<|system|>
You are a professional resume reviewer and career coach. Analyze the resume for the target role and return ONLY valid JSON.
<|user|>
Target Role: {job_role}
Experience Level: {experience}

Resume Text:
{resume_text[:4000]}

Return ONLY this JSON object:
{{
  "score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "missing_skills": ["<missing skill 1>", "<missing skill 2>", "<missing skill 3>"],
  "keywords": ["<skill/keyword 1>", "<skill/keyword 2>", "<skill/keyword 3>", "<skill/keyword 4>", "<skill/keyword 5>"],
  "suggestions": ["<improvement 1>", "<improvement 2>", "<improvement 3>", "<improvement 4>"]
}}
<|assistant|>
"""
    raw = _call_granite(prompt)
    if raw:
        parsed = _extract_json(raw)
        if parsed:
            try:
                return {
                    "score":          max(0, min(100, int(parsed.get("score", 60)))),
                    "summary":        str(parsed.get("summary", "")),
                    "strengths":      list(parsed.get("strengths", [])),
                    "missing_skills": list(parsed.get("missing_skills", [])),
                    "keywords":       list(parsed.get("keywords", [])),
                    "suggestions":    list(parsed.get("suggestions", [])),
                }
            except (ValueError, TypeError):
                pass

    # Stub fallback
    role_lower = job_role.lower()
    skills_map = {
        "software engineer": ["Python", "Java", "REST APIs", "SQL", "Docker", "Git", "Agile"],
        "data scientist":    ["Python", "Machine Learning", "SQL", "TensorFlow", "Statistics", "Pandas"],
        "devops engineer":   ["Docker", "Kubernetes", "CI/CD", "Terraform", "AWS", "Linux"],
        "frontend developer":["React", "JavaScript", "CSS", "HTML", "TypeScript", "REST APIs"],
    }
    kw = skills_map.get(role_lower, ["Python", "Communication", "Problem Solving", "Git", "Agile"])
    return {
        "score":          random.randint(52, 74),
        "summary":        f"Your resume shows relevant experience for a {job_role} role. Some key skills are present but there are areas to strengthen for higher competitiveness.",
        "strengths":      ["Relevant educational background", "Project experience demonstrated", "Technical skills listed"],
        "missing_skills": [f"Quantified impact metrics", f"Specific {job_role} certifications", "Contribution to open source or public projects"],
        "keywords":       kw,
        "suggestions":    [
            "Add measurable outcomes to each experience bullet (e.g. 'reduced load time by 30%').",
            f"Include certifications relevant to {job_role} (IBM, AWS, Google Cloud).",
            "Add a concise professional summary at the top of the resume.",
            "Tailor the resume keywords to match the specific job description.",
        ],
    }


# ── Skill Analysis ────────────────────────────────────────────────────────────

def analyze_skills(skills: list, target_role: str, experience: str) -> dict:
    """
    Analyze a skill set for competitiveness in the job market.
    Returns score, strong_skills, weak_skills, trending_skills, learning_suggestions, free_courses.
    """
    skills_str = ", ".join(skills)
    prompt = f"""<|system|>
You are a senior technical recruiter and career coach. Evaluate the skill set for the target role and return ONLY valid JSON.
<|user|>
Skills: {skills_str}
Target Role: {target_role if target_role else "General Tech"}
Experience Level: {experience}

Return ONLY this JSON:
{{
  "competitiveness_score": <integer 0-100>,
  "summary": "<2-3 sentences on market competitiveness>",
  "strong_skills": ["<skill>", "<skill>"],
  "weak_skills": ["<skill to improve or add>", "<skill to improve or add>"],
  "trending_skills": ["<trending skill 1>", "<trending skill 2>", "<trending skill 3>"],
  "learning_suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"],
  "free_courses": [
    {{"title": "<course name>", "platform": "<platform>", "url": "<url if known>", "reason": "<why useful>"}},
    {{"title": "<course name>", "platform": "<platform>", "url": "<url if known>", "reason": "<why useful>"}}
  ]
}}
<|assistant|>
"""
    raw = _call_granite(prompt)
    if raw:
        parsed = _extract_json(raw)
        if parsed:
            try:
                return {
                    "competitiveness_score": max(0, min(100, int(parsed.get("competitiveness_score", 60)))),
                    "summary":               str(parsed.get("summary", "")),
                    "strong_skills":         list(parsed.get("strong_skills", [])),
                    "weak_skills":           list(parsed.get("weak_skills", [])),
                    "trending_skills":       list(parsed.get("trending_skills", [])),
                    "learning_suggestions":  list(parsed.get("learning_suggestions", [])),
                    "free_courses":          list(parsed.get("free_courses", [])),
                }
            except (ValueError, TypeError):
                pass

    # Stub fallback
    trending = ["LLM / GenAI Integration", "Kubernetes", "TypeScript", "Rust", "MLOps", "Terraform"]
    return {
        "competitiveness_score": random.randint(55, 75),
        "summary":               f"Your skill set shows a solid foundation for {target_role or 'tech roles'}. Adding trending technologies and certifications will significantly boost your marketability.",
        "strong_skills":         skills[:3] if skills else ["Communication", "Problem Solving"],
        "weak_skills":           ["Cloud certifications (AWS/GCP/Azure)", "System design knowledge", "LLM/AI integration skills"],
        "trending_skills":       random.sample(trending, 3),
        "learning_suggestions":  [
            "Complete a cloud platform certification (AWS Solutions Architect or Google Cloud).",
            "Build 1-2 portfolio projects that use your listed skills end-to-end.",
            "Learn Docker + Kubernetes basics — these appear in 70%+ of job descriptions.",
        ],
        "free_courses": [
            {"title": "CS50: Introduction to Computer Science", "platform": "edX (Harvard)", "url": "https://cs50.harvard.edu/x/", "reason": "Best free CS foundation course globally."},
            {"title": "Machine Learning Specialization", "platform": "Coursera (DeepLearning.AI)", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "reason": "Industry-standard ML curriculum, audit for free."},
            {"title": "IBM Full Stack Cloud Developer", "platform": "Coursera (IBM)", "url": "https://www.coursera.org/professional-certificates/ibm-full-stack-cloud-developer", "reason": "Covers cloud + full stack with IBM Granite AI integration."},
        ],
    }


# ── Roadmap Generation ────────────────────────────────────────────────────────

def generate_roadmap(domain: str, level: str, goal: str) -> dict:
    """
    Generate a personalised learning roadmap for a domain/role.
    Returns overview, phases, weekly_plan, books, videos, tips.
    """
    prompt = f"""<|system|>
You are a senior career mentor and technical educator. Generate a detailed personalised learning roadmap and return ONLY valid JSON.
<|user|>
Domain / Target Role: {domain}
Current Level: {level}
Goal: {goal if goal else "Land a job in this domain"}

Return ONLY this JSON:
{{
  "overview": "<2-3 sentence overview of the roadmap>",
  "phases": [
    {{"title": "<phase title>", "description": "<what to focus on>", "topics": ["<topic>", "<topic>", "<topic>"], "duration": "<e.g. 2-3 weeks>"}},
    {{"title": "<phase title>", "description": "<what to focus on>", "topics": ["<topic>", "<topic>", "<topic>"], "duration": "<e.g. 4 weeks>"}},
    {{"title": "<phase title>", "description": "<what to focus on>", "topics": ["<topic>", "<topic>", "<topic>"], "duration": "<e.g. 4-6 weeks>"}},
    {{"title": "<phase title>", "description": "<what to focus on>", "topics": ["<topic>", "<topic>", "<topic>"], "duration": "<e.g. 3-4 weeks>"}}
  ],
  "weekly_plan": [
    {{"week": "Week 1-2", "focus": "<specific focus>"}},
    {{"week": "Week 3-4", "focus": "<specific focus>"}},
    {{"week": "Week 5-6", "focus": "<specific focus>"}},
    {{"week": "Week 7-8", "focus": "<specific focus>"}}
  ],
  "books": [
    {{"title": "<book title>", "author": "<author>", "reason": "<why recommended>"}},
    {{"title": "<book title>", "author": "<author>", "reason": "<why recommended>"}},
    {{"title": "<book title>", "author": "<author>", "reason": "<why recommended>"}}
  ],
  "videos": [
    {{"title": "<channel or playlist name>", "channel": "<YouTube channel>", "url": "https://www.youtube.com/...", "reason": "<why recommended>"}},
    {{"title": "<channel or playlist name>", "channel": "<YouTube channel>", "url": "https://www.youtube.com/...", "reason": "<why recommended>"}},
    {{"title": "<channel or playlist name>", "channel": "<YouTube channel>", "url": "https://www.youtube.com/...", "reason": "<why recommended>"}}
  ],
  "tips": ["<pro tip 1>", "<pro tip 2>", "<pro tip 3>"]
}}
<|assistant|>
"""
    raw = _call_granite(prompt)
    if raw:
        parsed = _extract_json(raw)
        if parsed and parsed.get("phases"):
            try:
                return {
                    "overview":    str(parsed.get("overview", "")),
                    "phases":      list(parsed.get("phases", [])),
                    "weekly_plan": list(parsed.get("weekly_plan", [])),
                    "books":       list(parsed.get("books", [])),
                    "videos":      list(parsed.get("videos", [])),
                    "tips":        list(parsed.get("tips", [])),
                }
            except (ValueError, TypeError):
                pass

    # Domain-aware stub fallbacks
    domain_l = domain.lower()

    if "data" in domain_l or "ml" in domain_l or "machine learning" in domain_l:
        return _roadmap_stub_data_science(domain, level)
    elif "devops" in domain_l or "sre" in domain_l:
        return _roadmap_stub_devops(domain, level)
    elif "cloud" in domain_l or "aws" in domain_l or "azure" in domain_l or "gcp" in domain_l:
        return _roadmap_stub_cloud(domain, level)
    elif "frontend" in domain_l or "react" in domain_l or "ui" in domain_l or "ux" in domain_l:
        return _roadmap_stub_frontend(domain, level)
    elif "security" in domain_l or "cyber" in domain_l:
        return _roadmap_stub_cyber(domain, level)
    elif "mobile" in domain_l or "android" in domain_l or "ios" in domain_l or "flutter" in domain_l:
        return _roadmap_stub_mobile(domain, level)
    elif "product" in domain_l or "pm" in domain_l:
        return _roadmap_stub_product_manager(domain, level)
    elif "business analyst" in domain_l or "business analysis" in domain_l or "ba " in domain_l:
        return _roadmap_stub_business_analyst(domain, level)
    elif "full stack" in domain_l or "fullstack" in domain_l:
        return _roadmap_stub_fullstack(domain, level)
    elif "backend" in domain_l:
        return _roadmap_stub_backend(domain, level)
    else:
        return _roadmap_stub_software(domain, level)


def _roadmap_stub_software(domain, level):
    return {
        "overview": f"This roadmap guides you from {level} to job-ready {domain}. Focus on fundamentals, build projects, and practice interviews consistently.",
        "phases": [
            {"title": "Foundations", "description": "Master the core programming and CS fundamentals.", "topics": ["Data Structures & Algorithms", "OOP Concepts", "Version Control (Git)", "Linux Basics"], "duration": "3-4 weeks"},
            {"title": "Core Skills", "description": "Learn the primary tech stack for your target role.", "topics": ["Backend/Frontend frameworks", "Databases (SQL + NoSQL)", "REST APIs", "Testing"], "duration": "4-6 weeks"},
            {"title": "Projects & Portfolio", "description": "Build 2-3 real-world projects to showcase on GitHub and resume.", "topics": ["End-to-end project", "Cloud deployment", "CI/CD basics", "Documentation"], "duration": "4-5 weeks"},
            {"title": "Interview Prep", "description": "Practice mock interviews, LeetCode, and system design.", "topics": ["LeetCode (Easy → Medium)", "System Design basics", "Behavioural questions", "Resume polish"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Data Structures — Arrays, Linked Lists, Stacks, Queues"},
            {"week": "Week 3-4", "focus": "Core language deep-dive + Git & GitHub"},
            {"week": "Week 5-6", "focus": "Build your first full project (CRUD app)"},
            {"week": "Week 7-8", "focus": "LeetCode practice + mock interviews"},
        ],
        "books": [
            {"title": "Clean Code", "author": "Robert C. Martin", "reason": "Essential for writing professional, maintainable code."},
            {"title": "Cracking the Coding Interview", "author": "Gayle Laakmann McDowell", "reason": "The gold standard for technical interview preparation."},
            {"title": "The Pragmatic Programmer", "author": "David Thomas & Andrew Hunt", "reason": "Mindset and practices every developer must internalize."},
        ],
        "videos": [
            {"title": "CS50 Full Course", "channel": "CS50 (Harvard)", "url": "https://www.youtube.com/@cs50", "reason": "Best free CS foundation on the internet."},
            {"title": "NeetCode 150 LeetCode Solutions", "channel": "NeetCode", "url": "https://www.youtube.com/@NeetCode", "reason": "Clear DSA explanations with optimal solutions."},
            {"title": "Fireship — Web Dev & CS Tutorials", "channel": "Fireship", "url": "https://www.youtube.com/@Fireship", "reason": "Fast, accurate overviews of tech topics."},
        ],
        "tips": [
            "Code every day — even 30 minutes of consistent practice beats 4-hour weekend sessions.",
            "Build projects that solve real problems; they make for far better interview stories.",
            "Practice explaining your code out loud — interviewers value communication as much as correctness.",
        ],
    }


def _roadmap_stub_data_science(domain, level):
    return {
        "overview": f"This roadmap takes you from {level} to industry-ready {domain}. Statistics, Python, and ML fundamentals are your core pillars.",
        "phases": [
            {"title": "Math & Python Foundations", "description": "Statistics, probability, linear algebra, and Python for data.", "topics": ["Python (NumPy, Pandas)", "Descriptive Statistics", "Probability & Distributions", "Linear Algebra basics"], "duration": "3-4 weeks"},
            {"title": "Machine Learning Core", "description": "Supervised and unsupervised learning algorithms.", "topics": ["Regression, Classification", "Decision Trees, Random Forests", "Clustering (K-Means)", "Model Evaluation & Cross-Validation"], "duration": "4-5 weeks"},
            {"title": "Deep Learning & Tools", "description": "Neural networks, frameworks, and data engineering.", "topics": ["TensorFlow / PyTorch basics", "CNNs and RNNs", "Feature Engineering", "SQL + BigQuery"], "duration": "4-5 weeks"},
            {"title": "Portfolio & Interviews", "description": "Kaggle competitions, end-to-end projects, and interview prep.", "topics": ["Kaggle projects", "Model deployment (Flask/FastAPI)", "ML system design", "Case study interviews"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Python for Data Science — NumPy, Pandas, Matplotlib"},
            {"week": "Week 3-4", "focus": "Statistics fundamentals + Probability"},
            {"week": "Week 5-6", "focus": "Scikit-learn — build first ML models"},
            {"week": "Week 7-8", "focus": "Deep Learning intro + first Kaggle competition"},
        ],
        "books": [
            {"title": "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", "author": "Aurélien Géron", "reason": "The most comprehensive practical ML book available."},
            {"title": "Python for Data Analysis", "author": "Wes McKinney", "reason": "Written by the creator of Pandas — definitive guide."},
            {"title": "The Elements of Statistical Learning", "author": "Hastie, Tibshirani, Friedman", "reason": "Deep theoretical foundation — free PDF available."},
        ],
        "videos": [
            {"title": "Machine Learning Specialization", "channel": "DeepLearning.AI / Andrew Ng", "url": "https://www.youtube.com/@Deeplearningai", "reason": "Best structured ML course by a pioneer in the field."},
            {"title": "StatQuest with Josh Starmer", "channel": "StatQuest", "url": "https://www.youtube.com/@statquest", "reason": "Statistics and ML explained clearly with great visuals."},
            {"title": "Sentdex Python Data Science", "channel": "sentdex", "url": "https://www.youtube.com/@sentdex", "reason": "Practical Python + ML tutorials for beginners."},
        ],
        "tips": [
            "Enter at least one Kaggle competition — it forces you to apply theory to real messy data.",
            "Document every project with a clear README and results — recruiters read these.",
            "Learn SQL deeply — most data science interviews include SQL rounds.",
        ],
    }


def _roadmap_stub_devops(domain, level):
    return {
        "overview": f"This roadmap builds your {domain} skills from {level} up. Linux, Docker, Kubernetes, and CI/CD are the four pillars you must master.",
        "phases": [
            {"title": "Linux & Scripting", "description": "Linux command line, shell scripting, and networking basics.", "topics": ["Linux commands & file system", "Bash scripting", "Networking (TCP/IP, DNS, HTTP)", "SSH & security basics"], "duration": "2-3 weeks"},
            {"title": "Containers & IaC", "description": "Docker, Kubernetes, and Infrastructure as Code tools.", "topics": ["Docker & Docker Compose", "Kubernetes (pods, services, deployments)", "Terraform basics", "Ansible"], "duration": "4-5 weeks"},
            {"title": "CI/CD & Cloud", "description": "Pipeline automation, cloud platforms, and monitoring.", "topics": ["GitHub Actions / Jenkins", "AWS / GCP / Azure basics", "Prometheus & Grafana", "Logging (ELK stack)"], "duration": "4-5 weeks"},
            {"title": "Certifications & Interviews", "description": "Certify your skills and practice DevOps interview questions.", "topics": ["AWS Solutions Architect (Associate)", "CKA (Kubernetes)", "Incident response scenarios", "System design for DevOps"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Linux fundamentals + Bash scripting"},
            {"week": "Week 3-4", "focus": "Docker — build, run, compose multi-container apps"},
            {"week": "Week 5-6", "focus": "Kubernetes — deploy your first app to a cluster"},
            {"week": "Week 7-8", "focus": "CI/CD pipeline + AWS fundamentals"},
        ],
        "books": [
            {"title": "The Phoenix Project", "author": "Gene Kim et al.", "reason": "Essential DevOps culture and principles — reads like a novel."},
            {"title": "Kubernetes in Action", "author": "Marko Lukša", "reason": "Most thorough Kubernetes book available."},
            {"title": "Terraform: Up & Running", "author": "Yevgeniy Brikman", "reason": "Practical IaC with real-world examples."},
        ],
        "videos": [
            {"title": "TechWorld with Nana", "channel": "TechWorld with Nana", "url": "https://www.youtube.com/@TechWorldwithNana", "reason": "Best DevOps tutorials on YouTube — Docker, K8s, CI/CD."},
            {"title": "FreeCodeCamp DevOps Course", "channel": "freeCodeCamp.org", "url": "https://www.youtube.com/@freecodecamp", "reason": "Full free DevOps curriculum in one channel."},
            {"title": "NetworkChuck", "channel": "NetworkChuck", "url": "https://www.youtube.com/@NetworkChuck", "reason": "Linux and networking fundamentals made fun."},
        ],
        "tips": [
            "Set up a local Kubernetes cluster with minikube — hands-on practice beats watching tutorials.",
            "Get your first cloud certification (AWS Cloud Practitioner) before the role-specific ones.",
            "Automate something in your daily life — even a simple cron job shows initiative.",
        ],
    }


def _roadmap_stub_frontend(domain, level):
    return {
        "overview": f"This roadmap guides you through modern {domain} development from {level}. HTML/CSS/JS fundamentals lead to React, then real projects and performance optimization.",
        "phases": [
            {"title": "HTML, CSS & JavaScript", "description": "The three pillars of the web — master these before any framework.", "topics": ["Semantic HTML5", "CSS Flexbox & Grid", "JavaScript ES6+", "DOM Manipulation"], "duration": "3-4 weeks"},
            {"title": "React & Ecosystem", "description": "Build reactive UIs with the most popular frontend framework.", "topics": ["React (hooks, state, props)", "React Router", "State management (Zustand/Redux)", "TypeScript basics"], "duration": "4-5 weeks"},
            {"title": "Performance & Tools", "description": "Optimization, testing, and build tooling.", "topics": ["Vite / Webpack", "Unit testing (Jest, Vitest)", "Accessibility (WCAG)", "Web performance (Core Web Vitals)"], "duration": "3-4 weeks"},
            {"title": "Portfolio & Job Prep", "description": "Build production-quality projects and prepare for interviews.", "topics": ["Full portfolio app (React + API)", "Deployment (Vercel/Netlify)", "CSS animations", "System design for frontend"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "HTML5 + CSS3 — build responsive layouts from scratch"},
            {"week": "Week 3-4", "focus": "JavaScript fundamentals + async/await + fetch API"},
            {"week": "Week 5-6", "focus": "React core — components, hooks, state management"},
            {"week": "Week 7-8", "focus": "Build and deploy a full React project"},
        ],
        "books": [
            {"title": "JavaScript: The Good Parts", "author": "Douglas Crockford", "reason": "Concise guide to the best features of JavaScript."},
            {"title": "Learning React", "author": "Alex Banks & Eve Porcello", "reason": "Best modern React book for beginners through intermediate."},
            {"title": "CSS: The Definitive Guide", "author": "Eric A. Meyer", "reason": "The most complete CSS reference — invaluable for layout mastery."},
        ],
        "videos": [
            {"title": "The Odin Project", "channel": "The Odin Project", "url": "https://www.youtube.com/@TheOdinProject", "reason": "Free full-stack curriculum — one of the best structured paths."},
            {"title": "Traversy Media", "channel": "Traversy Media", "url": "https://www.youtube.com/@TraversyMedia", "reason": "High-quality practical tutorials across the entire frontend stack."},
            {"title": "Kevin Powell — CSS", "channel": "Kevin Powell", "url": "https://www.youtube.com/@KevinPowell", "reason": "The best CSS-focused YouTube channel on the internet."},
        ],
        "tips": [
            "Build in public — share your projects on LinkedIn and GitHub; it creates real job opportunities.",
            "Recreate popular UIs (Spotify, Airbnb, Twitter) to push your CSS skills.",
            "Learn the browser DevTools deeply — it's your most powerful debugging tool.",
        ],
    }


def _roadmap_stub_cyber(domain, level):
    return {
        "overview": f"This roadmap prepares you for {domain} from {level}. Networking, OS fundamentals, and ethical hacking are the foundations — certifications validate your skills to employers.",
        "phases": [
            {"title": "Networking & OS Fundamentals", "description": "TCP/IP, OSI model, Linux, and Windows administration.", "topics": ["CompTIA Network+ concepts", "Linux administration", "Windows Active Directory", "Firewalls & proxies"], "duration": "3-4 weeks"},
            {"title": "Security Core Concepts", "description": "Cryptography, identity management, and threat analysis.", "topics": ["Encryption & PKI", "IAM & Zero Trust", "OWASP Top 10", "Vulnerability scanning (Nmap, Nessus)"], "duration": "3-4 weeks"},
            {"title": "Ethical Hacking & DFIR", "description": "Penetration testing methodology and digital forensics.", "topics": ["Kali Linux & Metasploit", "Web app pentesting (BurpSuite)", "Incident Response process", "SIEM & log analysis"], "duration": "4-5 weeks"},
            {"title": "Certifications & CTF", "description": "CompTIA Security+, CEH, or OSCP preparation.", "topics": ["CompTIA Security+ exam prep", "TryHackMe / HackTheBox CTFs", "Writing pentest reports", "SOC analyst scenarios"], "duration": "4+ weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Networking fundamentals — OSI model, TCP/IP, subnetting"},
            {"week": "Week 3-4", "focus": "Linux + security tools (Nmap, Wireshark)"},
            {"week": "Week 5-6", "focus": "OWASP Top 10 + web app pentesting basics"},
            {"week": "Week 7-8", "focus": "TryHackMe rooms + CompTIA Security+ study"},
        ],
        "books": [
            {"title": "The Web Application Hacker's Handbook", "author": "Stuttard & Pinto", "reason": "The definitive guide to web application security testing."},
            {"title": "CompTIA Security+ Study Guide", "author": "Mike Chapple & David Seidl", "reason": "Best prep book for the most recognized entry-level cert."},
            {"title": "Hacking: The Art of Exploitation", "author": "Jon Erickson", "reason": "Deep technical hacking knowledge with C programming context."},
        ],
        "videos": [
            {"title": "Professor Messer Security+", "channel": "Professor Messer", "url": "https://www.youtube.com/@professormesser", "reason": "Free and comprehensive CompTIA Security+ course."},
            {"title": "NetworkChuck Cybersecurity", "channel": "NetworkChuck", "url": "https://www.youtube.com/@NetworkChuck", "reason": "Engaging hacking and networking tutorials for beginners."},
            {"title": "John Hammond CTF Walkthroughs", "channel": "John Hammond", "url": "https://www.youtube.com/@_JohnHammond", "reason": "Real CTF solutions — teaches attacker thinking practically."},
        ],
        "tips": [
            "Set up a home lab with VirtualBox — practice on VMs, never on real systems without permission.",
            "Start with TryHackMe beginner paths before moving to HackTheBox.",
            "Get CompTIA Security+ first — it's recognized by most employers and DoD-approved.",
        ],
    }


def _roadmap_stub_cloud(domain, level):
    return {
        "overview": f"This roadmap takes you from {level} to a job-ready {domain}. Cloud fundamentals, core AWS/Azure/GCP services, and IaC are the pillars — certifications open doors fast.",
        "phases": [
            {"title": "Cloud & Networking Fundamentals", "description": "Understand virtualisation, networking, and the major cloud providers.", "topics": ["Cloud computing concepts (IaaS, PaaS, SaaS)", "Virtual networks & subnets", "IAM & identity basics", "Storage, compute, and database services"], "duration": "2-3 weeks"},
            {"title": "Core Cloud Services", "description": "Hands-on with the major platform services.", "topics": ["EC2 / VMs / GCE", "S3 / Blob Storage / GCS", "Lambda / Functions / Cloud Run", "Managed databases (RDS, Cosmos, Cloud SQL)"], "duration": "3-4 weeks"},
            {"title": "Architecture & IaC", "description": "Design scalable, resilient cloud architectures with automation.", "topics": ["Terraform / CloudFormation / Bicep", "Auto-scaling & load balancing", "High availability & disaster recovery", "Cost optimisation strategies"], "duration": "4-5 weeks"},
            {"title": "Certifications & Projects", "description": "Validate skills and build portfolio projects.", "topics": ["AWS Solutions Architect Associate / AZ-900 / GCP ACE", "Deploy a 3-tier app to the cloud", "Monitoring with CloudWatch / Azure Monitor", "CI/CD integration"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Cloud fundamentals — regions, availability zones, IAM, billing"},
            {"week": "Week 3-4", "focus": "Core services: compute, storage, networking — hands-on in console"},
            {"week": "Week 5-6", "focus": "Terraform basics — provision and destroy infrastructure as code"},
            {"week": "Week 7-8", "focus": "Deploy a real project + begin certification practice tests"},
        ],
        "books": [
            {"title": "AWS Certified Solutions Architect Study Guide", "author": "Ben Piper & David Clinton", "reason": "The most comprehensive AWS certification prep book."},
            {"title": "Cloud Native Patterns", "author": "Cornelia Davis", "reason": "Practical patterns for building resilient cloud-native applications."},
            {"title": "Terraform: Up & Running", "author": "Yevgeniy Brikman", "reason": "Best practical guide to Infrastructure as Code with Terraform."},
        ],
        "videos": [
            {"title": "FreeCodeCamp AWS Full Course", "channel": "freeCodeCamp.org", "url": "https://www.youtube.com/@freecodecamp", "reason": "Complete free AWS course covering all core services."},
            {"title": "TechWorld with Nana — Cloud & DevOps", "channel": "TechWorld with Nana", "url": "https://www.youtube.com/@TechWorldwithNana", "reason": "High quality, project-based cloud and DevOps tutorials."},
            {"title": "A Cloud Guru / Pluralsight", "channel": "A Cloud Guru", "url": "https://www.youtube.com/@ACloudGuru", "reason": "Industry-standard cloud certification prep content."},
        ],
        "tips": [
            "Use the free tier of AWS/Azure/GCP to practice — always set billing alerts to avoid surprises.",
            "Get cloud certified early; even the foundational certs (AZ-900, AWS Cloud Practitioner) open doors.",
            "Build a real project deployed to the cloud — deploy a backend API with a database and CI/CD pipeline.",
        ],
    }


def _roadmap_stub_mobile(domain, level):
    return {
        "overview": f"This roadmap guides you from {level} to building production-ready {domain} apps. Choose React Native or Flutter for cross-platform, or Swift/Kotlin for native-first development.",
        "phases": [
            {"title": "Programming & UI Basics", "description": "Core language and UI fundamentals for mobile.", "topics": ["Dart (Flutter) or JavaScript/TypeScript (React Native)", "Widgets / Components & layouts", "State management basics", "Navigation patterns"], "duration": "3-4 weeks"},
            {"title": "APIs & Data", "description": "Integrate backends, manage local data, and handle async operations.", "topics": ["REST API integration", "Local storage (SQLite, Hive, AsyncStorage)", "Authentication (JWT, OAuth, Firebase Auth)", "Push notifications"], "duration": "3-4 weeks"},
            {"title": "Advanced Features & Performance", "description": "Native device features, animations, and optimisation.", "topics": ["Camera, GPS, sensors", "Animations & gestures", "Performance profiling", "Background tasks"], "duration": "3-4 weeks"},
            {"title": "Publishing & Portfolio", "description": "App store deployment and interview preparation.", "topics": ["App Store & Google Play publishing", "CI/CD for mobile (Fastlane / GitHub Actions)", "End-to-end project", "Mobile system design questions"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Core language + first app — counter, to-do list, basic UI"},
            {"week": "Week 3-4", "focus": "API integration — fetch data from a public API and display it"},
            {"week": "Week 5-6", "focus": "Auth + local storage — build a login flow with persisted state"},
            {"week": "Week 7-8", "focus": "Polish & publish — deploy your app to TestFlight / Play Console"},
        ],
        "books": [
            {"title": "Flutter in Action", "author": "Eric Windmill", "reason": "Best comprehensive Flutter book for beginners through intermediate."},
            {"title": "Learning React Native", "author": "Bonnie Eisenman", "reason": "Practical guide to cross-platform mobile with React Native."},
            {"title": "Swift Programming: The Big Nerd Ranch Guide", "author": "Mikey Ward & Matthew Mathias", "reason": "Gold standard for iOS-native Swift development."},
        ],
        "videos": [
            {"title": "Flutter Tutorial for Beginners", "channel": "The Net Ninja", "url": "https://www.youtube.com/@NetNinja", "reason": "Best beginner-friendly Flutter course on YouTube."},
            {"title": "Academind React Native", "channel": "Academind", "url": "https://www.youtube.com/@academind", "reason": "Comprehensive and well-paced React Native tutorials."},
            {"title": "CodeWithChris iOS Development", "channel": "CodeWithChris", "url": "https://www.youtube.com/@CodeWithChris", "reason": "Approachable Swift and SwiftUI tutorials for iOS beginners."},
        ],
        "tips": [
            "Pick ONE framework first (Flutter is the most in-demand cross-platform choice in 2024) and build depth.",
            "Build at least one app and publish it — even a simple utility app on the store impresses recruiters.",
            "Learn the platform's HIG (Human Interface Guidelines / Material Design) — UI patterns matter.",
        ],
    }


def _roadmap_stub_product_manager(domain, level):
    return {
        "overview": f"This roadmap develops your {domain} skills from {level}. Mastering user research, data-driven decision making, and stakeholder communication are the core pillars.",
        "phases": [
            {"title": "PM Foundations", "description": "Understand the product lifecycle, core frameworks, and the PM role.", "topics": ["Product lifecycle & discovery", "User story mapping", "Agile & Scrum for PMs", "Stakeholder management basics"], "duration": "2-3 weeks"},
            {"title": "Research & Analytics", "description": "Learn user research methods and data analysis.", "topics": ["User interviews & surveys", "Google Analytics / Mixpanel", "A/B testing fundamentals", "KPIs, OKRs, and North Star metrics"], "duration": "3-4 weeks"},
            {"title": "Roadmap & Prioritisation", "description": "Frameworks for building and defending a product roadmap.", "topics": ["RICE / MoSCoW / Kano model", "PRD (Product Requirements Doc) writing", "Technical debt trade-offs", "Competitor analysis"], "duration": "3-4 weeks"},
            {"title": "Portfolio & Interviews", "description": "Build your PM portfolio and ace product interviews.", "topics": ["Product critique exercises", "Estimation & metrics questions", "Case study preparation", "PM interview formats (Google, Meta, Amazon)"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "PM fundamentals — read 'Inspired', complete a PM case study"},
            {"week": "Week 3-4", "focus": "User research — conduct 5 user interviews, write a PRD"},
            {"week": "Week 5-6", "focus": "Analytics — set up a metrics dashboard for a real or mock product"},
            {"week": "Week 7-8", "focus": "Mock PM interviews — practice product design & estimation questions"},
        ],
        "books": [
            {"title": "Inspired: How to Create Tech Products Customers Love", "author": "Marty Cagan", "reason": "The definitive book on modern product management — required reading."},
            {"title": "Continuous Discovery Habits", "author": "Teresa Torres", "reason": "Practical framework for ongoing user research and opportunity mapping."},
            {"title": "Cracking the PM Interview", "author": "Gayle Laakmann McDowell", "reason": "Best structured prep for PM interviews at top tech companies."},
        ],
        "videos": [
            {"title": "Lenny's Podcast — Product Management", "channel": "Lenny's Podcast", "url": "https://www.youtube.com/@LennysPodcast", "reason": "Deep-dive PM conversations with leaders from Airbnb, Google, Slack."},
            {"title": "Product School", "channel": "Product School", "url": "https://www.youtube.com/@ProductSchool", "reason": "Free webinars and talks from PMs at leading tech companies."},
            {"title": "My PM Interview", "channel": "My PM Interview", "url": "https://www.youtube.com/@mypmlibrary", "reason": "Structured practice for product design, metrics, and estimation questions."},
        ],
        "tips": [
            "Build a side project or improve an existing app — nothing is more impressive than shipping something real.",
            "Practise the CIRCLES framework for product design questions; structure your answers in every mock interview.",
            "Talk to real users — even 5 interviews will give you more insight than 10 hours of reading.",
        ],
    }


def _roadmap_stub_business_analyst(domain, level):
    return {
        "overview": f"This roadmap builds your {domain} skills from {level}. Requirements gathering, data analysis, and stakeholder communication are the core competencies every BA must master.",
        "phases": [
            {"title": "BA Foundations", "description": "Core BA concepts, methodologies, and business process modelling.", "topics": ["Business Analysis Body of Knowledge (BABOK)", "Agile vs Waterfall for BAs", "Process modelling (BPMN, flowcharts)", "Stakeholder identification & management"], "duration": "2-3 weeks"},
            {"title": "Requirements Engineering", "description": "Elicit, document, and validate requirements effectively.", "topics": ["User stories & acceptance criteria", "Use case diagrams & sequence diagrams", "Gap analysis & root cause analysis", "Functional vs non-functional requirements"], "duration": "3-4 weeks"},
            {"title": "Data & Analytics", "description": "SQL, Excel, and BI tools every BA needs.", "topics": ["SQL for data querying", "Excel / Google Sheets — pivot tables, VLOOKUP", "Power BI or Tableau basics", "KPI definition & dashboarding"], "duration": "3-4 weeks"},
            {"title": "Certifications & Portfolio", "description": "ECBA/CBAP prep, case studies, and interview preparation.", "topics": ["ECBA (Entry Certificate in BA) exam prep", "Building a requirements artefacts portfolio", "Business case writing", "BA case interview practice"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "BABOK basics + draw a BPMN process diagram for a real workflow"},
            {"week": "Week 3-4", "focus": "Requirements elicitation — write 10 user stories with acceptance criteria"},
            {"week": "Week 5-6", "focus": "SQL practice — write queries on a sample database (Mode Analytics)"},
            {"week": "Week 7-8", "focus": "Build a Power BI / Tableau dashboard + mock BA interview practice"},
        ],
        "books": [
            {"title": "A Guide to the Business Analysis Body of Knowledge (BABOK)", "author": "IIBA", "reason": "The official knowledge base for professional business analysts."},
            {"title": "Business Analysis for Practitioners", "author": "PMI", "reason": "Practical guide to eliciting and managing requirements."},
            {"title": "The Business Analyst's Handbook", "author": "Howard Podeswa", "reason": "Comprehensive, practical reference for day-to-day BA work."},
        ],
        "videos": [
            {"title": "Business Analysis Excellence", "channel": "Business Analysis Excellence", "url": "https://www.youtube.com/@businessanalysisexcellence", "reason": "Practical BA tutorials covering requirements, use cases, and interviews."},
            {"title": "365 Data Science — SQL for Beginners", "channel": "365 Data Science", "url": "https://www.youtube.com/@365DataScience", "reason": "Clear, free SQL tutorials essential for any BA role."},
            {"title": "Guy in a Cube — Power BI", "channel": "Guy in a Cube", "url": "https://www.youtube.com/@GuyInACube", "reason": "Best Power BI tutorials on YouTube — great for BI-focused BA roles."},
        ],
        "tips": [
            "Always tie requirements back to business value — ask 'why?' at least three times to get to the root need.",
            "Learn SQL to an intermediate level — it's the most transferable skill a BA can have.",
            "Build a portfolio of artefacts (BRDs, user stories, process diagrams) even from practice projects.",
        ],
    }


def _roadmap_stub_fullstack(domain, level):
    return {
        "overview": f"This roadmap builds your {domain} skills from {level}. You'll master the frontend, backend, databases, and deployment — the complete stack for building real-world web applications.",
        "phases": [
            {"title": "Frontend Foundations", "description": "HTML, CSS, JavaScript, and a modern frontend framework.", "topics": ["HTML5 + CSS3 (Flexbox, Grid)", "JavaScript ES6+ & DOM", "React or Vue basics", "Responsive design & accessibility"], "duration": "3-4 weeks"},
            {"title": "Backend & APIs", "description": "Server-side programming, REST APIs, and authentication.", "topics": ["Node.js + Express or Python + FastAPI/Django", "REST API design", "JWT authentication", "Error handling & middleware"], "duration": "4-5 weeks"},
            {"title": "Databases & Deployment", "description": "SQL/NoSQL databases and cloud deployment.", "topics": ["PostgreSQL or MySQL (SQL)", "MongoDB (NoSQL)", "Docker basics", "Deploy to Vercel / Railway / AWS"], "duration": "3-4 weeks"},
            {"title": "Portfolio & Interviews", "description": "Build end-to-end projects and prepare for technical interviews.", "topics": ["Full-stack project (e.g. SaaS clone)", "CI/CD with GitHub Actions", "System design for web apps", "LeetCode medium problems"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "HTML + CSS + JavaScript fundamentals — build 2 static pages"},
            {"week": "Week 3-4", "focus": "React basics + fetch from a public API"},
            {"week": "Week 5-6", "focus": "Build a REST API with Node.js + Express + PostgreSQL"},
            {"week": "Week 7-8", "focus": "Full-stack project — combine frontend + backend + deploy it"},
        ],
        "books": [
            {"title": "Full Stack Open (free online course)", "author": "University of Helsinki", "reason": "The best free full-stack curriculum — React, Node, GraphQL, TypeScript."},
            {"title": "Node.js Design Patterns", "author": "Mario Casciaro & Luciano Mammino", "reason": "Deep dive into scalable Node.js application architecture."},
            {"title": "Learning SQL", "author": "Alan Beaulieu", "reason": "Clear, practical introduction to SQL for application developers."},
        ],
        "videos": [
            {"title": "The Odin Project", "channel": "The Odin Project", "url": "https://www.youtube.com/@TheOdinProject", "reason": "Best free full-stack curriculum — HTML to Node.js with projects throughout."},
            {"title": "Traversy Media Full Stack", "channel": "Traversy Media", "url": "https://www.youtube.com/@TraversyMedia", "reason": "Practical project-based tutorials covering the full stack."},
            {"title": "Fireship — Web Dev in 100 Seconds", "channel": "Fireship", "url": "https://www.youtube.com/@Fireship", "reason": "Fast, accurate technology overviews — great for broadening your full-stack knowledge."},
        ],
        "tips": [
            "Build one complete full-stack project before diversifying — depth beats breadth early on.",
            "Learn PostgreSQL before MongoDB — relational databases are in 80%+ of production systems.",
            "Deploy every project publicly — a live URL on your resume is worth more than screenshots.",
        ],
    }


def _roadmap_stub_backend(domain, level):
    return {
        "overview": f"This roadmap builds your {domain} skills from {level}. Server-side architecture, APIs, databases, and system design are the pillars — master these and you can build the engine of any product.",
        "phases": [
            {"title": "Language & Core Concepts", "description": "Pick your language and master the fundamentals.", "topics": ["Python (Django/FastAPI) or Node.js or Java (Spring)", "OOP & design patterns", "HTTP, REST & API design", "Git & version control"], "duration": "3-4 weeks"},
            {"title": "Databases & Storage", "description": "SQL, NoSQL, caching, and data modelling.", "topics": ["PostgreSQL — advanced queries, indexes, transactions", "MongoDB — document modelling", "Redis — caching & queues", "ORMs (SQLAlchemy, Prisma, Hibernate)"], "duration": "3-4 weeks"},
            {"title": "System Design & Architecture", "description": "Scalable, maintainable backend systems.", "topics": ["Microservices vs monolith", "Message queues (RabbitMQ, Kafka)", "Authentication & authorization (JWT, OAuth2)", "Rate limiting, load balancing, CDN"], "duration": "4-5 weeks"},
            {"title": "DevOps & Interview Prep", "description": "Containers, CI/CD, and backend interview preparation.", "topics": ["Docker & Docker Compose", "CI/CD with GitHub Actions", "System design interviews", "LeetCode medium — trees, graphs, DP"], "duration": "3-4 weeks"},
        ],
        "weekly_plan": [
            {"week": "Week 1-2", "focus": "Language deep-dive — build a REST API from scratch"},
            {"week": "Week 3-4", "focus": "Database integration — CRUD with PostgreSQL + Redis caching"},
            {"week": "Week 5-6", "focus": "Auth + middleware + error handling — production-ready API"},
            {"week": "Week 7-8", "focus": "Dockerise your project + system design practice"},
        ],
        "books": [
            {"title": "Designing Data-Intensive Applications", "author": "Martin Kleppmann", "reason": "The single most important book for backend and systems engineers."},
            {"title": "Clean Architecture", "author": "Robert C. Martin", "reason": "Timeless principles for building maintainable backend systems."},
            {"title": "Database Internals", "author": "Alex Petrov", "reason": "Deep understanding of how databases work under the hood."},
        ],
        "videos": [
            {"title": "Traversy Media — Backend Web Dev", "channel": "Traversy Media", "url": "https://www.youtube.com/@TraversyMedia", "reason": "Project-based backend tutorials in Node, Python, and more."},
            {"title": "Gaurav Sen — System Design", "channel": "Gaurav Sen", "url": "https://www.youtube.com/@gkcs", "reason": "The best system design explanations on YouTube."},
            {"title": "Fireship", "channel": "Fireship", "url": "https://www.youtube.com/@Fireship", "reason": "Fast-paced, accurate overviews of backend technologies."},
        ],
        "tips": [
            "Read 'Designing Data-Intensive Applications' cover to cover — it will change how you think about systems.",
            "Build and deploy a project that handles real load (even 100 req/s) — performance is a skill.",
            "Practice system design questions weekly; they separate mid-level from senior engineers in interviews.",
        ],
    }


# ── Resume-based question generation ─────────────────────────────────────────

def extract_skills_from_resume(resume_text: str, job_role: str) -> list:
    """
    Extract a list of skills from resume text for personalised question targeting.
    Returns a list of skill strings.
    """
    prompt = f"""<|system|>
You are an expert resume parser. Extract the key technical and soft skills from the resume.
Return ONLY a valid JSON array of skill strings — no extra text.
<|user|>
Target Role: {job_role}
Resume:
{resume_text[:3000]}

Return format: ["skill1", "skill2", "skill3", ...]
<|assistant|>
"""
    raw = _call_granite(prompt)
    if raw:
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\[.*\]", raw_clean, re.DOTALL)
        if match:
            try:
                skills = json.loads(match.group())
                return [str(s).strip() for s in skills if str(s).strip()][:15]
            except json.JSONDecodeError:
                pass

    # Fallback: simple keyword extraction
    tech_keywords = [
        "python","java","javascript","typescript","react","node","django","flask",
        "sql","mysql","postgresql","mongodb","docker","kubernetes","aws","gcp","azure",
        "terraform","ansible","git","ci/cd","machine learning","tensorflow","pytorch",
        "html","css","rest","graphql","microservices","agile","scrum","linux",
    ]
    found = []
    text_lower = resume_text.lower()
    for kw in tech_keywords:
        if kw in text_lower:
            found.append(kw.title())
    return found[:10] if found else ["Communication", "Problem Solving", "Teamwork"]


# ── Resume Profile Parser ─────────────────────────────────────────────────────

def parse_resume_profile(resume_text: str) -> dict:
    """
    Extract candidate profile (name, job_role, experience_level, interview_type)
    from resume text so the interview setup form can be auto-filled.

    Returns a dict with keys:
        name            – candidate's full name (str)
        job_role        – most recent / target job role (str)
        experience      – one of the four experience level strings (str)
        interview_type  – "Mixed" | "Technical" | "HR"  (always Mixed from resume)
        skills          – list of detected skill strings
        summary         – 1 sentence profile summary
    """
    prompt = f"""<|system|>
You are an expert resume parser. Extract structured candidate profile information from the resume text.
Return ONLY a valid JSON object — no extra text, no markdown fences.
<|user|>
Resume Text:
{resume_text[:4000]}

Return ONLY this JSON:
{{
  "name": "<candidate full name or 'Unknown' if not found>",
  "job_role": "<most recent or target job title, e.g. Software Engineer, Data Scientist>",
  "experience": "<one of: Entry Level (0-2 years) | Mid Level (2-5 years) | Senior Level (5-8 years) | Lead / Principal (8+ years)>",
  "interview_type": "Mixed",
  "skills": ["<skill1>", "<skill2>", "<skill3>", "<skill4>", "<skill5>"],
  "summary": "<one sentence describing the candidate>"
}}

Rules:
- Infer experience level from years of experience, job titles, or education year mentioned in the resume.
- If the candidate mentions 0-2 years or is a fresher/student, use "Entry Level (0-2 years)".
- If the candidate mentions 2-5 years, use "Mid Level (2-5 years)".
- If the candidate mentions 5-8 years, use "Senior Level (5-8 years)".
- If the candidate mentions 8+ years or is a lead/architect/principal, use "Lead / Principal (8+ years)".
- Extract the most prominent technical skills from the resume.
<|assistant|>
"""
    raw = _call_granite(prompt)
    if raw:
        parsed = _extract_json(raw)
        if parsed:
            try:
                valid_levels = {
                    "Entry Level (0-2 years)",
                    "Mid Level (2-5 years)",
                    "Senior Level (5-8 years)",
                    "Lead / Principal (8+ years)",
                }
                exp = str(parsed.get("experience", "Entry Level (0-2 years)")).strip()
                if exp not in valid_levels:
                    exp = "Entry Level (0-2 years)"
                return {
                    "name":           str(parsed.get("name", "")).strip() or "Candidate",
                    "job_role":       str(parsed.get("job_role", "")).strip() or "Software Engineer",
                    "experience":     exp,
                    "interview_type": "Mixed",
                    "skills":         list(parsed.get("skills", [])),
                    "summary":        str(parsed.get("summary", "")),
                }
            except (ValueError, TypeError):
                pass
        logger.warning("Could not parse resume profile from Granite; using heuristic fallback.")

    # ── Heuristic fallback when Granite unavailable ───────────────────────────
    text  = resume_text.lower()
    lines = resume_text.strip().splitlines()

    # Name: try first non-empty line that looks like a name (no digits, < 5 words)
    name = "Candidate"
    for line in lines[:5]:
        line = line.strip()
        if line and not any(c.isdigit() for c in line) and 1 < len(line.split()) <= 5:
            name = line.title()
            break

    # Job role: scan for common title keywords
    role_keywords = [
        ("machine learning engineer", "Machine Learning Engineer"),
        ("ml engineer",               "Machine Learning Engineer"),
        ("data scientist",            "Data Scientist"),
        ("data analyst",              "Data Analyst"),
        ("devops engineer",           "DevOps Engineer"),
        ("cloud architect",           "Cloud Architect"),
        ("full stack developer",      "Full Stack Developer"),
        ("fullstack developer",       "Full Stack Developer"),
        ("frontend developer",        "Frontend Developer"),
        ("front-end developer",       "Frontend Developer"),
        ("backend developer",         "Backend Developer"),
        ("back-end developer",        "Backend Developer"),
        ("software engineer",         "Software Engineer"),
        ("software developer",        "Software Developer"),
        ("product manager",           "Product Manager"),
        ("business analyst",          "Business Analyst"),
        ("cybersecurity analyst",     "Cybersecurity Analyst"),
        ("ui/ux designer",            "UI/UX Designer"),
        ("ux designer",               "UX Designer"),
    ]
    job_role = "Software Engineer"
    for kw, label in role_keywords:
        if kw in text:
            job_role = label
            break

    # Experience: scan for year mentions
    import re as _re
    exp_match = _re.search(r'(\d+)\s*\+?\s*year', text)
    experience = "Entry Level (0-2 years)"
    if exp_match:
        yrs = int(exp_match.group(1))
        if yrs >= 8:
            experience = "Lead / Principal (8+ years)"
        elif yrs >= 5:
            experience = "Senior Level (5-8 years)"
        elif yrs >= 2:
            experience = "Mid Level (2-5 years)"

    # Skills: keyword scan
    skill_map = {
        "python": "Python", "java ": "Java", "javascript": "JavaScript",
        "typescript": "TypeScript", "react": "React", "node": "Node.js",
        "django": "Django", "flask": "Flask", "spring": "Spring Boot",
        "sql": "SQL", "mysql": "MySQL", "postgresql": "PostgreSQL",
        "mongodb": "MongoDB", "docker": "Docker", "kubernetes": "Kubernetes",
        "aws": "AWS", "azure": "Azure", "gcp": "GCP",
        "terraform": "Terraform", "git": "Git", "ci/cd": "CI/CD",
        "machine learning": "Machine Learning", "tensorflow": "TensorFlow",
        "pytorch": "PyTorch", "pandas": "Pandas", "numpy": "NumPy",
        "html": "HTML", "css": "CSS", "linux": "Linux", "agile": "Agile",
    }
    skills = [label for kw, label in skill_map.items() if kw in text][:8]

    return {
        "name":           name,
        "job_role":       job_role,
        "experience":     experience,
        "interview_type": "Mixed",
        "skills":         skills,
        "summary":        f"{name} appears to be a {job_role} with {experience} experience.",
    }
