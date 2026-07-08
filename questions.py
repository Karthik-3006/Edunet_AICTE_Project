"""
questions.py
------------
Provides a tailored set of interview questions.

Strategy (in priority order):
  1. Ask IBM Granite to generate bespoke questions for the role/experience/type.
  2. If Granite is unavailable or returns too few questions, pad / fall back
     to the static question bank below.
"""

import logging
import random

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static question bank (fallback)
# ---------------------------------------------------------------------------

HR_QUESTIONS = [
    "Tell me about yourself and your career journey so far.",
    "Why are you interested in this role and our company?",
    "Describe a time you faced a conflict at work and how you resolved it.",
    "Where do you see yourself in 5 years?",
    "What is your greatest professional achievement?",
    "How do you handle tight deadlines and pressure?",
    "Describe a situation where you had to adapt to a significant change.",
    "What are your biggest strengths and areas for improvement?",
    "How do you prioritise tasks when everything feels urgent?",
    "Tell me about a time you showed leadership, even without a formal title.",
    "How do you handle feedback or constructive criticism?",
    "Describe a time you went above and beyond for a project or client.",
]

TECHNICAL_QUESTIONS_BY_ROLE = {
    "software engineer": [
        "Explain the difference between REST and GraphQL APIs.",
        "What is the CAP theorem and why does it matter in distributed systems?",
        "Walk me through how you would design a URL shortener.",
        "Explain SOLID principles with a concrete example.",
        "What are the trade-offs between SQL and NoSQL databases?",
        "How does garbage collection work in your primary language?",
        "Explain the concept of eventual consistency.",
        "What strategies do you use for code review?",
        "Describe a challenging bug you fixed and how you debugged it.",
        "How would you improve the performance of a slow database query?",
    ],
    "data scientist": [
        "Explain the bias-variance trade-off.",
        "What is the difference between bagging and boosting?",
        "How do you handle missing data in a dataset?",
        "Explain precision, recall, and F1 score with examples.",
        "What is regularisation and when would you use it?",
        "Describe a machine learning project from data ingestion to deployment.",
        "What is cross-validation and why is it important?",
        "How do you detect and handle outliers?",
        "Explain the difference between supervised and unsupervised learning.",
        "What tools and libraries do you rely on most in your workflow?",
    ],
    "product manager": [
        "How do you prioritise features in a product backlog?",
        "Describe a product you launched from zero to one.",
        "How do you measure the success of a product feature?",
        "What frameworks do you use for roadmap planning?",
        "How do you balance technical debt with new feature development?",
        "Describe how you gather and validate user requirements.",
        "Walk me through how you would define a product strategy.",
        "How do you handle disagreements with the engineering team?",
        "What metrics matter most to you as a PM?",
        "How do you communicate product vision to stakeholders?",
    ],
    "devops engineer": [
        "Explain the difference between CI and CD.",
        "How would you design a zero-downtime deployment pipeline?",
        "What is Infrastructure as Code and which tools have you used?",
        "Explain the 12-factor app methodology.",
        "How do you monitor and respond to production incidents?",
        "Describe your experience with container orchestration (Kubernetes/ECS).",
        "How do you secure secrets in a CI/CD pipeline?",
        "What is the difference between blue-green and canary deployments?",
        "How would you reduce cloud infrastructure costs?",
        "Explain how you handle configuration drift.",
    ],
    "default": [
        "Describe a complex technical problem you solved recently.",
        "What tools and technologies are you most proficient with?",
        "How do you stay current with developments in your field?",
        "Walk me through your typical approach to a new project.",
        "How do you ensure quality in your deliverables?",
        "Describe your experience working in agile or scrum teams.",
        "What is the most challenging technical decision you have made?",
        "How do you approach documentation?",
        "Tell me about a time you had to learn a new technology quickly.",
        "How do you collaborate with cross-functional teams?",
    ],
}

MIXED_EXTRA = [
    "How do you balance technical depth with business impact in your work?",
    "Describe a project where you had to communicate complex ideas to non-technical stakeholders.",
    "How do you mentor or support junior team members?",
]


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_questions(job_role: str, experience: str, interview_type: str, count: int = 7,
                   resume_text: str = "", industry: str = "") -> list:
    """
    Return a list of *count* questions tailored to the inputs.

    Tries IBM Granite first; falls back to the static bank if needed.

    Parameters
    ----------
    job_role       : e.g. "Software Engineer"
    experience     : e.g. "Mid Level (2-5 years)"
    interview_type : "HR", "Technical", or "Mixed"
    count          : number of questions to return (default 7)
    resume_text    : full resume text for personalised question generation
    industry       : target industry / business domain for context
    """
    # ── 1. Try IBM Granite ──────────────────────────────────────────────────
    try:
        from granite_api import generate_questions
        ai_questions = generate_questions(job_role, experience, interview_type, count,
                                          resume_text=resume_text, industry=industry)
        if len(ai_questions) >= count:
            logger.info("Using %d IBM Granite-generated questions.", count)
            return ai_questions[:count]
        if ai_questions:
            logger.info(
                "Granite returned %d questions; padding with static bank.", len(ai_questions)
            )
            # Fall through to pad below
            static = _static_pool(job_role, interview_type)
            random.shuffle(static)
            combined = ai_questions + [q for q in static if q not in ai_questions]
            return combined[:count]
    except Exception as exc:
        logger.warning("Granite question generation skipped: %s", exc)

    # ── 2. Static fallback ──────────────────────────────────────────────────
    return _static_pool_sampled(job_role, interview_type, count)


def _static_pool(job_role: str, interview_type: str) -> list:
    """Return the full static pool for the given role and type (unsampled)."""
    interview_type = interview_type.upper()
    role_key       = job_role.lower().strip()
    tech_pool      = TECHNICAL_QUESTIONS_BY_ROLE.get(role_key, TECHNICAL_QUESTIONS_BY_ROLE["default"])

    if interview_type == "HR":
        return HR_QUESTIONS[:]
    elif interview_type == "TECHNICAL":
        return tech_pool[:]
    else:  # Mixed
        return HR_QUESTIONS[:] + tech_pool[:] + MIXED_EXTRA


def _static_pool_sampled(job_role: str, interview_type: str, count: int) -> list:
    """Return exactly *count* shuffled questions from the static bank."""
    interview_type = interview_type.upper()
    role_key       = job_role.lower().strip()
    tech_pool      = TECHNICAL_QUESTIONS_BY_ROLE.get(role_key, TECHNICAL_QUESTIONS_BY_ROLE["default"])

    if interview_type == "HR":
        pool = HR_QUESTIONS[:]
    elif interview_type == "TECHNICAL":
        pool = tech_pool[:]
    else:  # Mixed
        hr_sample   = random.sample(HR_QUESTIONS, min(3, len(HR_QUESTIONS)))
        tech_sample = random.sample(tech_pool,    min(4, len(tech_pool)))
        pool        = hr_sample + tech_sample + MIXED_EXTRA

    random.shuffle(pool)
    selected = pool[:count]

    # Pad with repeats from pool if pool is smaller than count
    while len(selected) < count:
        selected.append(random.choice(pool))

    return selected
