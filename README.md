# Interview Trainer Agent 🚀
### Powered by Agentic AI, IBM Granite, and RAG Technology

An end-to-end, interactive, and intelligent web platform built to empower students to crack competitive technical and soft-skill job interviews. By leveraging **Retrieval-Augmented Generation (RAG)** and **IBM Granite**, the application delivers role-tailored assessments, live scoring, resume analytics, and highly detailed visual feedback loops.

🔗 **Live Deployment:** [https://edunet-aicte-project.onrender.com/](https://edunet-aicte-project.onrender.com/)  
👤 **Developer:** Karthik Mannem  
🎓 **Affiliation:** AICTE - IBM University Engagement Phase

---

## 📌 Problem Statement & The Challenge
Job seekers and engineering students frequently struggle with interview preparation due to scattered, generic, and unreliable data across various recruitment platforms. Traditional prep tools fail to offer real-time, context-grounded evaluation of technical depth or behavioral readiness, leading to lack of confidence and low conversion rates during highly competitive corporate placement drives.

### Project Objective
To build a cohesive, accessible web ecosystem utilizing **IBM Cloud Lite** services, **Flask**, and **IBM Granite 3.x** models to automate structural profile onboarding, provide a 7-question specialized assessment mode, dynamically analyze resumes, and generate comprehensive roadmap guides for fresher career paths.

---

## ✨ Core & Unique Features

*   **7-Question Comprehensive Mock Test:** Simulates real-time corporate technical rounds. It automatically reviews answers for technical perfection and awards a precise contextual score.
*   **AI Resume Analyzer & Skill Scorer:** Evaluates plaintext resumes and skill lists against competitive global industry standards, providing explicit alignment ratings and pinpointing critical technical keyword gaps.
*   **Dynamic Tri-Component Feedback Output:** Enforces strict structural output rules where the underlying AI must present evaluations using three distinct formatting blocks:
    1.  *Comprehensive Text Summary:* Multi-paragraph critique detailing technical depth and delivery.
    2.  *Detailed Evaluation Matrix Table:* Clear Markdown tables mapping matching keywords and missing competencies.
    3.  *Strategic Path Flow Diagrams:* Text-based ASCII flowcharts outlining logical code paths or career correction routes.
*   **Integrated Voice Assistant:** Boosts portal accessibility by naturally reading aloud primary content blocks, headings, and interview questions.
*   **Fresher Roadmap Curator:** Instantly generates tailored learning pipelines for beginners, linking curated YouTube courses and expert-recommended reference literature.
*   **Eye-Comfort Theme Toggle:** Features a dynamic UI dark/light switcher to prioritize user visual comfort during extended preparation sessions.
*   **Built-in Operational Guidelines:** Interactive walk-through prompts allowing users to quickly navigate and operate the portal seamlessly.

---

## 🛠️ Technology Stack

*   **Cognitive Engine:** IBM Granite Model 3.x (`granite-4-h-small`) for semantic reasoning and multi-format text rendering.
*   **Knowledge Layer:** **Retrieval-Augmented Generation (RAG) Technology** utilizing a local domain matrix index (`questions.py`) to ground models and completely mitigate conversational hallucinations.
*   **Validation & QA Environment:** **IBM BOB Framework** (Heavily utilized to systematically audit backend routing logic, check parameter flags, and optimize token payload pathways).
*   **Backend Controller:** Flask (Python 3.x) microframework managing session states and data pipeline transfers.
*   **Frontend Interface:** Responsive HTML5, CSS3, and Vanilla JavaScript modules driving UI transitions, audio APIs, and tabular renders.
*   **Cloud Hosting Platform:** Render (Containerized web service orchestration).

---

## ⚙️ Programmatic Logic Architecture (Langflow Structure)

Even though built programmatically via Flask rather than a graphical node canvas, the internal execution engine reflects a modular component architecture:

```text
 [User Input / Resume Text] ──> [Flask Endpoint] ──> [Prompt Structuring Agent Component]
                                                               │
┌──────────────────────────────────────────────────────────────┘
▼
[RAG Engine Context Injection (questions.py Data)] ──> [IBM watsonx Connect API Module]
                                                               │
┌──────────────────────────────────────────────────────────────┘
▼
[IBM Granite Core Reasoning Compute] ──> [Tri-Component Parsing Logic Engine]
                                                               │
┌──────────────────────────────────────────────────────────────┘
▼
 [Client Layout Render] ──> Displays Text, Matrix Table Metrics, and Roadmap Pathways
