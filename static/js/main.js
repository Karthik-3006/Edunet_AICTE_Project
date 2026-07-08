/**
 * main.js — AI Interview Trainer
 * Handles: homepage form → start interview → question/answer loop → summary
 */

/* ─────────────────────────────────────────────────────────────────────
   HOME PAGE
   ───────────────────────────────────────────────────────────────────── */
function initHomePage() {
  const form        = document.getElementById("profileForm");
  const startBtn    = document.getElementById("startBtn");
  const radioCards  = document.querySelectorAll(".radio-card");

  if (!form) return;

  // Radio card visual toggle
  radioCards.forEach(card => {
    card.addEventListener("click", () => {
      radioCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      card.querySelector("input[type='radio']").checked = true;
    });
  });

  // Word count on any textarea (interview page reuses this too)
  document.querySelectorAll("textarea").forEach(ta => {
    ta.addEventListener("input", () => updateWordCount(ta));
  });

  // Form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!validateProfileForm()) return;

    startBtn.disabled = true;
    startBtn.innerHTML = '<div class="spinner" style="width:18px;height:18px;border-width:3px;margin:0"></div> Starting…';

    const payload = {
      name:           document.getElementById("name").value.trim(),
      job_role:       document.getElementById("jobRole").value.trim(),
      experience:     document.getElementById("experience").value,
      interview_type: document.querySelector("input[name='interview_type']:checked").value,
    };

    try {
      const res  = await fetch("/start_interview", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        showAlert(data.error || "Failed to start interview. Please try again.");
        resetStartButton();
        return;
      }

      // Store first question in sessionStorage so interview.html can pick it up
      sessionStorage.setItem("firstQuestion", JSON.stringify({
        question:         data.question,
        question_number:  data.question_number,
        total_questions:  data.total_questions,
      }));

      window.location.href = "/interview";
    } catch (err) {
      showAlert("Network error — please check your connection and try again.");
      resetStartButton();
    }

    function resetStartButton() {
      startBtn.disabled = false;
      startBtn.innerHTML = '<span class="btn-icon">&#9654;</span> Start Interview';
    }
  });
}

function validateProfileForm() {
  let valid = true;

  const fields = [
    { id: "name",       errId: "nameError",       msg: "Please enter your name." },
    { id: "jobRole",    errId: "jobRoleError",     msg: "Please enter a job role." },
    { id: "experience", errId: "experienceError",  msg: "Please select your experience level." },
  ];

  fields.forEach(({ id, errId, msg }) => {
    const el  = document.getElementById(id);
    const err = document.getElementById(errId);
    if (!el.value.trim()) {
      el.classList.add("invalid");
      if (err) err.textContent = msg;
      valid = false;
    } else {
      el.classList.remove("invalid");
      if (err) err.textContent = "";
    }
  });

  return valid;
}


/* ─────────────────────────────────────────────────────────────────────
   INTERVIEW PAGE
   ───────────────────────────────────────────────────────────────────── */

// State held in memory for the current session
const state = {
  currentQuestion:       "",
  currentQuestionNumber: 1,
  totalQuestions:        7,
  lastEvaluation:        null,
};

function initInterviewPage() {
  const meta = document.getElementById("sessionMeta");
  if (!meta) return;   // not on interview page

  state.totalQuestions = parseInt(meta.dataset.total, 10) || 7;

  // Retrieve the first question seeded from the homepage call
  const stored = sessionStorage.getItem("firstQuestion");
  if (!stored) {
    window.location.href = "/";
    return;
  }

  const { question, question_number, total_questions } = JSON.parse(stored);
  sessionStorage.removeItem("firstQuestion");

  state.totalQuestions        = total_questions;
  state.currentQuestion       = question;
  state.currentQuestionNumber = question_number;

  renderQuestion(question, question_number);

  // Wire buttons
  document.getElementById("submitAnswerBtn").addEventListener("click", submitAnswer);
  document.getElementById("nextQuestionBtn").addEventListener("click", loadNextQuestion);
  document.getElementById("reviewAnswerBtn").addEventListener("click", reviewAnswer);

  // Word count
  const ta = document.getElementById("answerInput");
  ta.addEventListener("input", () => updateWordCount(ta));
}

function renderQuestion(question, number) {
  document.getElementById("questionText").textContent  = question;
  document.getElementById("questionBadge").textContent = `Q${number}`;

  const meta = document.getElementById("sessionMeta");
  document.getElementById("questionMeta").textContent  =
    `${meta ? meta.dataset.type : "Mixed"} Interview · ${meta ? meta.dataset.role : ""}`;

  updateProgress(number, state.totalQuestions);

  // Reset answer area
  const ta = document.getElementById("answerInput");
  ta.value = "";
  updateWordCount(ta);

  // Show question card, hide feedback
  showSection("questionCard");
  hideSection("feedbackCard");

  // Reset next button label
  const nextBtn = document.getElementById("nextQuestionBtn");
  if (nextBtn) {
    nextBtn.innerHTML = 'Next Question <span class="btn-icon">&#8594;</span>';
  }
}

async function submitAnswer() {
  const ta     = document.getElementById("answerInput");
  const answer = ta.value.trim();

  if (!answer) {
    ta.style.borderColor = "var(--danger)";
    ta.placeholder = "⚠ Please type your answer before submitting.";
    setTimeout(() => {
      ta.style.borderColor = "";
      ta.placeholder = "Type your answer here. Be specific, use the STAR method for behavioural questions…";
    }, 2500);
    return;
  }

  showSection("loadingOverlay");

  try {
    const res  = await fetch("/submit_answer", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ answer }),
    });
    const data = await res.json();

    if (!res.ok) {
      hideSection("loadingOverlay");
      showAlert(data.error || "Could not evaluate your answer. Please try again.");
      return;
    }

    state.lastEvaluation = data;
    renderFeedback(data);
  } catch (err) {
    showAlert("Network error — please check your connection.");
  } finally {
    hideSection("loadingOverlay");
  }
}

function renderFeedback(data) {
  const { evaluation, suggested_answer, question_number, is_last, next_question, next_question_number } = data;
  const score = evaluation.score;

  // Score ring animation
  document.getElementById("scoreNumber").textContent = score;
  const circumference = 314;
  const offset = circumference - (score / 10) * circumference;
  const fill = document.getElementById("ringFill");
  fill.style.strokeDashoffset = offset;
  // Colour by score
  fill.style.stroke = score >= 7 ? "var(--success)" : score >= 5 ? "var(--warning)" : "var(--danger)";

  // Summary text
  document.getElementById("scoreSummary").textContent = evaluation.feedback;

  // Strengths
  const sl = document.getElementById("strengthsList");
  sl.innerHTML = evaluation.strengths.map(s => `<li>${escHtml(s)}</li>`).join("");

  // Weaknesses
  const wl = document.getElementById("weaknessesList");
  wl.innerHTML = evaluation.weaknesses.map(w => `<li>${escHtml(w)}</li>`).join("");

  // Suggested answer
  document.getElementById("suggestedText").textContent = suggested_answer;

  // Next button
  const nextBtn = document.getElementById("nextQuestionBtn");
  if (is_last) {
    nextBtn.innerHTML = 'View Full Summary <span class="btn-icon">&#9654;</span>';
    nextBtn.onclick = () => { window.location.href = "/summary"; };
  } else {
    nextBtn.innerHTML = 'Next Question <span class="btn-icon">&#8594;</span>';
    // Store next question info for loadNextQuestion()
    nextBtn.dataset.nextQuestion       = next_question;
    nextBtn.dataset.nextQuestionNumber = next_question_number;
    nextBtn.onclick = loadNextQuestion;
  }

  showSection("feedbackCard");
  hideSection("questionCard");

  // Smooth scroll to feedback
  document.getElementById("feedbackCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function loadNextQuestion() {
  const nextBtn = document.getElementById("nextQuestionBtn");
  const q  = nextBtn.dataset.nextQuestion;
  const qn = parseInt(nextBtn.dataset.nextQuestionNumber, 10);

  if (!q) {
    window.location.href = "/summary";
    return;
  }

  state.currentQuestion       = q;
  state.currentQuestionNumber = qn;

  renderQuestion(q, qn);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function reviewAnswer() {
  showSection("questionCard");
  hideSection("feedbackCard");
  window.scrollTo({ top: 0, behavior: "smooth" });
}


/* ─────────────────────────────────────────────────────────────────────
   SHARED UTILITIES
   ───────────────────────────────────────────────────────────────────── */
function updateProgress(current, total) {
  const pct = Math.round((current / total) * 100);
  document.getElementById("progressBar").style.width    = pct + "%";
  document.getElementById("progressPercent").textContent = pct + "%";
  document.getElementById("progressLabel").textContent  = `Question ${current} of ${total}`;
}

function updateWordCount(textarea) {
  const wc  = document.getElementById("wordCount");
  if (!wc) return;
  const words = textarea.value.trim().split(/\s+/).filter(Boolean).length;
  wc.textContent = `${words} word${words !== 1 ? "s" : ""}`;
}

function showSection(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}
function hideSection(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

function showAlert(msg) {
  // Simple inline banner — avoids browser alert()
  let banner = document.getElementById("alertBanner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "alertBanner";
    banner.style.cssText = `
      position:fixed; top:70px; left:50%; transform:translateX(-50%);
      background:#fef2f2; border:1px solid #fecaca; color:#dc2626;
      padding:12px 24px; border-radius:8px; font-size:14px; font-weight:600;
      z-index:999; max-width:90vw; text-align:center;
      box-shadow: 0 4px 16px rgba(0,0,0,.1);
    `;
    document.body.appendChild(banner);
  }
  banner.textContent = msg;
  banner.style.display = "block";
  setTimeout(() => { banner.style.display = "none"; }, 4000);
}

function escHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
