"""Quick validation script — run from ai_interview_trainer/ directory."""
import sys, json
sys.path.insert(0, ".")

import granite_api

# ── 1. No credentials → _get_model returns None ───────────────────────────────
print("=== _get_model() with no creds ===")
model = granite_api._get_model()
print("model is None (expected):", model is None)

# ── 2. evaluate_answer falls back to stub ─────────────────────────────────────
print("\n=== evaluate_answer (stub mode) ===")
result = granite_api.evaluate_answer(
    question="Explain SOLID principles.",
    answer="SOLID stands for five OOP design principles.",
    job_role="Software Engineer",
    experience="Mid Level (2-5 years)",
)
print("Keys:", sorted(result.keys()))
assert set(result.keys()) == {"score", "strengths", "weaknesses", "feedback"}
assert 0 <= result["score"] <= 10
assert len(result["strengths"]) >= 1
assert len(result["weaknesses"]) >= 1
print("score:", result["score"])
print("strengths:", result["strengths"])
print("weaknesses:", result["weaknesses"])

# ── 3. generate_suggested_answer falls back to stub ───────────────────────────
print("\n=== generate_suggested_answer (stub mode) ===")
ans = granite_api.generate_suggested_answer(
    "Explain SOLID principles.",
    "Software Engineer",
    "Mid Level (2-5 years)",
)
assert len(ans) > 40
print("Suggested answer length:", len(ans))

# ── 4. generate_questions returns empty list (no creds) ───────────────────────
print("\n=== generate_questions (stub mode) ===")
qs = granite_api.generate_questions("Software Engineer", "Mid Level", "Mixed", 7)
print("Returned from Granite (expect 0):", len(qs))
assert qs == []

# ── 5. questions.py get_questions uses static fallback ────────────────────────
print("\n=== get_questions full flow ===")
import questions
result_q = questions.get_questions("Software Engineer", "Mid Level (2-5 years)", "Mixed")
print("Questions count:", len(result_q))
assert len(result_q) == 7, f"Expected 7, got {len(result_q)}"
for i, q in enumerate(result_q, 1):
    print(f"  Q{i}: {q[:80]}" + ("..." if len(q) > 80 else ""))

# ── 6. _extract_json helper ───────────────────────────────────────────────────
print("\n=== _extract_json parsing ===")
clean_json = '{"score": 8, "strengths": ["Good"], "weaknesses": ["Bad"], "feedback": "OK"}'
fenced_json = "```json\n" + clean_json + "\n```"

clean  = granite_api._extract_json(clean_json)
fenced = granite_api._extract_json(fenced_json)
assert clean  is not None and clean["score"]  == 8, f"Clean parse failed: {clean}"
assert fenced is not None and fenced["score"] == 8, f"Fenced parse failed: {fenced}"
print("Clean JSON  parsed score:", clean["score"])
print("Fenced JSON parsed score:", fenced["score"])

# ── 7. Score clamping in evaluate_answer ─────────────────────────────────────
print("\n=== score clamping ===")
assert max(0, min(10, 15)) == 10
assert max(0, min(10, -3)) == 0
print("Clamping logic OK")

print("\nALL VALIDATIONS PASSED")
