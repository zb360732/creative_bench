import json
import tempfile
import unittest
from pathlib import Path

from triskill import TriSkillEnhancer, build_artifact, enhance_prompt, normalize_selected, profile_task
from triskill.analysis import write_summary
from triskill.dataset import enhance_dataset_file
from triskill.diagnostics import diagnose_transformation
from triskill.evalscope_bridge import artifacts_to_predictions
from triskill.execution_hooks import verify_math_solution, verify_python_code
from triskill.executor import _direct_seed, run_triskill
from triskill.llm import OpenAICompatibleLLM, parse_json_lenient
from triskill.paper_pipeline import audit_artifacts, create_experiment_manifest, join_scores, write_scored_summary
from triskill.runner import run_dataset


class TriSkillTest(unittest.TestCase):
    def test_rat_profile_selects_combinational_operator_modules(self):
        profile = profile_task("rat")
        enhancer = TriSkillEnhancer(profile)
        plan = enhancer.plan()

        self.assertEqual(plan["creativity_level"], "combinational")
        self.assertIn("associative_bridge", plan["canonical_metrics"])
        self.assertIn("unit_extraction", plan["skills"])
        self.assertIn("relation_property_abstraction", plan["skills"])
        self.assertIn("candidate_recombination", plan["skills"])
        self.assertIn("combination_verification", plan["skills"])

    def test_aut_profile_selects_exploratory_skills(self):
        profile = profile_task("aut")
        enhancer = TriSkillEnhancer(profile)
        plan = enhancer.plan()

        self.assertEqual(plan["creativity_level"], "exploratory")
        self.assertIn("novelty", plan["canonical_metrics"])
        self.assertIn("candidate_generation", plan["skills"])
        self.assertIn("feasibility_evaluation", plan["skills"])
        self.assertEqual(plan["skills"][-1], "output_normalization")

    def test_enhanced_prompt_preserves_answer_schema(self):
        prompt = "Find a single word for cottage/swiss/cake."
        enhanced = enhance_prompt(prompt, "rat")

        self.assertIn(prompt, enhanced)
        self.assertIn('<answer>{"word": "connecting_word"}</answer>', enhanced)
        self.assertIn("Do not use hidden gold answers", enhanced)
        self.assertIn("do not reveal", enhanced.lower())

    def test_transformation_profile_uses_reconstruction_skills(self):
        profile = profile_task("transformational_creativity")
        plan = TriSkillEnhancer(profile).plan()

        self.assertEqual(plan["task_name"], "transformation")
        self.assertEqual(plan["creativity_level"], "transformational")
        self.assertIn("system_reconstruction", plan["skills"])
        self.assertIn("residue_audit", plan["skills"])

    def test_profile_contains_budgets_and_output_normalization(self):
        plan = TriSkillEnhancer(profile_task("dat")).plan()

        self.assertEqual(plan["budgets"]["final_count"], 10)
        self.assertIn("output_normalization", plan["skills"])
        self.assertEqual(plan["canonical_metric_weights"]["semantic_diversity"], "high")

    def test_build_artifact_excludes_gold_fields(self):
        item = {
            "query": "Find a single word for cottage/swiss/cake.",
            "question": "cottage/swiss/cake",
            "answer": "cheese",
            "candidate_answers": ["cheese"],
            "category": "RAT",
        }
        artifact = build_artifact("rat", item)

        self.assertNotIn("answer", artifact["safe_item"])
        self.assertNotIn("candidate_answers", artifact["safe_item"])
        self.assertIn("excluded scoring-only fields", artifact["warnings"][0])
        self.assertIn("TriSkill creativity elicitation instructions", artifact["enhanced_prompt"])

    def test_normalize_selected_outputs_answer_block(self):
        output = normalize_selected("rat", {"word": "cheese"})

        self.assertIn("<answer>", output)
        self.assertIn('"word": "cheese"', output)

    def test_dataset_file_enhancement_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rat.json"
            output_path = Path(tmp) / "enhanced.jsonl"
            input_path.write_text(
                json.dumps([
                    {"query": "Find a single word for cottage/swiss/cake.", "answer": "cheese", "category": "RAT"}
                ]),
                encoding="utf-8",
            )

            rows = enhance_dataset_file("rat", input_path, output_path)

            self.assertEqual(len(rows), 1)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertIn("enhanced_prompt", payload)
            self.assertNotIn("answer", payload["safe_item"])

    def test_lenient_json_parser_handles_answer_block(self):
        parsed = parse_json_lenient('<answer>\n{"word":"cheese",}\n</answer>')

        self.assertEqual(parsed, {"word": "cheese"})

    def test_full_executor_with_fake_llm(self):
        item = {"query": "Find a single word for cottage/swiss/cake.", "answer": "cheese", "category": "RAT"}
        artifact = run_triskill("rat", item, llm=FakeLLM(), method="triskill_full")

        self.assertIn("<answer>", artifact["final_answer"])
        self.assertIn('"word": "cheese"', artifact["final_answer"])
        self.assertGreaterEqual(artifact["num_llm_calls"], 4)
        self.assertNotIn("answer", artifact["safe_item"])
        self.assertIn("direct_seed", artifact["artifacts"])

    def test_full_executor_normalizes_rationale_fallback(self):
        item = {"query": "Find a single word for cottage/swiss/cake.", "answer": "cheese", "category": "RAT"}
        artifact = run_triskill("rat", item, llm=RationaleFallbackLLM(), method="triskill_full")

        self.assertIn("<answer>", artifact["final_answer"])
        self.assertIn('"word": "cheese"', artifact["final_answer"])
        self.assertNotIn("thinking step by step", artifact["final_answer"])

    def test_bats_prefers_direct_seed_over_weaker_verifier(self):
        item = {"query": "Complete analogy: berlin : germany :: conakry : ?", "target_words": ["guinea"]}
        artifact = run_triskill("bats", item, llm=VerifierOverwritesLLM(), method="triskill_full")

        self.assertIn('"target": "guinea"', artifact["final_answer"])

    def test_context_fit_tasks_do_not_force_direct_seed(self):
        item = {
            "query": "Replace *approach* with one word.",
            "metaphor_word": "approach",
            "candidate_answers": ["direction", "method"],
        }
        artifact = run_triskill("metaphor", item, llm=MetaphorDirectLLM(), method="triskill_full")

        self.assertIn('"word": "method"', artifact["final_answer"])

    def test_placeholder_words_are_rejected(self):
        output = normalize_selected("rat", {"word": "connecting_word"})

        self.assertIn('"word": ""', output)

    def test_empty_cleaned_word_is_safe(self):
        output = normalize_selected("rat", "!!!")

        self.assertIn('"word": ""', output)

    def test_discourse_words_are_rejected(self):
        output = normalize_selected("rat", {"word": "Alright"})

        self.assertIn('"word": ""', output)

    def test_explicit_choice_phrase_can_seed_candidate(self):
        item = {"query": "Find a single word for cottage/swiss/cake.", "category": "RAT"}
        artifact = run_triskill("rat", item, llm=ExplicitChoiceLLM(), method="triskill_full")

        self.assertIn('"word": "cheese"', artifact["final_answer"])
        self.assertEqual(artifact["artifacts"]["direct_seed"]["candidates"][0]["word"], "cheese")

    def test_repeated_direct_seed_candidates_are_recorded_as_soft_evidence(self):
        artifact = _direct_seed(
            "rat",
            "Find a single word for cottage/swiss/cake.",
            llm=ConsensusSeedLLM(),
            config={"direct_seed_samples": 3, "direct_seed_max_tokens": 128},
        )

        self.assertEqual(artifact["candidates"][0]["seed_votes"], 2)
        self.assertEqual(artifact["num_calls"], 3)

    def test_dat_normalization_filters_skill_names_and_fills(self):
        output = normalize_selected("dat", {"words": ["semantic_domain_expansion", "Whale"]})

        self.assertNotIn("semantic_domain_expansion", output)
        self.assertIn('"Whale"', output)
        self.assertIn('"nebula"', output)

    def test_openai_client_has_retries(self):
        client = OpenAICompatibleLLM(api_url="https://example.invalid/v1", model="m", retries=3, retry_interval=0)

        self.assertEqual(client.retries, 3)

    def test_runner_prompt_only_and_without_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rat.json"
            output_path = Path(tmp) / "run.jsonl"
            input_path.write_text(json.dumps([{"query": "Find a word.", "answer": "x"}]), encoding="utf-8")

            rows = run_dataset("rat", str(input_path), str(output_path), method="triskill_without_verifier", llm=FakeLLM())

            self.assertEqual(len(rows), 1)
            self.assertNotIn("combination_verification", rows[0]["skills"])
            self.assertIn("output_normalization", rows[0]["skills"])

    def test_analysis_and_prediction_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "artifacts.jsonl"
            summary_path = Path(tmp) / "summary.json"
            pred_path = Path(tmp) / "predictions.json"
            row = {
                "task_name": "rat",
                "method": "triskill_full",
                "level": "combinational",
                "item_id": "1",
                "final_answer": '<answer>{"word":"cheese"}</answer>',
                "parse_success": True,
                "output_length": 1,
                "num_llm_calls": 4,
            }
            artifact_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = write_summary([artifact_path], summary_path)
            predictions = artifacts_to_predictions(artifact_path, pred_path)

            self.assertEqual(summary["num_rows"], 1)
            self.assertEqual(predictions[0]["prediction"], '<answer>{"word":"cheese"}</answer>')

    def test_transformation_diagnostics(self):
        diag = diagnose_transformation("We will patch the existing system and keep the old process.")

        self.assertIn("local_patching", diag["active_failure_modes"])
        self.assertGreater(diag["num_active_failure_modes"], 0)

    def test_execution_hooks_are_conservative(self):
        ok = verify_python_code("print('hi')", expected_stdout="hi")
        unchecked = verify_python_code("def solve():\n    pass")
        math = verify_math_solution("A proof sketch")

        self.assertTrue(ok["execution_pass"])
        self.assertIsNone(unchecked["execution_pass"])
        self.assertIsNone(math["execution_pass"])

    def test_paper_pipeline_manifest_audit_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            artifact = Path(tmp) / "artifact.jsonl"
            scores = Path(tmp) / "scores.jsonl"
            joined = Path(tmp) / "joined.jsonl"
            summary = Path(tmp) / "scored_summary.json"

            manifest_rows = create_experiment_manifest(manifest, tasks={"rat": "rat.json"}, methods=("direct", "triskill_full"), limit=3)
            artifact.write_text(
                json.dumps({
                    "task_name": "rat",
                    "method": "direct",
                    "safe_item": {"query": "q"},
                    "original_prompt": "q",
                    "enhanced_prompt": "q",
                    "final_answer": "a",
                    "parse_success": True,
                    "output_length": 1,
                    "num_llm_calls": 1,
                    "warnings": [],
                }) + "\n",
                encoding="utf-8",
            )
            scores.write_text(json.dumps({"id": 0, "score": 0.5}) + "\n", encoding="utf-8")

            audit = audit_artifacts(artifact)
            joined_rows = join_scores(artifact, scores, joined)
            scored = write_scored_summary([joined], summary)

            self.assertEqual(len(manifest_rows), 2)
            self.assertTrue(audit["pass"])
            self.assertEqual(joined_rows[0]["scores"]["score"], 0.5)
            self.assertEqual(scored["task_method"][0]["mean_score"], 0.5)


class FakeLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if "output_normalization" in prompt.lower():
            return '<answer>{"word":"cheese"}</answer>'
        if "scored_candidates" in prompt or "combination_verification" in prompt:
            return '{"scored_candidates":[{"candidate":"cheese","score":3,"valid":true}],"best_candidate":"cheese"}'
        return '{"candidates":[{"word":"cheese","connections":[{"clue":"cottage","connection":"cottage cheese"}],"all_three_fit":true}],"best_candidate":"cheese"}'


class RationaleFallbackLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        if "final triskill output normalizer" in prompt.lower():
            return "After checking the compounds, the final answer is cheese."
        return "I am thinking step by step and not returning JSON yet."


class VerifierOverwritesLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            return '{"target":"guinea"}'
        if "combination_verification" in lower or "constraint_preservation" in lower:
            return '{"best_candidate":{"target":"indonesia","score":1}}'
        return '{"candidates":[{"target":"indonesia"}],"best_candidate":{"target":"indonesia"}}'


class MetaphorDirectLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            return '{"word":"direction"}'
        if "combination_verification" in lower or "constraint_preservation" in lower:
            return '{"best_candidate":{"word":"method","score":1}}'
        return '{"candidates":[{"word":"method"}],"best_candidate":{"word":"method"}}'


class ExplicitChoiceLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            return "I compare the compounds and choose cheese."
        if "combination_verification" in lower or "constraint_preservation" in lower:
            return '{"best_candidate":{"word":"cheese","score":1}}'
        return '{"candidates":[{"word":"cheese"}],"best_candidate":{"word":"cheese"}}'


class ConsensusSeedLLM:
    def __init__(self):
        self.seed_calls = 0

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            self.seed_calls += 1
            if self.seed_calls == 1:
                return '{"word":"cake"}'
            return '{"word":"cheese"}'
        if "combination_verification" in lower or "constraint_preservation" in lower:
            return '{"best_candidate":{"word":"cake","score":1}}'
        return '{"candidates":[{"word":"cake"}],"best_candidate":{"word":"cake"}}'


if __name__ == "__main__":
    unittest.main()
