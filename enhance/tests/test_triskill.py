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
from triskill.executor import _direct_seed, _openended_seed_prompt, run_triskill
from triskill.llm import OpenAICompatibleLLM, parse_json_lenient
from triskill.paper_pipeline import audit_artifacts, create_experiment_manifest, join_scores, write_scored_summary
from triskill.runner import run_dataset
from triskill.runtime_skills import _extract_forbidden_code_terms, _select_semantically_spread_words, _select_words_by_similarity
from triskill.state import ElicitationState


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

    def test_bats_uses_relation_module_consensus_for_final_target(self):
        item = {"query": "Complete analogy: berlin : germany :: jakarta : ?", "target_words": ["indonesia"]}
        artifact = run_triskill("bats", item, llm=VerifierOverwritesLLM(), method="triskill_full")

        self.assertIn('"target": "indonesia"', artifact["final_answer"])

    def test_bats_consensus_rejects_copying_visible_input_word(self):
        item = {
            "query": "Complete analogy: conakry : guinea :: london : ?",
            "word_a": "conakry",
            "word_b": "guinea",
            "word_c": "london",
        }
        artifact = run_triskill("bats", item, llm=BATSInputCopyLLM(), method="triskill_full")

        self.assertIn('"target": "uk"', artifact["final_answer"])

    def test_context_fit_tasks_do_not_force_direct_seed(self):
        item = {
            "query": "Replace *approach* with one word.",
            "metaphor_word": "approach",
            "candidate_answers": ["direction", "method"],
        }
        artifact = run_triskill("metaphor", item, llm=MetaphorDirectLLM(), method="triskill_full")

        self.assertIn('"word": "method"', artifact["final_answer"])

    def test_aut_normalizes_candidates_into_uses_list(self):
        item = {"query": "List creative uses for a brick.", "item": "brick"}
        artifact = run_triskill("aut", item, llm=AUTCandidateLLM(), method="triskill_full")

        self.assertIn('"uses"', artifact["final_answer"])
        self.assertIn("doorstop", artifact["final_answer"])
        self.assertIn("garden marker", artifact["final_answer"])

    def test_aut_normalization_filters_schema_placeholders(self):
        output = normalize_selected("aut", {"uses": ["use 1", "use 2", "doorstop", "Doorstop"]})

        self.assertNotIn("use 1", output)
        self.assertNotIn("use 2", output)
        self.assertIn("doorstop", output)

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

    def test_aut_direct_seed_keeps_use_portfolio(self):
        artifact = _direct_seed(
            "aut",
            "List creative uses for a brick.",
            llm=AUTSeedLLM(),
            config={"direct_seed_samples": 1, "direct_seed_max_tokens": 128},
        )

        self.assertEqual(artifact["candidates"][0]["uses"], ["doorstop", "garden marker"])

    def test_openended_tasks_use_direct_seed_as_final_fidelity_anchor(self):
        item = {"query": "Write a reconstruction plan."}
        artifact = run_triskill("transformation", item, llm=OpenEndedAnchorLLM(), method="triskill_full")

        self.assertIn("Shared network time", artifact["final_answer"])
        self.assertNotIn("'type':", artifact["final_answer"])
        self.assertIn("direct_seed", artifact["artifacts"])

    def test_exploratory_openended_tasks_preserve_direct_anchor(self):
        item = {"query": "Write a coherent constrained story."}
        artifact = run_triskill("cs4", item, llm=ExploratoryAnchorLLM(), method="triskill_full")

        self.assertIn("A complete coherent story", artifact["final_answer"])
        self.assertNotIn("regressed rewrite", artifact["final_answer"])

    def test_openended_normalization_extracts_finished_answer_body(self):
        item = {"query": "Solve this in a novel way."}
        artifact = run_triskill("creative_math", item, llm=FinishedBodyLLM(), method="triskill_full")

        self.assertTrue(artifact["final_answer"].startswith("<answer>\nEvelyn opened the cafe door"))
        self.assertNotIn("I need to plan", artifact["final_answer"])
        self.assertNotIn("</think>", artifact["final_answer"])
        self.assertNotIn("</thinking>", artifact["final_answer"])

    def test_openended_normalization_rejects_planning_only_artifact(self):
        item = {"query": "Write a coherent constrained story."}
        artifact = run_triskill("cs4", item, llm=PlanningOnlyLLM(), method="triskill_full")

        self.assertIn("A plain but complete seed story", artifact["final_answer"])
        self.assertNotIn("Thinking Process", artifact["final_answer"])

    def test_story_tasks_keep_direct_anchor_over_long_workflow_plan(self):
        item = {"query": "Write a coherent constrained story."}
        artifact = run_triskill("cs4", item, llm=LongPlanningStoryLLM(), method="triskill_full")

        self.assertIn("A plain but complete seed story", artifact["final_answer"])
        self.assertNotIn("Refining Constraint", artifact["final_answer"])

    def test_story_tasks_extract_finished_draft_from_planning_shell(self):
        item = {"query": "Write a coherent constrained story."}
        artifact = run_triskill("cs4", item, llm=DraftShellStoryLLM(), method="triskill_full")

        self.assertIn("Mara crossed the station", artifact["final_answer"])
        self.assertNotIn("Review against constraints", artifact["final_answer"])
        self.assertNotIn("Drafting Plan", artifact["final_answer"])

    def test_code_tasks_finalize_unfinished_reasoning_anchor(self):
        item = {"query": "Return JSON with think and solve_lines for a solve() function."}
        artifact = run_triskill("neocoder", item, llm=CodeFinalizerLLM(), method="triskill_full")

        self.assertIn('"solve_lines"', artifact["final_answer"])
        self.assertIn("def solve():", artifact["final_answer"])
        self.assertNotIn("Let me analyze", artifact["final_answer"])

    def test_code_tasks_select_candidate_that_passes_visible_example(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Input\nOne integer n.\nOutput\nPrint twice n.\n"
                "Example\nInput\n2\nOutput\n4"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeExampleSelectionLLM(), method="triskill_full")

        self.assertIn("n * 2", artifact["final_answer"])
        self.assertNotIn("print(0)", artifact["final_answer"])

    def test_code_tasks_reject_top_level_solve_call(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function. Do not call solve().\n"
                "Example\nInput\n3\nOutput\n3"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeTopLevelCallLLM(), method="triskill_full")

        self.assertIn("print(input())", artifact["final_answer"])
        self.assertNotIn('solve()"]', artifact["final_answer"])

    def test_code_tasks_apply_generic_forbidden_technique_filter(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Example\nInput\n3\nOutput\n3"
            ),
            "constraints": ["for loop"],
        }
        artifact = run_triskill("neocoder", item, llm=CodeConstraintSelectionLLM(), method="triskill_full")

        self.assertIn("while i < n", artifact["final_answer"])
        self.assertNotIn("for _ in range", artifact["final_answer"])

    def test_code_tasks_reject_if_statement_constraint(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Example\nInput\n3\nOutput\n3"
            ),
            "constraints": ["if statement"],
        }
        artifact = run_triskill("neocoder", item, llm=CodeIfConstraintSelectionLLM(), method="triskill_full")

        self.assertIn("print(input())", artifact["final_answer"])
        self.assertNotIn('"    if n > 0:"', artifact["final_answer"])

    def test_code_tasks_treat_map_as_for_loop_under_visible_constraint(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Programming constraints: DO NOT use the following techniques\n"
                "- for loop\n"
                "Example\nInput\n3\nOutput\n3"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeMapConstraintSelectionLLM(), method="triskill_full")

        self.assertIn("print(input())", artifact["final_answer"])
        self.assertNotIn("map(", artifact["final_answer"])

    def test_code_tasks_parse_forbidden_techniques_from_prompt(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Programming constraints: DO NOT use the following techniques\n"
                "- if statement\n"
                "Example\nInput\n3\nOutput\n3"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeIfConstraintSelectionLLM(), method="triskill_full")

        self.assertIn("print(input())", artifact["final_answer"])
        self.assertNotIn('"    if n > 0:"', artifact["final_answer"])

    def test_code_constraint_parser_ignores_problem_text_words(self):
        state = ElicitationState(
            task_name="neocoder",
            item_id="x",
            raw_item={"query": "The first line contains the number of test cases. Return JSON with solve_lines."},
            original_prompt="The first line contains the number of test cases. Return JSON with solve_lines.",
            level="exploratory",
            workflow="exploratory",
        )

        self.assertEqual(_extract_forbidden_code_terms(state), set())

    def test_code_constraint_parser_reads_constraint_block_bullets(self):
        state = ElicitationState(
            task_name="neocoder",
            item_id="x",
            raw_item={"query": "Return JSON with solve_lines."},
            original_prompt=(
                "Return JSON with solve_lines.\n"
                "Programming constraints: DO NOT use the following techniques\n"
                "- for loop\n"
                "- set\n"
                "Problem statement starts here with ordinary text."
            ),
            level="exploratory",
            workflow="exploratory",
        )

        self.assertEqual(_extract_forbidden_code_terms(state), {"for loop", "set"})

    def test_code_tasks_strip_comments_and_main_guard(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function. Do not call solve(). Do not include comments.\n"
                "Example\nInput\n3\nOutput\n3"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeMainGuardCommentLLM(), method="triskill_full")

        self.assertIn("print(input())", artifact["final_answer"])
        self.assertNotIn("#", artifact["final_answer"])
        self.assertNotIn("__main__", artifact["final_answer"])
        self.assertNotIn('solve()"]', artifact["final_answer"])

    def test_code_tasks_repair_when_all_candidates_fail_visible_example(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Input\nOne integer n.\nOutput\nPrint n plus one.\n"
                "Example\nInput\n4\nOutput\n5"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeRepairLLM(), method="triskill_full")

        self.assertIn("n + 1", artifact["final_answer"])
        self.assertNotIn("print(0)", artifact["final_answer"])

    def test_code_tasks_do_not_accept_constraint_violating_repair_over_clean_candidate(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Programming constraints: DO NOT use the following techniques\n"
                "- for loop\n"
                "Input\nOne integer n.\nOutput\nPrint n.\n"
                "Example\nInput\n4\nOutput\n4"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeViolatingRepairLLM(), method="triskill_full")

        self.assertIn("print(0)", artifact["final_answer"])
        self.assertNotIn("for _ in range", artifact["final_answer"])

    def test_code_tasks_apply_constraint_preserving_final_rewrite(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Programming constraints: DO NOT use the following techniques\n"
                "- for loop\n"
                "Input\nOne integer n.\nOutput\nPrint n.\n"
                "Example\nInput\n4\nOutput\n4"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeConstraintRewriteLLM(), method="triskill_full")

        self.assertIn("while i < n", artifact["final_answer"])
        self.assertNotIn("for _ in range", artifact["final_answer"])

    def test_code_tasks_keep_direct_anchor_when_no_candidate_passes_example(self):
        item = {
            "query": (
                "Return JSON with solve_lines for a solve() function.\n"
                "Example\nInput\n2\nOutput\n5"
            )
        }
        artifact = run_triskill("neocoder", item, llm=CodeDirectAnchorLLM(), method="triskill_full")

        self.assertIn("print(1)", artifact["final_answer"])
        self.assertNotIn("print(0)", artifact["final_answer"])

    def test_code_direct_seed_prompt_lists_visible_forbidden_terms(self):
        prompt = (
            "Return JSON with solve_lines.\n"
            "Programming constraints: DO NOT use the following techniques\n"
            "- for loop\n"
            "- if statement\n"
            "Problem text mentions set the variable to zero."
        )
        seed_prompt = _openended_seed_prompt("neocoder", prompt)

        self.assertIn("Visible forbidden techniques parsed from the prompt: for loop, if statement", seed_prompt)
        self.assertIn("avoid `for ... in ...`", seed_prompt)
        self.assertNotIn("Visible forbidden techniques parsed from the prompt: for loop, if statement, set", seed_prompt)

    def test_aut_extracts_real_ideas_from_raw_reasoning(self):
        item = {"query": "List creative uses for a boot.", "item": "boot"}
        artifact = run_triskill("aut", item, llm=AUTRawIdeasLLM(), method="triskill_full")

        self.assertIn("Planter", artifact["final_answer"])
        self.assertIn("Doorstop", artifact["final_answer"])
        self.assertNotIn("use 1", artifact["final_answer"])

    def test_aut_filters_workflow_meta_and_renders_portfolio(self):
        item = {"query": "List creative uses for a belt.", "item": "belt"}
        artifact = run_triskill("aut", item, llm=AUTMetaFallbackLLM(), method="triskill_full")

        self.assertIn("waist support", artifact["final_answer"])
        self.assertIn("garden tie", artifact["final_answer"])
        self.assertNotIn("TriSkill pipeline", artifact["final_answer"])
        self.assertNotIn("The prompt asks", artifact["final_answer"])

    def test_aut_living_entity_filters_moral_statements_and_adds_constructive_uses(self):
        item = {"query": "What are some creative uses for a baby?", "item": "baby"}
        artifact = run_triskill("aut", item, llm=AUTLivingEntityLLM(), method="triskill_full")

        self.assertIn("learning about baby care needs", artifact["final_answer"])
        self.assertIn("designing safer spaces for a baby", artifact["final_answer"])
        self.assertNotIn("Respect Life", artifact["final_answer"])
        self.assertNotIn("fundamentally wrong", artifact["final_answer"])

    def test_aut_polishes_large_label_pool_into_self_contained_uses(self):
        item = {"query": "What are some creative uses for a box?", "item": "box"}
        artifact = run_triskill("aut", item, llm=AUTLabelPoolLLM(), method="triskill_full")

        self.assertIn("use the box as a pet bed", artifact["final_answer"])
        self.assertIn("turn the box into a seedling tray", artifact["final_answer"])
        self.assertNotIn('"Egg fryer"', artifact["final_answer"])
        self.assertNotIn('"Toy organizer"', artifact["final_answer"])

    def test_aut_direct_seed_ignores_schema_example_when_raw_has_ideas(self):
        artifact = _direct_seed(
            "aut",
            "List creative uses for a candle.",
            llm=AUTThinkingSchemaLLM(),
            config={"direct_seed_samples": 1, "direct_seed_max_tokens": 128},
        )

        self.assertEqual(artifact["candidates"][0]["uses"], ["emergency light", "wax seal", "table centerpiece"])

    def test_dat_normalization_filters_skill_names_and_fills(self):
        output = normalize_selected("dat", {"words": ["semantic_domain_expansion", "Whale"]})

        self.assertNotIn("semantic_domain_expansion", output)
        self.assertIn('"Whale"', output)
        self.assertIn('"nebula"', output)

    def test_semantic_spread_selector_avoids_root_cluster(self):
        words = [
            "quantum",
            "quasar",
            "quintessence",
            "quagmire",
            "quorum",
            "apple",
            "trumpet",
            "covenant",
            "stratus",
            "laughter",
        ]

        selected = _select_semantically_spread_words(words, final_count=6)

        q_words = [word for word in selected if word.lower().startswith("qu")]
        self.assertLessEqual(len(q_words), 1)
        self.assertIn("apple", selected)
        self.assertIn("trumpet", selected)

    def test_embedding_spread_selector_uses_pairwise_semantic_distance(self):
        words = ["quasar", "nebula", "cucumber", "invoice", "violin"]
        similarity = [
            [1.0, 0.90, 0.10, 0.10, 0.10],
            [0.90, 1.0, 0.10, 0.10, 0.10],
            [0.10, 0.10, 1.0, 0.20, 0.20],
            [0.10, 0.10, 0.20, 1.0, 0.20],
            [0.10, 0.10, 0.20, 0.20, 1.0],
        ]

        selected = _select_words_by_similarity(words, similarity, final_count=4)

        self.assertEqual(len(selected), 4)
        self.assertFalse({"quasar", "nebula"}.issubset(set(selected)))
        self.assertIn("cucumber", selected)
        self.assertIn("invoice", selected)

    def test_openai_client_has_retries(self):
        client = OpenAICompatibleLLM(api_url="https://example.invalid/v1", model="m", retries=3, retry_interval=0)

        self.assertEqual(client.retries, 3)
        self.assertTrue(client.disable_thinking)

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


class BATSInputCopyLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            return '{"target":"london"}'
        if "candidate_recombination" in lower or "constraint_preservation" in lower:
            return '{"target":"uk"}'
        if "combination_verification" in lower:
            return '{"target":"london"}'
        return '{"target":"london"}'


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


class AUTCandidateLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "semantic_deduplication" in lower:
            return '{"candidates":[{"candidate":"doorstop"},{"candidate":"garden marker"}]}'
        if "feasibility_evaluation" in lower:
            return '{"candidates":[{"candidate":"doorstop"},{"candidate":"garden marker"}],"best_candidate":{"candidate":"doorstop"}}'
        return '{"candidates":[{"candidate":"doorstop"},{"candidate":"garden marker"}]}'


class AUTSeedLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return '{"uses":["doorstop","garden marker"]}'


class OpenEndedAnchorLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return "Shared network time replaces local clocks, with dispatch, timetables, records, training, and audits all keyed to the same standard."
        if "final triskill answer normalizer" in lower:
            return "{'type': 'reconstruction_text', 'bad': 'dict-shaped artifact'}"
        return "This is exploratory prose, not JSON."


class ExploratoryAnchorLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return "A complete coherent story that satisfies the visible constraints from beginning to end."
        if "final triskill answer normalizer" in lower:
            return "regressed rewrite"
        return "This exploratory artifact should not replace the anchor."


class FinishedBodyLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return "A plain but complete seed story."
        if "candidate_generation" in lower:
            return (
                "<answer>\n"
                "I need to plan the story and list all constraints first.\n"
                "</thinking>\n\n"
                "Evelyn opened the cafe door, felt the room's grief as a pressure behind her ribs, "
                "and still smiled at the stranger by the window.\n"
                "</answer>"
            )
        return "{}"


class PlanningOnlyLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return "A plain but complete seed story."
        if "candidate_generation" in lower:
            return (
                "<answer>\nThinking Process:\n"
                "1. Analyze the request.\n"
                "2. Drafting Strategy: list constraints.\n"
                "3. Constraint Conflict: reconcile details.\n"
                "4. Refining Constraint 1 through 23.\n"
                "</answer>"
            )
        return "{}"


class LongPlanningStoryLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return "A plain but complete seed story."
        if "candidate_generation" in lower:
            return "Thinking Process:\\n" + "\\n".join(f"Refining Constraint {idx}: plan details." for idx in range(1, 30))
        return "{}"


class DraftShellStoryLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return (
                "Drafting Plan:\n1. List constraints.\n\n"
                "Drafting:\n"
                "Mara crossed the station with a paper lantern in one hand and a lie on her tongue. "
                "The delayed train gave her exactly enough time to apologize, forgive herself, and leave before dawn.\n\n"
                "Review against constraints:\n1. Yes."
            )
        return "{}"


class CodeFinalizerLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final-answer renderer" in lower:
            return '{"think":"use stdin and print once","solve_lines":["def solve():","    print(1)"]}'
        if "final answer anchor" in lower:
            return "Let me analyze the programming problem first. The key insight is to produce code later."
        return "{}"


class CodeExampleSelectionLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "candidate_generation" in lower:
            return '{"solve_lines":["def solve():","    n = int(input())","    print(n * 2)"]}'
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeTopLevelCallLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "candidate_generation" in lower:
            return '{"solve_lines":["def solve():","    print(input())"]}'
        return '{"solve_lines":["def solve():","    print(input())","solve()"]}'


class CodeConstraintSelectionLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "execution_verification" in lower:
            return (
                '{"solve_lines":["def solve():","    n = int(input())","    i = 0","    total = 0",'
                '"    while i < n:","        total += 1","        i += 1","    print(total)"]}'
            )
        if "candidate_generation" in lower:
            return (
                '{"solve_lines":["def solve():","    n = int(input())","    total = 0",'
                '"    for _ in range(n):","        total += 1","    print(total)"]}'
            )
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeIfConstraintSelectionLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "execution_verification" in lower:
            return '{"solve_lines":["def solve():","    print(input())"]}'
        if "candidate_generation" in lower:
            return (
                '{"solve_lines":["def solve():","    n = input()",'
                '"    if n > 0:","        print(n)"]}'
            )
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeMapConstraintSelectionLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "execution_verification" in lower:
            return '{"solve_lines":["def solve():","    print(input())"]}'
        if "candidate_generation" in lower:
            return '{"solve_lines":["def solve():","    xs = list(map(int, input().split()))","    print(xs[0])"]}'
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeMainGuardCommentLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return (
            '{"solve_lines":["# forbidden comment","def solve():","    print(input())",'
            '"if __name__ == \\"__main__\\":","    solve()"]}'
        )


class CodeRepairLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "repair this programming answer" in lower:
            return '{"solve_lines":["def solve():","    n = int(input())","    print(n + 1)"]}'
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeViolatingRepairLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "repair this programming answer" in lower:
            return (
                '{"solve_lines":["def solve():","    n = int(input())","    total = 0",'
                '"    for _ in range(n):","        total += 1","    print(total)"]}'
            )
        return '{"solve_lines":["def solve():","    print(0)"]}'


class CodeConstraintRewriteLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "rewrite this programming answer" in lower:
            return (
                '{"solve_lines":["def solve():","    n = int(input())","    i = 0",'
                '"    while i < n:","        i += 1","    print(i)"]}'
            )
        return (
            '{"solve_lines":["def solve():","    n = int(input())","    total = 0",'
            '"    for _ in range(n):","        total += 1","    print(total)"]}'
        )


class CodeDirectAnchorLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "final answer anchor" in lower:
            return '{"solve_lines":["def solve():","    print(1)"]}'
        return '{"solve_lines":["def solve():","    print(0)"]}'


class AUTRawIdeasLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "give your best direct answer first" in lower:
            return "Thinking Process:\nIdeas: Planter, Doorstop, Cable holder\n" + '{"uses":["use 1","use 2"]}'
        return "{}"


class AUTMetaFallbackLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "create a final answer for this alternative-uses prompt" in lower:
            return '{"uses":["waist support","garden tie","door latch","book strap","cable organizer","improvised handle","wall hanging","training aid","bag repair","game boundary","shelf support","curtain tie"]}'
        if "candidate_generation" in lower or "coverage_balancing" in lower:
            return '{"uses":["The prompt asks for creative uses.","Item: belt","This is my current skill in the TriSkill pipeline"]}'
        return '{"uses":["use 1","use 2","use 3"]}'


class AUTLivingEntityLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return '{"uses":["The subject is a human infant requiring care.","Respect Life","The concept is fundamentally wrong."]}'


class AUTLabelPoolLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        lower = prompt.lower()
        if "select and rewrite a final answer" in lower:
            return (
                '{"uses":["use the box as a pet bed","turn the box into a seedling tray",'
                '"store craft tools in the box","use the box as a mail sorter",'
                '"make the box into a puppet theater","use the box to protect fragile gifts",'
                '"turn the box into a tabletop recycling bin","use the box as a portable first aid kit",'
                '"make the box into a shadow display frame","use the box to organize charging cables",'
                '"turn the box into a small compost collector","use the box as a game-piece arena"]}'
            )
        return (
            '{"uses":["Pet bed","Plant pot","Storage bin","Toy organizer","Desk drawer","Egg fryer",'
            '"Wine rack","Cookie jar","use the box as a pet bed","turn the box into a seedling tray",'
            '"store craft tools in the box","use the box as a mail sorter","make the box into a puppet theater",'
            '"use the box to protect fragile gifts","turn the box into a tabletop recycling bin",'
            '"use the box as a portable first aid kit","make the box into a shadow display frame",'
            '"use the box to organize charging cables","turn the box into a small compost collector",'
            '"use the box as a game-piece arena","a temporary radiation shield block for low-level alpha/beta sources",'
            '"a counterweight for a submerged hydroelectric turbine blade","Bread box","Egg warmer","Soda dispenser","Knife block",'
            '"Fork rest","Mirror frame","Drawer insert"]}'
        )


class AUTThinkingSchemaLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return 'Thinking Process:\nSchema: {"uses":["use 1","use 2","use 3"]}\nIdeas: emergency light, wax seal, table centerpiece'


if __name__ == "__main__":
    unittest.main()
