# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.llm_judge import LLMJudge
from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / 'dataprocess'
    / 'transformation'
    / 'research'
    / 'outputs'
    / 'problem_method'
    / 'problem_method'
    / 'problem_method.jsonl'
)
_DEFAULT_JUDGE_CONFIG_PATH = Path(__file__).resolve().parents[3] / 'run' / 'llm_judge.json'
_REPO_ROOT = Path(__file__).resolve().parents[3]

_PROMPT_TEMPLATE = (
    "You are given a research problem from a target paper and a method from a reference paper.\n"
    "Your task is to propose a concrete plan to solve the problem by explicitly adapting the given method.\n"
    "You must use the specified method; do not substitute unrelated techniques.\n\n"
    "[Target Paper]\nTitle: {problem_title}\nProblem: {problem}\n\n"
    "[Reference Paper]\nTitle: {method_title}\nMethod: {method}\n\n"
    "Provide a clear solution plan with core idea, steps, and experimental setup.\n"
    "Return your answer as JSON inside <answer> tags using the following schema:\n"
    "<answer>\n"
    "{{\n"
    "  \"proposal\": \"...\",\n"
    "  \"steps\": [\"...\"],\n"
    "  \"experiments\": [\"...\"]\n"
    "}}\n"
    "</answer>"
)

_FEASIBILITY_PROMPT = (
    "You are judging whether the proposal is feasible and whether it truly uses the given method to solve the given problem.\n"
    "Be slightly lenient: if the plan is plausible and reasonably grounded, mark YES.\n"
    "Mark NO if the plan is infeasible, unrelated, or ignores the specified method.\n\n"
    "Problem: {problem}\n"
    "Method: {method}\n"
    "Proposal: {proposal}\n\n"
    "Answer with only YES or NO."
)

_NOVELTY_PROMPT = (
    "You are judging the novelty of a proposal that adapts a given method to solve a given problem.\n"
    "Rate the novelty/transformative adaptation from 1 (routine) to 5 (highly novel).\n"
    "Consider how creatively the method is re-purposed and whether the plan goes beyond a trivial application.\n\n"
    "Problem: {problem}\n"
    "Method: {method}\n"
    "Proposal: {proposal}\n\n"
    "Answer with only a single integer 1-5."
)


@register_benchmark(
    BenchmarkMeta(
        name='problem_method',
        pretty_name='Problem-Method Transfer',
        tags=[Tags.REASONING, Tags.INSTRUCTION_FOLLOWING, Tags.CUSTOM],
        description=(
            'Generate a solution plan for a target paper problem by explicitly adapting a specified reference paper method. '
            'Metrics cover feasibility (LLM-judge voting), novelty, flexibility (semantic distance), and fluency (feasible count).'
        ),
        dataset_id=str(_DEFAULT_DATASET_PATH),
        subset_list=['default'],
        default_subset='default',
        metric_list=['feasibility', 'novelty', 'flexibility', 'fluency'],
        eval_split='test',
        prompt_template='{query}',
        review_timeout=60,
        extra_params={
            'evaluation_mode': {
                'type': 'str',
                'description': 'Evaluation mode: "simplified" (skip judges) or "full" (3 LLM judges).',
                'value': 'simplified'
            },
            'judge_api_url': {
                'type': 'str',
                'description': 'API URL for LLM judges in full mode (optional).',
                'value': None
            },
            'judge_api_key': {
                'type': 'str',
                'description': 'API key for LLM judges in full mode (optional).',
                'value': None
            },
            'judge_model_id': {
                'type': 'str',
                'description': 'Model ID for LLM judges in full mode (optional).',
                'value': None
            },
            'cluster_count': {
                'type': 'int',
                'description': 'Number of clusters to form; cluster centers are used as target papers.',
                'value': 50
            },
            'max_problems_per_target': {
                'type': 'int',
                'description': 'Max problems sampled from each target paper (<=0 means all).',
                'value': 2
            },
            'max_refs_per_target': {
                'type': 'int',
                'description': 'Max reference papers sampled per target paper.',
                'value': 3
            },
            'max_methods_per_ref': {
                'type': 'int',
                'description': 'Max methods sampled from each reference paper (<=0 means all).',
                'value': 1
            },
            'max_samples': {
                'type': 'int',
                'description': 'Optional hard cap for total samples (<=0 means no cap).',
                'value': 0
            },
            'reference_selection': {
                'type': 'str',
                'description': 'Reference paper selection strategy: "farthest", "random", "farthest_mid_between", or "near_mid_far".',
                'value': 'farthest'
            },
            'similarity_backend': {
                'type': 'str',
                'description': 'Similarity backend: "embedding" (default) or "tfidf".',
                'value': 'embedding'
            },
            'embedding_model': {
                'type': 'str',
                'description': 'Sentence-Transformer model name/path for embedding backend.',
                'value': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'random_seed': {
                'type': 'int',
                'description': 'Random seed for sampling.',
                'value': 42
            },
        },
    )
)
class ProblemMethodAdapter(DefaultDataAdapter):
    """Adapter for problem-method transfer benchmark."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        params = self.extra_params or {}
        self.evaluation_mode = params.get('evaluation_mode', 'simplified')
        self.cluster_count = int(params.get('cluster_count', 50))
        self.max_problems_per_target = int(params.get('max_problems_per_target', 2))
        self.max_refs_per_target = int(params.get('max_refs_per_target', 3))
        self.max_methods_per_ref = int(params.get('max_methods_per_ref', 1))
        self.max_samples = int(params.get('max_samples', 0))
        self.reference_selection = str(params.get('reference_selection', 'farthest')).strip().lower()
        self.random_seed = int(params.get('random_seed', 42))
        self.similarity_backend = str(params.get('similarity_backend', 'embedding')).strip().lower()
        self.embedding_model = str(params.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')).strip()

        self._judges: Optional[List[LLMJudge]] = None
        self._encoder = None
        self._encoder_lock = threading.RLock()

        logger.info(
            'ProblemMethodAdapter init: mode=%s cluster_count=%s max_problems=%s max_refs=%s max_methods=%s',
            self.evaluation_mode,
            self.cluster_count,
            self.max_problems_per_target,
            self.max_refs_per_target,
            self.max_methods_per_ref,
        )
        logger.info('Similarity backend=%s embedding_model=%s', self.similarity_backend, self.embedding_model)

    def _load_judge_config(self) -> Tuple[str, str, str]:
        params = self.extra_params or {}
        api_url = params.get('judge_api_url')
        api_key = params.get('judge_api_key')
        model_id = params.get('judge_model_id')

        if api_url and model_id:
            return str(api_url), str(api_key or 'EMPTY'), str(model_id)

        if not _DEFAULT_JUDGE_CONFIG_PATH.exists():
            raise FileNotFoundError(f'LLM judge config not found: {_DEFAULT_JUDGE_CONFIG_PATH}')

        data = json.loads(_DEFAULT_JUDGE_CONFIG_PATH.read_text(encoding='utf-8'))
        models = data.get('models', [])
        if not isinstance(models, list) or not models:
            raise ValueError(f'Invalid judge config, missing models list: {_DEFAULT_JUDGE_CONFIG_PATH}')

        entry = models[0]
        model_id = os.path.expandvars(str(entry.get('model') or entry.get('name') or '').strip())
        api_url = os.path.expandvars(str(entry.get('api_url') or '').strip())
        api_key = os.path.expandvars(str(entry.get('api_key', 'EMPTY')))
        api_key_env = entry.get('api_key_env')
        if api_key_env:
            api_key = os.getenv(str(api_key_env), api_key)
        if api_key in {'', 'YOUR_API_KEY'}:
            api_key = os.getenv('EVALSCOPE_API_KEY', api_key)
        if api_key in {'', 'YOUR_API_KEY'}:
            api_key = os.getenv('OPENAI_API_KEY', api_key)

        if not model_id or not api_url:
            raise ValueError(f'Invalid judge config entry: {entry}')

        return api_url, api_key, model_id

    def _init_judges(self):
        if self.evaluation_mode != 'full':
            return
        if self._judges is not None:
            return

        api_url, api_key, model_id = self._load_judge_config()
        self._judges = []
        for _ in range(3):
            self._judges.append(
                LLMJudge(
                    api_key=api_key,
                    api_url=api_url,
                    model_id=model_id,
                    generation_config={'temperature': 0.0, 'max_tokens': 256},
                )
            )
        logger.info('Initialized %d LLM judges using %s', len(self._judges), model_id)

    def _get_judges(self) -> List[LLMJudge]:
        if self._judges is None:
            self._init_judges()
        if not self._judges:
            raise ValueError('LLM judges not initialized; set evaluation_mode="full" for judge-based scoring.')
        return self._judges

    def load_from_disk(self, dataset_name_or_path=None, subset_list=None, **kwargs):
        if dataset_name_or_path is None:
            dataset_name_or_path = self.dataset_id

        data_path = Path(dataset_name_or_path)
        if not data_path.exists():
            raise FileNotFoundError(f'Dataset not found at: {data_path}')

        records = self._load_records(data_path)
        samples = self._build_samples(records)

        if self.limit is not None:
            if isinstance(self.limit, float) and 0 < self.limit < 1:
                limit_num = int(len(samples) * self.limit)
            else:
                limit_num = int(self.limit)
            samples = samples[:limit_num]
            logger.info('Limited to %d samples (limit=%s)', len(samples), self.limit)

        return {'default': samples}, None

    def _load_records(self, data_path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen = set()
        with data_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                problems = [p for p in (obj.get('problems') or []) if str(p).strip()]
                methods = [m for m in (obj.get('methods') or []) if str(m).strip()]
                if not problems or not methods:
                    continue
                key = (obj.get('paper_id'), obj.get('abstract_sha1') or obj.get('file_path'))
                if key in seen:
                    continue
                seen.add(key)
                obj['problems'] = problems
                obj['methods'] = methods
                records.append(obj)

        logger.info('Loaded %d records from %s', len(records), data_path)
        return records

    def _paper_text(self, record: Dict[str, Any]) -> str:
        parts = [
            record.get('title', ''),
            record.get('one_liner', ''),
            ' '.join(record.get('problems', [])),
            ' '.join(record.get('methods', [])),
        ]
        return ' '.join(p for p in parts if p)

    def _get_encoder(self):
        """Lazy-load embedding encoder (aligned with AUT setup)."""
        with self._encoder_lock:
            if self._encoder is not None:
                return self._encoder
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise ImportError('sentence-transformers is required for embedding backend') from exc

            model_cache_dir = _REPO_ROOT / 'dataprocess' / 'model'
            model_cache_dir.mkdir(parents=True, exist_ok=True)

            safe_name = self.embedding_model.replace('/', '--')
            local_root = model_cache_dir / f'models--{safe_name}'
            snapshot_dir = local_root / 'snapshots'
            candidate_path = None
            if snapshot_dir.exists():
                snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
                if snapshots:
                    candidate_path = snapshots[0]
            elif local_root.exists() and (local_root / 'config.json').exists():
                candidate_path = local_root

            try:
                if candidate_path and candidate_path.exists():
                    logger.info('Loading embedding model locally from %s', candidate_path)
                    self._encoder = SentenceTransformer(str(candidate_path), device='cpu', local_files_only=True)
                else:
                    logger.info('Loading embedding model %s (cache=%s)', self.embedding_model, model_cache_dir)
                    self._encoder = SentenceTransformer(
                        self.embedding_model,
                        device='cpu',
                        cache_folder=str(model_cache_dir),
                    )
            except Exception as exc:
                logger.warning('Failed to load embedding model, fallback to tfidf backend: %s', exc)
                self.similarity_backend = 'tfidf'
                self._encoder = None
            return self._encoder

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self.similarity_backend != 'embedding':
            raise ValueError('Embedding encoder requested but backend is not embedding')
        encoder = self._get_encoder()
        embeddings = encoder.encode(texts, show_progress_bar=False, device='cpu')
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _build_feature_matrix(self, records: List[Dict[str, Any]]):
        texts = [self._paper_text(r) for r in records]
        if self.similarity_backend == 'embedding':
            return self._encode_texts(texts)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        return vectorizer.fit_transform(texts)

    def _build_samples(self, records: List[Dict[str, Any]]) -> List[Sample]:
        rng = random.Random(self.random_seed)

        feature_matrix = self._build_feature_matrix(records)

        cluster_count = min(max(1, self.cluster_count), len(records))
        labels = None
        target_indices = list(range(len(records)))

        if cluster_count > 1:
            kmeans = KMeans(n_clusters=cluster_count, random_state=self.random_seed, n_init='auto')
            labels = kmeans.fit_predict(feature_matrix)
            distances = kmeans.transform(feature_matrix)
            target_indices = []
            for cluster_id in range(cluster_count):
                members = np.where(labels == cluster_id)[0]
                if members.size == 0:
                    continue
                closest = members[np.argmin(distances[members, cluster_id])]
                target_indices.append(int(closest))

        target_set = set(target_indices)
        reference_indices = [i for i in range(len(records)) if i not in target_set]
        if not reference_indices:
            reference_indices = list(range(len(records)))

        samples: List[Sample] = []
        for target_idx in target_indices:
            target = records[target_idx]
            problems = self._select_items(target.get('problems', []), self.max_problems_per_target, rng)
            if not problems:
                continue

            ref_candidates = self._select_references(
                target_idx=target_idx,
                reference_indices=reference_indices,
                matrix=feature_matrix,
                rng=rng,
            )
            if not ref_candidates:
                continue

            for ref_idx, distance, distance_norm in ref_candidates:
                ref = records[ref_idx]
                methods = self._select_items(ref.get('methods', []), self.max_methods_per_ref, rng)
                if not methods:
                    continue
                for problem in problems:
                    for method in methods:
                        prompt = _PROMPT_TEMPLATE.format(
                            problem_title=target.get('title', ''),
                            problem=problem,
                            method_title=ref.get('title', ''),
                            method=method,
                        )
                        sample = Sample(
                            input=[ChatMessageUser(content=prompt)],
                            target='',
                            id=len(samples),
                            metadata={
                                'problem_paper_id': target.get('paper_id'),
                                'problem_title': target.get('title'),
                                'problem': problem,
                                'method_paper_id': ref.get('paper_id'),
                                'method_title': ref.get('title'),
                                'method': method,
                                'cluster_id_problem': int(labels[target_idx]) if labels is not None else None,
                                'cluster_id_method': int(labels[ref_idx]) if labels is not None else None,
                                'semantic_distance': float(distance_norm),
                                'semantic_distance_raw': float(distance),
                            },
                        )
                        samples.append(sample)
                        if self.max_samples > 0 and len(samples) >= self.max_samples:
                            logger.info('Reached max_samples=%d', self.max_samples)
                            return samples

        logger.info('Built %d samples', len(samples))
        return samples

    def _select_items(self, items: List[str], max_items: int, rng: random.Random) -> List[str]:
        filtered = [str(item).strip() for item in items if str(item).strip()]
        if not filtered:
            return []
        if max_items <= 0 or max_items >= len(filtered):
            return filtered
        return rng.sample(filtered, max_items)

    def _select_references(
        self,
        target_idx: int,
        reference_indices: List[int],
        matrix,
        rng: random.Random,
    ) -> List[Tuple[int, float, float]]:
        candidates = [idx for idx in reference_indices if idx != target_idx]
        if not candidates:
            return []
        max_refs = max(1, self.max_refs_per_target)

        target_vec = matrix[target_idx : target_idx + 1]
        distances = cosine_distances(target_vec, matrix[candidates]).flatten()
        ordered = sorted(zip(candidates, distances), key=lambda x: x[1])  # near -> far
        total = len(ordered)
        if total == 0:
            return []

        def _quantile_by_rank(rank: int) -> float:
            if total <= 1:
                return 0.5
            return float(rank) / float(total - 1)

        quantile_map = {idx: _quantile_by_rank(rank) for rank, (idx, _) in enumerate(ordered)}

        if self.reference_selection == 'random':
            chosen = rng.sample(candidates, min(max_refs, len(candidates)))
            chosen_dist = cosine_distances(target_vec, matrix[chosen]).flatten().tolist()
            return [
                (idx, dist, quantile_map.get(idx, 0.0))
                for idx, dist in zip(chosen, chosen_dist)
            ]

        selected: List[Tuple[int, float, float]] = []

        def _pick_positions(pos_list: List[int]) -> List[int]:
            seen = set()
            picks: List[int] = []
            for pos in pos_list:
                pos = max(0, min(total - 1, pos))
                if pos in seen:
                    continue
                seen.add(pos)
                picks.append(pos)
            return picks

        if self.reference_selection == 'near_mid_far':
            positions = _pick_positions([0, total // 2, total - 1])
            for pos in positions:
                idx, dist = ordered[pos]
                selected.append((idx, dist, quantile_map.get(idx, 0.0)))
        elif self.reference_selection == 'farthest_mid_between':
            positions = _pick_positions([total - 1, total // 2, (total + total // 2) // 2])
            for pos in positions:
                idx, dist = ordered[pos]
                selected.append((idx, dist, quantile_map.get(idx, 0.0)))
        else:  # default: farthest
            ordered_desc = list(reversed(ordered))
            for idx, dist in ordered_desc[: min(max_refs, len(ordered_desc))]:
                selected.append((idx, dist, quantile_map.get(idx, 0.0)))

        # Fill remaining slots (if any) with nearest-remaining to reach max_refs
        if len(selected) < min(max_refs, total):
            seen = {item[0] for item in selected}
            for idx, dist in ordered:
                if idx in seen:
                    continue
                selected.append((idx, dist, quantile_map.get(idx, 0.0)))
                if len(selected) >= min(max_refs, total):
                    break

        return selected[: min(max_refs, len(selected))]

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', prediction, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return prediction.strip()

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        metadata = task_state.metadata or {}
        problem = metadata.get('problem', '')
        method = metadata.get('method', '')
        proposal = filtered_prediction.strip()

        feasibility_result = self._evaluate_feasibility(problem, method, proposal)
        is_feasible = feasibility_result['final_decision'] == 'YES'

        novelty_result = self._evaluate_novelty(problem, method, proposal)
        novelty_score = novelty_result['average_score'] if is_feasible else 0.0

        flexibility_score = metadata.get('semantic_distance', 0.0) if is_feasible else 0.0
        fluency_score = 1.0 if is_feasible else 0.0

        score.value = {
            'feasibility': 1.0 if is_feasible else 0.0,
            'novelty': float(novelty_score),
            'flexibility': float(flexibility_score),
            'fluency': float(fluency_score),
        }
        score.metadata = {
            'feasibility': feasibility_result,
            'novelty': novelty_result,
            'semantic_distance': metadata.get('semantic_distance', 0.0),
            'semantic_distance_raw': metadata.get('semantic_distance_raw', 0.0),
            'cluster_id_problem': metadata.get('cluster_id_problem'),
            'cluster_id_method': metadata.get('cluster_id_method'),
        }
        score.main_score_name = 'feasibility'
        return score

    def _evaluate_feasibility(self, problem: str, method: str, proposal: str) -> Dict[str, Any]:
        if self.evaluation_mode != 'full':
            return {'final_decision': 'YES', 'mode': 'simplified'}

        judges = self._get_judges()
        results = []
        for idx, judge in enumerate(judges, start=1):
            prompt = _FEASIBILITY_PROMPT.format(problem=problem, method=method, proposal=proposal)
            response = judge.judge(prompt=prompt)
            decision = self._extract_yes_no(response)
            results.append({'judge': idx, 'response': response, 'decision': decision})

        yes_count = sum(1 for r in results if r['decision'] == 'YES')
        final = 'YES' if yes_count >= 2 else 'NO'
        return {'final_decision': final, 'votes': results, 'mode': 'full'}

    def _evaluate_novelty(self, problem: str, method: str, proposal: str) -> Dict[str, Any]:
        if self.evaluation_mode != 'full':
            return {'average_score': 3.0, 'mode': 'simplified'}

        judges = self._get_judges()
        scores = []
        results = []
        for idx, judge in enumerate(judges, start=1):
            prompt = _NOVELTY_PROMPT.format(problem=problem, method=method, proposal=proposal)
            response = judge.judge(prompt=prompt)
            rating = self._extract_rating(response)
            scores.append(rating)
            results.append({'judge': idx, 'response': response, 'score': rating})

        average_score = float(sum(scores)) / float(len(scores)) if scores else 0.0
        return {'average_score': average_score, 'votes': results, 'mode': 'full'}

    @staticmethod
    def _extract_yes_no(response: str) -> str:
        if not response:
            return 'NO'
        text = response.strip().upper()
        if 'YES' in text:
            return 'YES'
        if 'NO' in text:
            return 'NO'
        if re.fullmatch(r'[AB]', text):
            return 'YES' if text == 'A' else 'NO'
        return 'NO'

    @staticmethod
    def _extract_rating(response: str) -> int:
        if not response:
            return 1
        match = re.search(r'\b([1-5])\b', response)
        if match:
            return int(match.group(1))
        return 1

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        total = len(sample_scores)
        if total == 0:
            logger.warning('No samples to aggregate')
            return []

        feasible_scores = [s for s in sample_scores if s.score.value.get('feasibility', 0.0) >= 0.5]
        feasible_count = len(feasible_scores)

        feasibility_ratio = feasible_count / total
        novelty_vals = [s.score.value.get('novelty', 0.0) for s in feasible_scores]
        flexibility_vals = [s.score.value.get('flexibility', 0.0) for s in feasible_scores]

        novelty_score = float(sum(novelty_vals)) / float(len(novelty_vals)) if novelty_vals else 0.0
        flexibility_score = float(sum(flexibility_vals)) / float(len(flexibility_vals)) if flexibility_vals else 0.0

        agg_scores = [
            AggScore(metric_name='feasibility', score=feasibility_ratio, num=total, metadata={
                'feasible_count': feasible_count,
                'total': total,
            }),
            AggScore(metric_name='novelty', score=novelty_score, num=len(novelty_vals) or 0, metadata={
                'feasible_count': feasible_count,
                'total': total,
            }),
            AggScore(metric_name='flexibility', score=flexibility_score, num=len(flexibility_vals) or 0, metadata={
                'feasible_count': feasible_count,
                'total': total,
            }),
            AggScore(metric_name='fluency', score=float(feasible_count), num=total, metadata={
                'feasible_count': feasible_count,
                'total': total,
            }),
        ]

        return agg_scores
