import os
import re
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.constants import JudgeScoreType
from evalscope.utils.logger import get_logger

logger = get_logger()

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
"""  # noqa: E501


DEFAULT_NUMERIC_SCORE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\"

[Question]
{question}

[Response]
{pred}
"""  # noqa: E501

DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'


class LLMJudge:
    """
    A metric that uses LLM to judge the quality of model predictions by comparing them with reference answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        score_pattern: Optional[str] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        score_type: str = JudgeScoreType.PATTERN,  # 'pattern', 'numeric'
        **kwargs
    ):
        """
        Initialize LLMJudge metric.

        Args:
            api_key (str, optional): API key for OpenAI or compatible service
            api_base (str, optional): API base URL
            model_id (str, optional): Model ID for LLM
            system_prompt (str, optional): System prompt for the judge
            prompt_template (str, optional): Prompt template for the judge
            generation_config (dict, optional): Generation configuration for the judge
            score_pattern (str, optional): Regex pattern to extract score from LLM response
            score_mapping (dict, optional): Mapping from extracted score to float value
            score_type (str, optional): Type of score extraction strategy ('pattern', 'numeric') defaults to 'pattern'.
                - 'pattern': Use score_pattern and score_mapping to extract categorical scores
                - 'numeric': Treat the extracted value as a direct numerical score
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.generation_config = generation_config or {'temperature': 0.0, 'max_tokens': 1024}

        # Default score mapping for A/B pattern
        self.score_type = score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            self.score_pattern = score_pattern or r'\[\[(\d+(?:\.\d+)?)\]\]'
            self.prompt_template = prompt_template or os.environ.get(
                'JUDGE_PROMPT_TEMPLATE', DEFAULT_NUMERIC_SCORE_TEMPLATE
            )
        elif self.score_type == JudgeScoreType.PATTERN:
            # Anchor to only accept a standalone A or B (avoid false positives)
            self.score_pattern = score_pattern or r'^\s*([AB])\s*$'
            self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}. Must be 'pattern' or 'numeric'.")
        self.score_mapping = score_mapping or {'A': 1.0, 'B': 0.0}

        self._init_server_adapter()

    def _init_server_adapter(self):
        from evalscope.api.model import GenerateConfig, get_model

        self.model = get_model(
            model=self.model_id,
            eval_type='openai_api',
            base_url=self.api_url,
            api_key=self.api_key,
            config=GenerateConfig(**self.generation_config),
        )

    def judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and context.
        If messages is provided, it will be used as the input context.

        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
            messages (List[ChatMessage], optional): A list of chat messages to include in the evaluation
        Returns:
            str: The response from the LLM
        """
        input_messages = self._build_input_messages(prompt=prompt, system_prompt=system_prompt, messages=messages)
        try:
            # Send request using ServerModelAdapter
            response = self.model.generate(input_messages)

            # Extract content from response
            llm_response = response.completion
            return llm_response
        except Exception as e:
            error_message = f'Error occurred during {self.model_id}@{self.api_url} LLM judge evaluation: {e}'
            logger.warning(error_message)
            return f'[ERROR] {error_message}'

    def batch_judge(
        self,
        prompts: Optional[List[str]] = None,
        system_prompts: Optional[List[Optional[str]]] = None,
        messages_list: Optional[List[List[ChatMessage]]] = None,
    ) -> List[str]:
        """Judge a batch of prompts/messages via model.batch_generate."""
        if messages_list is None:
            prompts = prompts or []
            system_prompts = system_prompts or [None] * len(prompts)
            messages_list = [
                self._build_input_messages(prompt=prompt, system_prompt=system_prompt)
                for prompt, system_prompt in zip(prompts, system_prompts)
            ]

        if not messages_list:
            return []

        try:
            responses = list(
                self.model.batch_generate(
                    inputs=messages_list,
                    tools=[[] for _ in messages_list],
                    tool_choices=[None for _ in messages_list],
                    configs=[None for _ in messages_list],
                )
            )
            return [response.completion for response in responses]
        except Exception as exc:
            logger.warning(
                f'Batch judge failed for {self.model_id}@{self.api_url}: {exc}. Falling back to single requests.'
            )
            return [self.judge(messages=messages) for messages in messages_list]

    def _build_input_messages(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None,
    ) -> List[ChatMessage]:
        if messages is not None:
            return messages

        system_content = system_prompt or self.system_prompt
        input_messages: List[ChatMessage] = [ChatMessageUser(content=prompt)]
        if system_content:
            input_messages.insert(0, ChatMessageSystem(content=system_content))
        return input_messages

    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None):
        if question is None:
            question = 'Not provided'

        # check variables in prompt_template
        prompt = self.prompt_template
        if '{question}' in self.prompt_template:
            prompt = prompt.replace('{question}', question)
        if '{pred}' in self.prompt_template:
            prompt = prompt.replace('{pred}', pred)
        if '{gold}' in self.prompt_template:
            prompt = prompt.replace('{gold}', gold)
        return prompt

    def get_score(self, response: str) -> float:
        """
        Extract score from LLM response using the configured pattern and mapping.

        Args:
            response (str): The response from the LLM

        Returns:
            float: The numeric score extracted from the response
        """
        if response is None:
            return 0.0

        # choose extraction method based on score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            return self._extract_numeric_score(response)
        elif self.score_type == JudgeScoreType.PATTERN:
            return self._extract_pattern_score(response)

    def _extract_numeric_score(self, response: str) -> float:
        """extract numeric score from the response using the score_pattern"""
        # Find all numeric tokens like [[0.5]] and take the last one (most decisive)
        matches = list(re.finditer(self.score_pattern, response))
        if not matches:
            logger.warning(f"No match found for pattern '{self.score_pattern}' in response: {response}")
            return 0.0

        # iterate from last to first to pick the final rating
        for match in reversed(matches):
            # prefer captured groups
            for group in match.groups():
                if group is None:
                    continue
                try:
                    val = float(group)
                    # clamp to [0, 1] per instruction
                    return max(0.0, min(1.0, val))
                except (ValueError, TypeError):
                    continue
            # fallback: try entire match if groups fail
            try:
                val = float(match.group(0))
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                continue

        logger.warning(f'Failed to convert extracted values to float in response: {response}')
        return 0.0

    def _extract_pattern_score(self, response: str) -> float:
        """use the score_pattern to extract categorical scores"""
        # strict standalone A/B matching using MULTILINE to handle simple outputs
        match = re.search(self.score_pattern, response, re.MULTILINE)
        if match:
            answer = match.group(1) if match.lastindex else match.group(0).strip()
            return self.score_mapping.get(answer, 0.0)
        else:
            logger.warning(f"No match found for pattern '{self.score_pattern}' in response: {response}")
            return 0.0
