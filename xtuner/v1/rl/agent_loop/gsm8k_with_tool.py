import copy
import json
import re
from typing import cast

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger


logger = get_logger()


class GSM8KToolAgentLoopConfig(AgentLoopConfig):
    max_turns: int

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "GSM8KToolAgentLoop":
        return GSM8KToolAgentLoop(
            max_turns=self.max_turns,
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            enable_batch_judge=self.enable_batch_judge,
        )


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict


class GSM8KToolAgentLoop(AgentLoop):
    def __init__(
        self,
        max_turns: int,
        rollout_ctl: RolloutController,
        hf_checkpoint: str,
        sample_params: SampleParams | None,
        judger: Judger | None = None,
        enable_batch_judge: bool = False,
    ):
        super().__init__(
            rollout_ctl=rollout_ctl,
            hf_checkpoint=hf_checkpoint,
            sample_params=sample_params,
            judger=judger,
            enable_batch_judge=enable_batch_judge,
        )
        self.max_turns = max_turns
        self.tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

    def calc_gsm8k_reward(self, answer: dict, ground_truth: str) -> float:
        from xtuner.v1.rl.judger.gsm8k import compute_reward

        extra_info = {"score": 1.0, "format_score": 0}
        actual_answer = answer.get("answer", "")
        if not actual_answer.startswith("#### "):
            actual_answer = "#### " + actual_answer
        return compute_reward(actual_answer, ground_truth, extra_info)

    def extract_tool_calls(self, token_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        text = self.tokenizer.decode(token_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_pattern.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=arguments))
            except Exception as e:
                logger.error(f"Error parsing tool call JSON: {e}")
                continue

        content = self.tool_call_pattern.sub("", text)
        return content, function_calls

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        # 已发布样本（重试组中保留的 COMPLETED 成员）直接复用，避免重复生成与重复追加。
        if rollout_state.train_meta is not None:
            return rollout_state
        if rollout_state.response_length > 0:
            # 不复用上一轮的 partial response：gsm8k 的工具 token 语义 mask 不随 state 持久化，
            # 续写无法重建 labels。旧实现由 worker 在 partial rollout 关闭时 reset；这里统一 reset。
            rollout_state.reset_response()

        base_sample_params = copy.deepcopy(rollout_state.sample_params or self.sample_params)
        max_len = base_sample_params.max_tokens
        prompt_ids = list(rollout_state.prompt_ids or [])
        prompt_length = rollout_state.effective_prompt_length()
        if prompt_length is None:
            prompt_length = len(prompt_ids)
        if not prompt_ids and rollout_state.input_ids is not None:
            prompt_ids = list(rollout_state.input_ids[:prompt_length])
        assert prompt_ids and prompt_length > 0, f"Prompt ids cannot be None or empty in data: {rollout_state}"

        final_response_mask: list[int] = []
        final_logprobs: list[float] = []
        logprobs_available = True
        cur_turn = 0
        while True:
            response_length = rollout_state.response_length
            if cur_turn >= self.max_turns or response_length >= max_len:
                break

            # 每轮都是一次新的生成；重置 status 以免命中 worker 的 COMPLETED 早退。
            rollout_state.status = Status.INIT
            rollout_state.tokens = prompt_ids + list(rollout_state.response_ids or [])
            rollout_state.sample_params = copy.deepcopy(base_sample_params)
            rollout_state.sample_params.max_tokens = max_len - response_length

            before = rollout_state.response_length
            assert self.rollout_ctl is not None
            rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
            cur_turn += 1
            if rollout_state.status != Status.COMPLETED:
                return rollout_state

            generated_length = rollout_state.response_length - before
            final_response_mask.extend([1] * generated_length)
            if rollout_state.logprobs is None:
                logprobs_available = False
            elif generated_length:
                final_logprobs.extend(list(rollout_state.logprobs)[-generated_length:])
            # TODO: 处理 routed_experts, 要注意这里涉及到是否要解引用的问题

            if rollout_state.response_length >= max_len:
                break

            current_turn_ids = list(rollout_state.response_ids or [])[-generated_length:] if generated_length else []
            _, function_calls = self.extract_tool_calls(current_turn_ids)
            if not function_calls:
                break

            tool_messages = []
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = function_call.arguments
                if tool_name == "calc_gsm8k_reward":
                    answer = tool_args
                    ground_truth = cast(dict, rollout_state.reward_model).get("ground_truth", "")
                    function_results = self.calc_gsm8k_reward(answer, ground_truth)
                    tool_message = {
                        "role": "tool",
                        "content": json.dumps({"result": function_results}, ensure_ascii=False),
                    }
                    tool_messages.append(tool_message)

            # 工具 token 追加进同一条训练序列；mask=0 表示不参与训练。
            tools_response_ids = self.tokenizer.apply_chat_template(tool_messages, remove_system_prompt=True)
            rollout_state.append_response_tokens(tools_response_ids)
            final_logprobs.extend([0.0] * len(tools_response_ids))
            final_response_mask.extend([0] * len(tools_response_ids))

        # 截断到 max_len，并按最终 suffix + mask 重建 labels/logprobs/response。
        if rollout_state.input_ids is not None:
            del rollout_state.input_ids[prompt_length + max_len :]
        response_ids = list(rollout_state.response_ids or [])
        response_mask = final_response_mask[: len(response_ids)]
        if len(response_mask) < len(response_ids):
            raise ValueError(
                f"response mask length={len(response_mask)} does not cover response suffix={len(response_ids)}"
            )
        rollout_state.labels = [-100] * prompt_length + [
            resp_id if mask_value else -100 for resp_id, mask_value in zip(response_ids, response_mask)
        ]
        if logprobs_available:
            final_logprobs = final_logprobs[: len(response_ids)]
            final_logprobs.extend([0.0] * (len(response_ids) - len(final_logprobs)))
            rollout_state.logprobs = [0.0] * prompt_length + final_logprobs
        else:
            rollout_state.logprobs = None
        rollout_state.response = self.tokenizer.decode(response_ids)
        if self.judger is not None and not self.enable_batch_judge:
            rollout_state = await self.run_judger(rollout_state)
        return rollout_state
