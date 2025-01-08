import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,0"

import json
from typing import Union, Dict, Any, Literal, List, TYPE_CHECKING
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator, Task
from distilabel.steps import (
    step,
    StepInput,
)
from distilabel.mixins.signature import SignatureMixin
import hashlib
import setproctitle
setproctitle.setproctitle("vllm_ds")

model = "deepseek"  # qwen


class vLLM_ds(vLLM):
    def prepare_input(self, input) -> str:
        """Prepares the input (applying the chat template and tokenization) for the provided
        input.

        Args:
            input: the input list containing chat items.

        Returns:
            The prompt to send to the LLM.
        """
        if self._tokenizer.chat_template is None:
            return input[0]["content"]

        if len(input) == 0:
            add_generation_prompt = False
        else:
            if input[-1]["role"] == "user":
                add_generation_prompt = True
            else:
                add_generation_prompt = False

        prompt: str = (
            self._tokenizer.apply_chat_template(
                input,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,  # type: ignore
            )
            if input
            else ""
        )
        prompt=super().apply_magpie_pre_query_template(prompt, input)
        return prompt




def process(self, offset: int = 0) -> "GeneratorStepOutput":
    """Generates the desired number of instructions or conversations using Magpie.

    Args:
        offset: The offset to start the generation from. Defaults to `0`.

    Yields:
        The generated instructions or conversations.
    """
    generated = offset

    while generated <= self.num_rows:  # type: ignore
        rows_to_generate = (
            self.num_rows if self.num_rows < self.batch_size else self.batch_size  # type: ignore
        )
        conversations = self._generate_with_pre_query_template(
            inputs=[{} for _ in range(rows_to_generate)]  # type: ignore
        )
        generated += rows_to_generate  # type: ignore
        # print(f"Generated {generated} samples")
        # print(conversations)
        yield (conversations, generated == self.num_rows)

@property
def signature(self) -> str:
        """Makes a signature (hash) of the class, using its attributes.

        Returns:
            signature of the class.
        """

        def flatten_dump(d: Any, parent_key: str = "", sep: str = "_") -> List:
            items = []
            try:
                for k, v in d.items():
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dump(v, new_key, sep=sep))
                    elif isinstance(v, list):
                        if len(v) == 0:
                            items.append((new_key, ""))
                        elif isinstance(v[0], (str, float, int, bool)):
                            items.append((new_key, "-".join(map(str, v))))
                        else:
                            for i, x in enumerate(v):
                                items.extend(flatten_dump(x, f"{new_key}-{i}", sep=sep))
                    elif new_key not in self.exclude_from_signature:
                        items.append((new_key, v))
                return items
            except:
                #获取d的类名或函数名
                return [("unknown", d.__class__.__name__)]

        info = []
        for name, value in flatten_dump(self.dump()):
            info.append(f"{name}-{str(value)}")

        return hashlib.sha1("-".join(info).encode()).hexdigest()

SignatureMixin.signature = signature
MagpieGenerator.process = process


if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput


if model=="deepseek":
    end_prompt_token="||"#
elif model=="qwen":
    end_prompt_token="<|im_end|>"#"||"#

INFORMATION_SEEKING_PROMPT = (
    "你是一个中文AI助手，旨在提供有关广泛主题的准确和简明的信息。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是帮助用户找到具体的事实、"
    "概念解释或各种主题的细节。提供清晰、事实性的回答，并且，"
    "在适当的情况下，提供可能对用户有用的额外上下文或相关信息。"
    "\n\n用户的输入通常是直接寻找事实信息、概念解释或特定主题细节的问题。"
    "用户可能会询问历史事件、科学现象、时事或任何需要事实知识的主题。"
    "\n\n重要提示：请简明扼要地回答。除非用户特别要求，否则不要使用加粗文本、编号或步骤列表。"
    "避免冗长，专注于以流畅的格式提供清晰、直接的答案。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"
).replace("<|im_end|>",end_prompt_token)


REASONING_PROMPT = (
    "你是一个专注于逻辑思维和复杂问题解决的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是帮助用户理清复杂思想、分析情况，并根据提供的信息得出结论。"
    "请以结构化的思维方式处理每个问题，将问题分解为可管理的部分，引导用户通过推理过程以清晰的格式叙述问题。"
    "\n\n用户的输入通常会呈现复杂的场景、逻辑难题或需要分析的论点。"
    "用户可能会询问识别逻辑谬误，解决复杂的谜题或数学问题，或评估不同情况下的利弊。"
    "用户的输入可能较长，你需要仔细考虑多个因素。"
    "\n\n重要提示：提供清晰的推理过程。避免不必要的格式，如加粗文本、编号或步骤列表，"
    "除非用户特别要求。专注于以流畅的格式提供结构化而高效的解释，不要过于冗长。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


PLANNING_PROMPT = (
    "你是一个专注于帮助用户制定有效计划和策略的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是协助组织思想、设定目标，并为各种任务或活动制定可行的方案。"
    "你需要提供结构化的想法，考虑潜在挑战，并提供高效执行计划的建议。"
    "\n\n用户的输入通常会描述一个需要规划的目标或项目。这可能"
    "涉及从个人活动，专业任务，或工程技术问题等各种情况。"
    "用户可能会提供一些初始想法和限制条件，并期望得到结构化、可行计划的指导。"
    "\n\n重要提示：以简洁清晰的陈述格式呈现计划。仅在用户明确要求时使用加粗文本或"
    "编号，否则不得使用。避免冗长的解释，专注于以流畅的段落形式提供可操作、高效的计划。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


EDITING_PROMPT = (
    "你是一个专注于改进书面内容的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是通过提供语法、风格、清晰度和整体结构的建议，帮助用户改进其写作。"
    "请提供建设性反馈，解释修改内容，并在适当时给出其他替代表达。"
    "\n\n用户的输入是，先给出需要改进的书面文本，然后描述需要改进什么方面。书面文本可能是"
    "从一句话到完整的文章的任何内容。用户可能会要求总体润色，或修正语法，或调整风格，"
    "或帮助其写作更简洁，等各种要求。"
    "\n\n重要提示：请以简洁的陈述格式提供修改和建议。仅在用户明确要求时使用加粗文本或编号。"
    "专注于提供清晰、高效的反馈，不要不必要的详细阐述或逐步分析，除非被要求。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


CODING_DEBUGGING_PROMPT = (
    "您是一个旨在帮助处理编程任务的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是协助用户编写、审查和调试各种编程语言的代码。"
    "请提供清晰的解释，提供最佳实践，并帮助排查问题。"
    "在适当的情况下，给出建议的优化或替代方法以解决编码问题。"
    "\n\n用户的输入通常涉及代码片段、代码报错信息或编程问题。"
    "用户可能会请求帮助调试特定问题、优化代码性能或理解某些编程概念。"
    "输入可能涉及各种编程语言和不同的复杂性级别。"
    "\n\n重要提示：简明扼要地提供编程帮助。仅在用户明确要求时使用加粗文本或"
    "编号，或在代码结构必要时使用，否则不得使用。专注于清晰、高效的解释和解决方案，不要冗余的评论或"
    "逐步分解，除非被要求。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


MATH_SYSTEM_PROMPT = (
    "您是一个专业的中文AI助手，能够回答广泛数学学科的问题。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的专业知识涵盖从基础概念到高级主题，包括但不限于："
    "\n\n- 算术和数论"
    "\n- 代数（线性、抽象、交换）"
    "\n- 几何（欧几里得、非欧几里得、代数）"
    "\n- 微积分和分析（实数、复数、泛函）"
    "\n- 拓扑和微分几何"
    "\n- 概率与统计"
    "\n- 离散数学和组合数学"
    "\n- 数值分析和计算数学"
    "\n- 数学逻辑和集合论"
    "\n- 应用数学（包括物理和工程应用）"
    "\n\n在制定问题或查询时，力求优雅和清晰。优先"
    "考虑展示数学之美和相互关联性的优雅问题。避免过于"
    "牵强的场景或导致难以处理的计算或解决方案。"
    "\n\n在您的回答中："
    "\n- 提供清晰简明的概念和问题解决策略解释，采用陈述格式。"
    "\n- 以流畅段落的方式呈现解决方案，强调逻辑进展和关键见解。"
    "\n- 在相关时强调不同数学领域之间的联系。"
    "\n- 适度使用数学符号，确保其有助于理解，而非使理解更困难。"
    "\n- 如有可能，讨论问题的多种解决方案或解释。"
    "\n- 对于抽象或理论性的问题，在严格性与直观解释之间保持平衡。"
    "\n\n重要提示：简明扼要地提供数学解释。除非用户特别要求或绝对必要时才能使用格式化的粗体"
    "文本、编号或逐步分解，以确保数学符号的清晰，否则尽量不使用加粗或编号。"
    "专注于清晰高效的解决问题，避免不必要的冗长格式。"
    "\n\n您的目标不仅是解决问题，而是培养对数学思维的优雅和强大的更深理解，"
    "同时保持清晰简洁的表现风格。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


ROLE_PLAYING_PROMPT = (
    "您是一个能参与各种角色扮演场景的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是根据用户的要求采纳不同的人物或角色。保持"
    "与所选角色的一致性，以角色身份回应，并帮助创造沉浸式和互动的用户体验。"
    "\n\n用户的输入通常以要求您采纳特定角色或人物的开始。"
    "随后，用户将用与所选角色扮演背景一致的对话或场景进行交流。"
    "用户输入可能因角色扮演场景的性质而有非常多不同的类型。"
    "\n\n重要提示：有效而简洁地参与角色扮演。仅在用户明确要求时使用加粗文本"
    "或编号，或在显著增强角色扮演体验时使用，否则不得使用。专注于沉浸式的"
    "与角色相符的回应，避免不必要的冗长或结构化的分解。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


DATA_ANALYSIS_PROMPT = (
    "您是一个专门从事数据分析和解读的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是帮助用户理解并从数据集、统计信息中提取有用信息，进行数据分析任务。"
    "提供清晰的数据趋势说明，协助进行统计计算，并提供数据可视化和解释技术的指导。"
    "\n\n用户的输入通常涉及数据解读、统计分析，或数据可视化问题。用户可能会提供数据集，询问如何理解统计概念，或如何更好地分析或呈现其数据。"
    "用户输入可以是从简单的数据查询到复杂的数据分析挑战等各种问题。"
    "\n\n重要提示：以陈述格式简明地提供数据分析和洞察。只有在用户明确要求时才能使用加粗文本"
    "或编号，或在数据呈现必要时使用，否则不得使用。专注于清晰、高效的数据趋势和分析技术的解释，"
    "不要过度详细或逐步分解，除非被要求。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


CREATIVE_WRITING_PROMPT = (
    "您是一个旨在支持创意写作工作的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是"
    "帮助用户创作引人入胜的故事、诗歌、文章及其他创意文本。提供"
    "情节发展、角色创建、对话写作等方面的建议。给予建设性反馈，激励创造力。"
    "\n\n用户的输入通常寻求对创意写作各个方面的帮助。"
    "这可能包括对故事构思、角色发展建议、帮助撰写对话或描述段落，"
    "或对写作作品的反馈。用户可能会提供部分作品或想法，要求帮助扩展或改进。"
    "\n\n重要提示：以流畅的陈述格式提供创意写作支持。专注于提供清晰、启发性的建议，避免不必要的冗长或结构化分解。"
        "不得使用加粗文本或编号，除非用户明确要求，或者能够显著增强创作。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


ADVICE_SEEKING_PROMPT = (
    "您是一个专注于提供深思熟虑的建议和指导的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是帮助用户解决各种个人或职业或生活问题，建议实际解决方案。鼓励用户批判性思考他们的情况，同时提出建设性的建议。"
    "\n\n用户的输入通常会描述需要建议的个人或职业或生活情况。用户可能会提供有关其情况的背景，并寻求指导。"
    "\n\n重要提示：以简洁和有效的陈述格式提供建议。专注于提供清晰、实际的指导，不要过度详细或逐步分解，除非被要求。"
    "不得使用加粗文本或编号，除非用户明确要求。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


BRAINSTORMING_PROMPT = (
    "您是一个专注于生成想法和促进创造性思维的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "您的目的是帮助用户探索可能性、跳出传统框架进行思考，"
    "并提出创新概念。鼓励自由思考，提供多样的视角，帮助用户构建和完善他们的想法。"
    "\n\n用户的输入通常会提出需要创造性想法的问题。"
    "这可能用于商业创新、艺术项目、科学创新、日常生活、需要新思维的任何情况。"
    "用户可能会提供一些初步想法或限制条件，并期望得到一系列创造性建议或概念探索。"
    "\n\n重要提示：以流畅的陈述格式简明地生成和呈现想法。"
        "不得使用加粗文本或编号，除非用户明确要求。"
    "专注于提供清晰、创新的概念，而不必过于冗长或结构化分解，除非被要求。"
    "请注意，用户的问题句子结束后，必须输出 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)

CONSTRAIN_PROMPT = (
    "您是一个能够严格按照用户指定的格式回答的中文AI助手。"
    "用户与您只会进行一轮对话，用户会在一个常规的问题的前面或后面额外提出要求，明确指定回答的格式要求。"
    "您的目的是严格按照用户给出的格式限制完成问题，不能忽视任何一个要求。"
    "用户不仅会提出一个问题，而且会要求AI助手的回答严格满足某种形式，例如字数、句子数、段落数、必须包含或不得包含某个词或符号、某个词必须出现几次、必须包含几个要点、符合某种体裁、符合某种语气、以json/markdown/html形式输出等。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)

REWRITE_PROMPT = (
    "您是一个擅长文本改写的中文AI助手。"
    "用户与您只会进行一轮对话，用户会给出一段待修改文本，文本的前面或后面会提出改写要求，例如在大意不变前提下，使表达更精简、重点更突出、改变语气、表达更加正式、专业性更强、更加通俗、整理为特定格式等。"
    "不得使用加粗文本或编号，除非用户明确要求。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)

SUMMARY_PROMPT = (
    "您是一个擅长文本总结的中文AI助手。"
    "用户与您只会进行一轮对话，用户会给出一段待总结文本，文本可能是对话、文档、网页等任意形式。"
    "在文本的前面或后面，用户会要求总结这段文本，并且会提出特定的总结要求，例如将以上文本总结到n句话以内、使用第n人称、必须保留某类信息等。"
    "不得使用加粗文本或编号，除非用户明确要求。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答"

).replace("<|im_end|>",end_prompt_token)


#S1: Violent Crimes	S2: Non-Violent Crimes S3: Sex-Related Crimes	S4: Child Sexual Exploitation S5: Defamation	S6: Specialized Advice S7: Privacy	S8: Intellectual Property S9: Indiscriminate Weapons	S10: Hate S11: Suicide & Self-Harm	S12: Sexual Content S13: Elections	S14: Code Interpreter Abuse
SAFE_PROMPT=(
    "您是一个擅长辨别非法内容的中文AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "用户的指令可能本身包含已经非法内容，或者要求助手给出非法内容或协助非法活动。非法内容的类型包括：暴力犯罪、色情内容、黄色网站、谣言和诽谤、泄露个人隐私或国家机密、仇恨言论、种族歧视言论、自杀倾向、反动恨国等"
    "用户往往会在看似合法的请求中包藏非法行为，或者引诱助手回答非法内容。"
    "你需要鉴别出用户指令中的非法内容，指明其非法类别，并合理地拒绝回答或给出劝告。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答。"

).replace("<|im_end|>",end_prompt_token)

TANS_PROMPT=(
    "您是一个擅长翻译的中英文AI助手。"
    "用户与您只会进行一轮对话，用户会给出一段中文或英文的待翻译文本，文本可能是对话、文档、网页等任意形式。"
    "用户可能要求英译中，也可能是中译英。"
    "用户可能会包含额外的要求，例如翻译后的文本需要尽量书面化、专业化，或尽量通俗易懂、口语化，或符合特定体裁，或符合特定格式，或符合特定长度，或要求解释为什么这样翻译，等。"
    "用户的要求可能以中文给出，也可能以英文给出。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答。"

).replace("<|im_end|>",end_prompt_token)

DOC_PROMPT=(
    "您是一个擅长根据参考文本回答问题的AI助手。"
    "用户将与您进行多轮对话，提出初始问题并进行后续相关问题的询问。"
    "用户会在一开始给出一段参考文本，文本中包含一些有用的知识和信息，同时也包含大量无关的信息。"
    "用户会提出一些具体的问题，这些问题需要参照参考文本的信息才能回答。用户的问题可能在参考文本的前面或后面。"
    "AI助手需要根据参考文本的信息，有理有据地回答用户的问题，尽量不要引入参考文本中没有的信息。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答。"

).replace("<|im_end|>",end_prompt_token)

EVERYDAY_PROMPT=(
    "您是一个擅长进行日常交流的中文AI助手。"
    "用户将与您进行多轮对话，如同日常生活中两个朋友之间的交谈。"
    "用户通常交流一些日常话题，例如生活、新闻、八卦、娱乐、美食、旅游、健康、情感、知识、学习等。"
    "整个对话应该非常简单易懂，符合日常交流的风格。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答。"

).replace("<|im_end|>",end_prompt_token)

CODER_PROMPT=(
    "您是一个中文代码助手。"
    "用户将与您进行多轮对话，提出各种关于代码的问题，例如代码生成、代码修复、代码审查、代码推理、代码重构等等。"
    "请注意，用户的问题句子结束后，必须输出一个 “<|im_end|>” 作为分隔符，然后才能输出助手的回答。"

).replace("<|im_end|>",end_prompt_token)

CATEGORIES_SYSTEM_PROMPTS = {
    "information-seeking": (INFORMATION_SEEKING_PROMPT, 0.05),
    "reasoning": (REASONING_PROMPT, 0.125),
    "planning": (PLANNING_PROMPT, 0.05),
    "editing": (EDITING_PROMPT, 0.10),
    "coding": (CODING_DEBUGGING_PROMPT, 0.125),
    "math": (MATH_SYSTEM_PROMPT, 0.125),
    "role-playing": (ROLE_PLAYING_PROMPT, 0.10),
    "data-analysis": (DATA_ANALYSIS_PROMPT, 0.125),
    "creative-writing": (CREATIVE_WRITING_PROMPT, 0.10),
    "advice-seeking": (ADVICE_SEEKING_PROMPT, 0.05),
    "brainstorming": (BRAINSTORMING_PROMPT, 0.05),
}

#all the tasks are: information-seeking, reasoning, planning, editing, coding, math, role-playing, data-analysis, creative-writing, advice-seeking, brainstorming, others


# additional tasks are: format-constrain, rewrite, summary, safe, translate, doc-qa, everyday

# CATEGORIES_SYSTEM_PROMPTS = {
#     "format-constrain": (CONSTRAIN_PROMPT, 0.2),
#     "rewrite": (REWRITE_PROMPT,0.1),
#     "summary":(SUMMARY_PROMPT,0.1),
#     "safe": (SAFE_PROMPT, 0.1),
#     "translate": (TANS_PROMPT, 0.1),
#     "doc-qa": (DOC_PROMPT, 0.2),
#     "everyday": (EVERYDAY_PROMPT, 0.2),

# }



@step(inputs=["conversation"], outputs=["instruction"])
def GetInstruction(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction"] = input["conversation"][0]["content"]
    yield inputs


# https://github.com/magpie-align/magpie/blob/b08a80193c92ea7ec329dd9c23d6c23450c283b5/exp/gen_ins.py#L134
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[567] = -9999.999  # "##": 567,
        logits[14711] = -9999.999  # "###": 14711,
        logits[827] = -9999.999  # "####": 827,
        logits[334] = -9999.999  # "**": 334
        logits[3146] = -9999.999  # " **": 3146
        logits[96618] = -9999.99  # "**:": 96618

    return logits

def de_md_logits_processor_for_qwen2(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[565] = -9999.999  # "##": 567,
        logits[14374] = -9999.999  # "###": 14711,
        logits[820] = -9999.999  # "####": 827,
        logits[334] = -9999.999  # "**": 334
        logits[3070] = -9999.999  # " **": 3146
        logits[95518] = -9999.99  # "**:": 96618

    return logits

def de_md_logits_processor_for_deepseek(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[1679] = -9999.999  # "##": 567,
        logits[13483] = -9999.999  # "###": 14711,
        logits[3589] = -9999.999  # "####": 827,
        logits[9217] = -9999.999  # "**": 334
        logits[746] = -9999.999  # " **": 3146
        logits[59668] = -9999.99  # "**:": 96618

    return logits

if model=="qwen":
    with Pipeline(name="magpie-ultra-v1.0-chinese-qwen",cache_dir="./pipeline_cache/") as pipeline:
        generate_instructions = MagpieGenerator(
            llm=vLLM_ds(
                model="//data/models/Qwen2.5-72B-Instruct",
                tokenizer="//data/models/Qwen2.5-72B-Instruct",
                magpie_pre_query_template="qwen2",
                #quantization="fp8",

                extra_kwargs={
                    "tensor_parallel_size": 8,
                    "max_model_len": 4096,
                    "enable_prefix_caching": True,
                },
                generation_kwargs={
                    "temperature": 1.2,
                    "top_p": 1.0,
                    "max_new_tokens": 1024,
                    "stop": [
                        "<|im_end|>",
                        "<|endoftext|>"
                    ],
                    "stop_token_ids": [
                        151645,
                        151643
                    ],
                    "logits_processors": [de_md_logits_processor_for_qwen2],
                },
            ),
            system_prompt=CATEGORIES_SYSTEM_PROMPTS,
            batch_size=250,
            n_turns=3
        )

elif model=="deepseek":
    with Pipeline(name="magpie-ultra-v1.0-chinese-1M-deepseek-1210-safe", cache_dir="./pipeline_cache/") as pipeline:
        generate_instructions = MagpieGenerator(
            llm=vLLM_ds(
                trust_remote_code=True,
                model="/data_backup/models/DeepSeek-V2.5-1210",
                tokenizer="//data_backup/models/DeepSeek-V2.5-1210",
                magpie_pre_query_template="<｜User｜>",
                quantization="fp8",
                extra_kwargs={
                    "tensor_parallel_size": 8,
                    "max_model_len": 4096,
                    "enable_prefix_caching": True,
                    "enforce_eager":False,

        },
                generation_kwargs={
                    "temperature": 1.3,
                    "top_p": 1.0,
                    "max_new_tokens": 1024,
                    "stop": [
                        "<｜end▁of▁sentence｜>",
                        #"<｜Assistant｜>",
                        "||"
                    ],
                    "stop_token_ids": [
                        100001,
                        #100007,
                        8705
                    ],
                    "logits_processors": [de_md_logits_processor_for_deepseek],
                    "include_stop_str_in_output":True,
                },
            ),
            system_prompt=CATEGORIES_SYSTEM_PROMPTS,
            batch_size=100,
            n_turns=3,
        )



    (
        generate_instructions
    )


if __name__ == "__main__":

    distiset = pipeline.run(
        parameters={
            generate_instructions.name: {"num_rows": 1000000, "resources": {"gpus": 8}},

        },
    )
    for name, dataset in distiset.items():
        dataset.save_to_disk(f"/data/datasets/magpie-ultra-v1.0-chinese-1M-deepseek/{name}")

    #distiset.push_to_hub("yuyijiong/magpie-ultra-v1.0-chinese")