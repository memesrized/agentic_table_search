import json
import re

from langchain_core.messages import HumanMessage

from ats.chat.prompts import guardrail_prompt
from ats.logger import get_logger
from ats.chat.utils import words_for_guardrails

logger = get_logger("guardrails")


class Guardrails:
    """...kind of"""

    def __init__(self, fallback_to_llm: bool = False, llm=None):
        escaped_words = "|".join(re.escape(word) for word in words_for_guardrails)
        self.regexp = re.compile(
            r"(?:^|(?<=\W))(" + escaped_words + r")(?=\W|$)", re.IGNORECASE | re.UNICODE
        )
        self.fallback_to_llm = fallback_to_llm
        self.llm = llm
        self.llm_prompt = guardrail_prompt

    def rail(self, messages: list[dict]) -> bool:
        flag = self.check_messages_regexp(messages[-3:])
        logger.debug("Regexp guardrail check: {}".format(flag))
        if not flag and self.fallback_to_llm:
            flag = self.check_messages_llm(messages)
            logger.debug("LLM guardrail check: {}".format(flag))
        return flag

    def check_messages_regexp(self, messages: list[dict]) -> bool:
        messages = " ".join([x["content"] for x in messages[-5:] if x["role"] == "user"])
        logger.debug("regexp check messages: {}".format(messages))
        if self.regexp.search(messages) is not None:
            return True
        else:
            return False

    def check_messages_llm(self, messages: list[dict]) -> bool:
        messages = json.dumps(messages, ensure_ascii=False)
        logger.debug("llm check messages: {}".format(messages))
        res = self.llm.invoke([HumanMessage(self.llm_prompt + messages)])
        return res["flag"]
