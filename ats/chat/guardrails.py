import json
import re

from langchain_core.messages import HumanMessage

from ats.chat.prompts import guardrail_prompt
from ats.logger import get_logger
from ats.chat.utils import words_for_guardrails
from ats.chat.utils import convert_langchain_messages_to_openai

logger = get_logger("guardrails")

# who need a library when you can reinvent it
# but actually it was unnecessary to add another library with it's own flow for llms
# this class can be easily modified in any way after poc stage
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

    def _filter_messages(self, messages: list[dict]) -> list[dict]:
        return [x for x in messages if x["role"] in ["assistant", "user"]]
    
    def _prepare_messages(self, messages) -> list[dict]:
        messages = convert_langchain_messages_to_openai(messages)
        messages = self._filter_messages(messages)
        return messages

    def rail(self, messages) -> bool:
        messages = self._prepare_messages(messages)
        flag = self.check_messages_regexp(messages)
        logger.debug("Regexp guardrail check: {}".format(flag))
        if not flag and self.fallback_to_llm:
            flag = self.check_messages_llm(messages)
            logger.debug("LLM guardrail check: {}".format(flag))
        return flag

    # stupid and simple, to reduce number of queries that go to LLM and hence reduce latency and price
    # it would be nice to replace it with BERT-ish model, but it's out of scope for now
    # stemming / lemmatization would be nice too, but I don't have time for this
    def check_messages_regexp(self, messages: list[dict]) -> bool:
        messages = " ".join([x["content"] for x in messages[-1:] if x["role"] == "user"])  # is 1 msg too strict?
        logger.debug("regexp check messages: {}".format(messages))
        if self.regexp.search(messages) is not None:
            return True
        else:
            return False

    def check_messages_llm(self, messages: list[dict]) -> bool:
        messages = json.dumps(messages[-5:], ensure_ascii=False)
        logger.debug("llm check messages: {}".format(messages))
        res = self.llm.invoke([HumanMessage(self.llm_prompt + messages)])
        return res["flag"]
