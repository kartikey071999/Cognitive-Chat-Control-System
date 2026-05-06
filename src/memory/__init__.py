from collections import deque

from src.llm.cloud import cloud_chat

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SUMMARIZE_PROMPT = (
    "Summarize this conversation into a brief context paragraph. "
    "Keep key facts, user preferences, and decisions. Drop filler. "
    "Reply with ONLY the summary, nothing else."
)


class Memory:
    def __init__(self, max_turns: int = 5):
        self._history = deque(maxlen=max_turns * 2)
        self._context: str = ""  # running summarized context
        self.public: list[dict] = []
        self.model = MODEL
        self.system_prompt = SUMMARIZE_PROMPT

    async def add(self, role: str, content: str):
        msg = {"role": role, "content": content}
        self._history.append(msg)
        self.public = list(self._history)

        # Summarize when history is full
        if len(self._history) == self._history.maxlen:
            await self._summarize()

    async def add_exchange(self, user_msg: str, ai_msg: str):
        await self.add("user", user_msg)
        await self.add("assistant", ai_msg)

    async def _summarize(self):
        # Build text from current history + existing context
        parts = []
        if self._context:
            parts.append(f"Previous context: {self._context}")
        for msg in self._history:
            parts.append(f"{msg['role']}: {msg['content']}")
        text = "\n".join(parts)

        messages = [{"role": "user", "content": text}]
        self._context = cloud_chat(
            messages, model=self.model, system_prompt=self.system_prompt, stream=False
        )
        # Keep only last 2 messages after summarizing
        recent = list(self._history)[-2:]
        self._history.clear()
        for msg in recent:
            self._history.append(msg)
        self.public = list(self._history)

    def get(self) -> list[dict]:
        msgs = []
        if self._context:
            msgs.append(
                {"role": "system", "content": f"Memory context: {self._context}"}
            )
        msgs.extend(self._history)
        return msgs
