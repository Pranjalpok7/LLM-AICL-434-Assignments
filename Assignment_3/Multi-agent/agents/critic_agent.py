from agents.base_agent import BaseAgent
from utils.message_system import Message

class CriticAgent(BaseAgent):
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="CriticAgent",
            agent_type="critic",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )

    def _handle_task_request(self, message: Message):
        content = message.content
        draft_key = content.get("draft_key")

        print(f"[{self.name}] Received request to critique draft from key: '{draft_key}'")
        draft = self.retrieve_from_memory(draft_key)

        if not draft:
            print(f"[{self.name}] Could not find draft in memory. Aborting.")
            return

        prompt = f"Please provide a constructive, high-level critique of the following article draft. Focus on argument, structure, and engagement, not small grammar fixes.\n\nDraft:\n{draft}"

        critique = self.generate_llm_response(prompt)

        key = "critique_of_v1"
        self.store_in_memory(key, critique)
        print(f"[{self.name}] Stored critique in shared memory with key: '{key}'")
