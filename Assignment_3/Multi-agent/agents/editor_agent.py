from agents.base_agent import BaseAgent
from utils.message_system import Message

class EditorAgent(BaseAgent):
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="EditorAgent",
            agent_type="editor",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )

    def _handle_task_request(self, message: Message):
        content = message.content
        draft_key = content.get("draft_key")
        critique_key = content.get("critique_key")

        print(f"[{self.name}] Received request to edit draft using critique.")
        draft = self.retrieve_from_memory(draft_key)
        critique = self.retrieve_from_memory(critique_key)

        if not draft or not critique:
            print(f"[{self.name}] Missing draft or critique in memory. Aborting.")
            return

        prompt = f"Please revise the following article draft based on the provided critique to create a polished final version.\n\n---\nOriginal Draft:\n{draft}\n\n---\nCritique:\n{critique}\n\n---\nRevised Article:"

        final_article = self.generate_llm_response(prompt)

        key = "final_article"
        self.store_in_memory(key, final_article)
        print(f"[{self.name}] Stored final article in shared memory with key: '{key}'")
