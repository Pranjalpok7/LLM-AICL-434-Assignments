from agents.base_agent import BaseAgent
from utils.message_system import Message

class WriterAgent(BaseAgent):
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="WriterAgent",
            agent_type="writer",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )

    def _handle_task_request(self, message: Message):
        content = message.content
        topic = content.get("topic")
        ideas_key = content.get("ideas_key")

        print(f"[{self.name}] Received writing request for topic: '{topic}'")
        ideas = self.retrieve_from_memory(ideas_key)

        if not ideas:
            print(f"[{self.name}] Could not find ideas in memory. Aborting.")
            return

        prompt = f"Write a draft article on the topic '{topic}', using the following brainstormed points as a guide:\n\n{ideas}"

        draft = self.generate_llm_response(prompt)

        key = "draft_v1"
        self.store_in_memory(key, draft)
        print(f"[{self.name}] Stored draft in shared memory with key: '{key}'")
