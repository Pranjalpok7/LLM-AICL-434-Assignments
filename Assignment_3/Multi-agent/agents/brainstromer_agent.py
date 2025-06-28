from agents.base_agent import BaseAgent
from utils.message_system import Message

class BrainstormerAgent(BaseAgent):
    def __init__(self, message_bus, shared_memory, llm_interface):
        super().__init__(
            name="BrainstormerAgent",
            agent_type="brainstormer",
            message_bus=message_bus,
            shared_memory=shared_memory,
            llm_interface=llm_interface
        )

    def _handle_task_request(self, message: Message):
        content = message.content
        topic = content.get("topic")

        print(f"[{self.name}] Received brainstorming request for topic: '{topic}'")

        prompt = f"Please brainstorm a structured list of key points for an article on the topic: '{topic}'."

        ideas = self.generate_llm_response(prompt)

        key = "brainstorm_results"
        self.store_in_memory(key, ideas)
        print(f"[{self.name}] Stored brainstormed ideas in shared memory with key: '{key}'")
