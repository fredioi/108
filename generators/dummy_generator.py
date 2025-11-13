"""
Dummy Generator for Testing

Returns mock data without requiring API keys or GPU.
Used for testing protocol communication.
"""

from typing import Dict
from .base import StoryGenerator


class DummyGenerator(StoryGenerator):
    """
    Dummy generator that returns mock data.
    Used for testing protocol without real AI.
    """

    def __init__(self, config: Dict = None):
        """Initialize dummy generator."""
        self.config = config or {}

    async def generate(self, input_data: Dict) -> Dict:
        """
        Return mock data based on task type.

        Args:
            input_data: Input with task_type field

        Returns:
            Mock response matching expected format
        """
        task_type = input_data.get("task_type", "blueprint")

        if task_type == "blueprint":
            return {
                "title": "测试故事",
                "genre": "科幻",
                "setting": "未来世界",
                "core_conflict": "人机对抗",
                "themes": ["AI觉醒", "人性"],
                "tone": "严肃",
                "target_audience": "成年人"
            }

        elif task_type == "characters":
            return {
                "characters": [
                    {"name": f"角色{i}", "role": "主角" if i == 1 else "配角"}
                    for i in range(1, 6)
                ]
            }

        elif task_type == "story_arc":
            return {
                "title": "测试故事线",
                "description": "这是一个测试故事",
                "chapters": [{"id": i, "title": f"第{i}章"} for i in range(1, 13)],
                "arcs": [],
                "themes": ["测试"],
                "hooks": []
            }

        elif task_type == "chapters":
            return {
                "chapters": [
                    {
                        "id": i,
                        "title": f"第{i}章",
                        "content": "测试内容",
                        "choices": []
                    }
                    for i in input_data.get("chapter_ids", [1])
                ]
            }

        return {}

    async def health_check(self) -> bool:
        """Always healthy."""
        return True

    def get_mode(self) -> str:
        """Return mode name."""
        return "dummy"

    def get_model_info(self) -> Dict:
        """Return dummy model info."""
        return {
            "mode": "dummy",
            "model": "mock-generator",
            "provider": "test"
        }
