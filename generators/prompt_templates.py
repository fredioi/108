"""
Prompt Template System

Allows miners to customize generation prompts without modifying code.
Supports:
- Template files (YAML, JSON, or .txt)
- Variable interpolation
- Multiple task types (blueprint, characters, story, etc.)
- Custom templates per model
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from string import Template


class PromptTemplateManager:
    """
    Manages prompt templates for story generation.

    Templates can be:
    1. Loaded from files (YAML/JSON/TXT)
    2. Defined inline in code
    3. Customized per task type

    Example YAML template:
        blueprint:
          system: "You are a creative story writer..."
          user: |
            Create a story blueprint based on:
            ${user_input}

    Example usage:
        manager = PromptTemplateManager()
        prompt = manager.render("blueprint", {"user_input": "space adventure"})
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template manager.

        Args:
            template_dir: Directory containing template files (default: config/prompts/)
        """
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "prompts"
            )

        self.template_dir = template_dir
        self.templates: Dict[str, Dict] = {}

        # Load default templates (built-in)
        self._load_default_templates()

        # Load custom templates from files (if directory exists)
        if os.path.exists(self.template_dir):
            self._load_template_files()

    def _load_default_templates(self):
        """Load built-in default templates."""
        self.templates = {
            "blueprint": {
                "system": "You are a creative story writer for an interactive story game.",
                "user": Template("""Create a story blueprint based on the following user input:

User Request: ${user_input}

Generate a comprehensive story blueprint with:
1. Title
2. Genre
3. Setting
4. Core Conflict
5. Themes (3-5)
6. Tone
7. Target Audience

Output as a single JSON object (NOT an array). Example format:
{
  "title": "...",
  "genre": "...",
  "setting": "...",
  "core_conflict": "...",
  "themes": [...],
  "tone": "...",
  "target_audience": "..."
}""")
            },

            "characters": {
                "system": "You are a character designer for interactive stories.",
                "user": Template("""Create 5 diverse characters for the following story:

Story: ${user_input}
${blueprint_context}

For each character, provide:
- id (unique identifier)
- name
- archetype (Hero, Mentor, Shadow, etc.)
- background
- motivation
- skills (3-5)
- personality_traits (3-5)
- relationships (to other characters)

Output as a JSON object with this structure:
{
  "characters": [
    {...character 1...},
    {...character 2...},
    ...
  ]
}""")
            },

            "story_arc": {
                "system": "You are a narrative designer for interactive stories.",
                "user": Template("""Create a story arc with 5-7 chapters based on:

Story: ${user_input}
${blueprint_context}
${characters_context}

For each chapter, provide:
- chapter_number
- title
- summary
- key_events (3-5)
- character_focus
- chapter_goal

Output as a JSON object with this structure:
{
  "title": "...",
  "description": "...",
  "chapters": [...array of chapter objects...],
  "arcs": [...story arcs...],
  "themes": [...themes...],
  "hooks": [...plot hooks...]
}""")
            },

            "chapters": {
                "system": "You are a writer creating immersive story chapters.",
                "user": Template("""Generate detailed chapter content based on:

Story Arc: ${story_arc}
Characters: ${characters_context}
Blueprint: ${blueprint_context}
Chapter IDs: ${chapter_ids}

For each requested chapter, write engaging content (500-1000 words) that:
1. Continues the narrative
2. Develops characters
3. Includes meaningful choices for the player
4. Advances the plot

Output as a JSON object with this structure:
{
  "chapters": [
    {
      "chapter_number": 1,
      "title": "...",
      "content": "...chapter text...",
      "choices": [...player choices...]
    },
    ...
  ]
}""")
            },

            "chapter_content": {
                "system": "You are a writer creating immersive story chapters.",
                "user": Template("""Write chapter content for:

Chapter: ${chapter_number} - ${chapter_title}
${chapter_summary}

Story Context:
${story_context}

Characters:
${characters_context}

Write engaging, immersive content (500-1000 words) that:
1. Continues the narrative
2. Develops characters
3. Includes meaningful choices for the player
4. Advances the plot

Include 2-3 branching choices at the end.""")
            },

            "generic": {
                "system": "You are a creative story writer for an interactive story game.",
                "user": Template("""Generate story content based on:

${user_input}

${context}""")
            }
        }

    def _load_template_files(self):
        """Load custom templates from files."""
        if not os.path.exists(self.template_dir):
            return

        for filename in os.listdir(self.template_dir):
            filepath = os.path.join(self.template_dir, filename)

            # Skip directories and non-template files
            if not os.path.isfile(filepath):
                continue

            # Determine format and load
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self._load_yaml_template(filepath)
            elif filename.endswith('.json'):
                self._load_json_template(filepath)
            elif filename.endswith('.txt'):
                self._load_txt_template(filepath)

    def _load_yaml_template(self, filepath: str):
        """Load template from YAML file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Merge with existing templates
            for task_type, template_data in data.items():
                self.templates[task_type] = template_data
                # Convert user template to Template object
                if 'user' in template_data and isinstance(template_data['user'], str):
                    template_data['user'] = Template(template_data['user'])

            print(f"✅ Loaded templates from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"⚠️  Failed to load template {filepath}: {e}")

    def _load_json_template(self, filepath: str):
        """Load template from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Merge with existing templates
            for task_type, template_data in data.items():
                self.templates[task_type] = template_data
                # Convert user template to Template object
                if 'user' in template_data and isinstance(template_data['user'], str):
                    template_data['user'] = Template(template_data['user'])

            print(f"✅ Loaded templates from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"⚠️  Failed to load template {filepath}: {e}")

    def _load_txt_template(self, filepath: str):
        """Load template from text file."""
        try:
            # File name (without extension) is the task type
            task_type = os.path.splitext(os.path.basename(filepath))[0]

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple text file becomes the user prompt
            self.templates[task_type] = {
                "system": "You are a creative story writer.",
                "user": Template(content)
            }

            print(f"✅ Loaded template from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"⚠️  Failed to load template {filepath}: {e}")

    def render(self, task_type: str, variables: Dict[str, Any]) -> str:
        """
        Render a prompt from template.

        Args:
            task_type: Type of task (blueprint, characters, etc.)
            variables: Variables to substitute in template

        Returns:
            Rendered prompt string
        """
        # Get template (fallback to generic if not found)
        template_data = self.templates.get(task_type, self.templates.get("generic"))

        if not template_data:
            raise ValueError(f"No template found for task type: {task_type}")

        # Build context strings from variables
        context_vars = self._build_context_vars(variables)

        # Render user prompt
        user_template = template_data.get("user")
        if isinstance(user_template, Template):
            try:
                user_prompt = user_template.safe_substitute(**context_vars)
            except KeyError as e:
                # If variable missing, just use template as-is
                user_prompt = user_template.template
        else:
            user_prompt = str(user_template)

        # Combine system + user prompts
        system_prompt = template_data.get("system", "")

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        return full_prompt

    def _build_context_vars(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """
        Build context variables for template substitution.

        Converts complex objects (dicts, lists) to readable strings.
        """
        context_vars = {}

        for key, value in variables.items():
            if value is None:
                context_vars[key] = ""
            elif isinstance(value, str):
                context_vars[key] = value
            elif isinstance(value, dict):
                # Convert dict to readable format
                context_vars[key] = self._format_dict(value)
            elif isinstance(value, list):
                # Convert list to readable format
                context_vars[key] = self._format_list(value)
            else:
                context_vars[key] = str(value)

        # Build special context strings
        if 'blueprint' in variables and variables['blueprint']:
            context_vars['blueprint_context'] = f"Blueprint:\n{self._format_dict(variables['blueprint'])}"
        else:
            context_vars['blueprint_context'] = ""

        if 'characters' in variables and variables['characters']:
            context_vars['characters_context'] = f"Characters:\n{self._format_list(variables['characters'])}"
        else:
            context_vars['characters_context'] = ""

        if 'story_arc' in variables and variables['story_arc']:
            context_vars['story_context'] = f"Story Arc:\n{self._format_dict(variables['story_arc'])}"
        else:
            context_vars['story_context'] = ""

        # Fallback context
        if 'context' not in context_vars:
            context_vars['context'] = ""

        return context_vars

    def _format_dict(self, data: Dict) -> str:
        """Format dictionary as readable text."""
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except:
            return str(data)

    def _format_list(self, data: list) -> str:
        """Format list as readable text."""
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except:
            return str(data)

    def list_templates(self) -> list:
        """List all available template task types."""
        return list(self.templates.keys())

    def add_template(self, task_type: str, system: str, user_template: str):
        """
        Add or update a template programmatically.

        Args:
            task_type: Task type identifier
            system: System prompt
            user_template: User prompt template (use ${variable} for substitution)
        """
        self.templates[task_type] = {
            "system": system,
            "user": Template(user_template)
        }

    def remove_template(self, task_type: str):
        """Remove a template."""
        if task_type in self.templates:
            del self.templates[task_type]
