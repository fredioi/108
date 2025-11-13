# Prompt Templates Directory

This directory contains customizable prompt templates for story generation.

## How It Works

The template system allows you to customize prompts without modifying code. Templates support:

- **Multiple formats**: YAML, JSON, or plain text
- **Variable substitution**: Use `${variable_name}` for dynamic content
- **Task-specific templates**: Different prompts for different generation tasks
- **Fallback system**: Missing templates use built-in defaults

## Template Structure

### YAML/JSON Format
```yaml
task_type:
  system: "System prompt (sets context and role)"
  user: |
    User prompt template
    Use ${variable_name} for substitution
```

### Text File Format
Create a `.txt` file named after the task type (e.g., `blueprint.txt`). The entire file content becomes the user prompt.

## Available Task Types

- `blueprint` - Story blueprint generation
- `characters` - Character creation
- `story_arc` - Story arc/chapter planning
- `chapter_content` - Chapter content generation
- `generic` - Fallback for unknown tasks

## Available Variables

### Common Variables
- `${user_input}` - User's request/input
- `${context}` - General context (fallback)

### Context Variables (automatically formatted)
- `${blueprint_context}` - Story blueprint data
- `${characters_context}` - Characters data
- `${story_context}` - Story arc data

### Chapter-Specific Variables
- `${chapter_number}` - Chapter number
- `${chapter_title}` - Chapter title
- `${chapter_summary}` - Chapter summary

## Examples

### Example 1: Simple Text Template

**File**: `config/prompts/blueprint.txt`
```
Create a story blueprint for: ${user_input}

Include title, genre, setting, and themes.
```

### Example 2: YAML Template

**File**: `config/prompts/my_templates.yaml`
```yaml
characters:
  system: "You are a character designer."
  user: |
    Create 5 characters for: ${user_input}

    ${blueprint_context}

    Make them diverse and interesting!
```

### Example 3: JSON Template

**File**: `config/prompts/advanced.json`
```json
{
  "chapter_content": {
    "system": "You are a master storyteller.",
    "user": "Write chapter ${chapter_number}:\n\n${chapter_summary}\n\nMake it engaging!"
  }
}
```

## Usage in Code

Templates are automatically loaded by generators:

```python
from generators.prompt_templates import PromptTemplateManager

# Initialize (loads from config/prompts/)
manager = PromptTemplateManager()

# Render prompt
prompt = manager.render("blueprint", {
    "user_input": "A space adventure story"
})

# Add custom template
manager.add_template(
    "my_task",
    system="You are...",
    user_template="Generate: ${user_input}"
)
```

## Best Practices

1. **Start with examples** - Copy `custom_templates.yaml.example` to `custom_templates.yaml`
2. **Test your templates** - Verify variable substitution works correctly
3. **Be specific** - Clear prompts get better results
4. **Use system prompts** - Set context and role for better output
5. **Version control** - Track template changes for reproducibility

## Configuration

In `config/generator_config.yaml`:

```yaml
generator:
  # Template directory (optional)
  template_dir: "./config/prompts"  # default location
```

## Troubleshooting

**Templates not loading?**
- Check file format (YAML/JSON syntax)
- Verify directory path exists
- Look for error messages in logs

**Variables not substituting?**
- Use `${variable_name}` syntax (not `{variable_name}`)
- Check spelling of variable names
- Verify variable is passed to `render()`

**Want to disable custom templates?**
- Delete template files (built-in defaults will be used)
- Or specify empty directory in config
