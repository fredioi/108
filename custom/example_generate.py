#!/usr/bin/env python3
"""
Example Custom Generator Script

This is a template for creating custom story generators.
The script receives JSON input via stdin and outputs JSON via stdout.

Usage:
    echo '{"user_input": "test"}' | python custom/example_generate.py
"""

import json
import sys


def generate_story(input_data):
    """
    Generate story content based on input data.
    
    Args:
        input_data: Dict containing:
            - user_input: str
            - blueprint: Dict (optional)
            - characters: Dict (optional)
            - story_arc: Dict (optional)
            - chapter_ids: List (optional)
            - task_type: str
    
    Returns:
        str: Generated story content (can be JSON string)
    """
    user_input = input_data.get("user_input", "")
    task_type = input_data.get("task_type", "unknown")
    
    # Example: Simple template-based generation
    if task_type == "blueprint":
        return json.dumps({
            "title": f"Story based on: {user_input}",
            "genre": "Adventure",
            "setting": "A mysterious world",
            "core_conflict": "The protagonist must overcome challenges",
            "themes": ["courage", "friendship", "discovery"],
            "tone": "Exciting",
            "target_audience": "Young adults"
        }, ensure_ascii=False)
    
    elif task_type == "characters":
        return json.dumps({
            "characters": [
                {
                    "id": "protagonist",
                    "name": "Hero",
                    "archetype": "Hero",
                    "background": "A brave adventurer",
                    "motivation": "To save the world",
                    "skills": ["Swordsmanship", "Leadership", "Courage"],
                    "personality_traits": ["Brave", "Kind", "Determined"],
                    "relationships": {
                        "ally": "Trusted companion",
                        "rival": "Worthy opponent"
                    }
                },
                # Add 4 more characters...
            ]
        }, ensure_ascii=False)
    
    else:
        # Default: Simple text generation
        return f"Generated story based on: {user_input}\n\nThis is a custom generated story..."


def main():
    """Main entry point."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Generate story
        content = generate_story(input_data)
        
        # Output JSON to stdout
        output = {
            "content": content
        }
        print(json.dumps(output, ensure_ascii=False))
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        # Output error to stderr (will be captured)
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
