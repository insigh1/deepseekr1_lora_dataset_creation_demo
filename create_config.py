import yaml
import argparse
from typing import Dict, Any, List, Union
import os
import requests
import json
import logging
from pathlib import Path
import datetime
import sys
from config_models import (
    Configuration, SamplingParameters, PromptStructure, DatasetParameters,
    AdvancedOptions, QualityControls, DiversityParameters, EvaluationMetrics,
    Metadata, MessageSequence, Message
)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_fireworks_headers() -> Dict[str, str]:
    """Get the headers for Fireworks API requests."""
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable must be set")
    
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

def generate_smart_config(dataset_type: str, requirements: str) -> Configuration:
    """Use Fireworks AI to generate appropriate configuration parameters."""
    system_prompt = """You are a configuration expert for AI training data generation.
Given a dataset type and requirements, generate appropriate configuration parameters.
Focus on optimal settings for the specific use case.

Temperature Range Guidelines:
- Technical/Precise Content (0.0-0.3): For documentation, code, facts, and exact specifications
- Professional/Formal Content (0.3-0.5): For business, legal, and formal communications
- Balanced Content (0.5-0.7): For general assistance and balanced responses
- Creative Content (0.7-0.9): For creative writing, brainstorming, and imaginative tasks

Consider these factors when setting temperature:
1. Need for precision vs creativity
2. Technical complexity
3. Required consistency
4. Domain-specific requirements

For prompt templates, you should include relevant sections based on the use case.
Common sections include:
- [Task]: Main task description
- [Requirements]: Specific requirements and constraints
- [Quality Checks]: Validation steps and quality criteria
- [Input Data]: Input format and placeholders
- [Context]: Additional context or background
- [Examples]: Example inputs/outputs
- [Constraints]: Specific limitations or requirements
- [Format]: Output format requirements

Guidelines for template sections:
1. Include only sections relevant to the specific use case
2. Each section should serve a clear purpose
3. Sections can be customized based on domain needs
4. Use appropriate placeholders for dynamic content
5. Consider the target audience and their needs

Example template structures:

For code generation:
[Task] {task_description}
[Requirements]
- Technical requirements
- Style requirements
- Performance requirements
[Quality Checks]
- Code quality checks
- Performance validation
- Security checks
[Input Data]
Function: "{function_description}"
Context: "{context}"
Constraints: "{constraints}"

For creative writing:
[Task] {task_description}
[Requirements]
- Style requirements
- Genre conventions
- Length requirements
[Quality Checks]
- Narrative flow
- Character consistency
- World-building elements
[Input Data]
Prompt: "{story_prompt}"
Style: "{writing_style}"
[Context]
Setting: "{setting}"
Characters: "{characters}"

For business content:
[Task] {task_description}
[Requirements]
- Professional tone
- Data requirements
- Format requirements
[Quality Checks]
- Clarity checks
- Professional standards
- Data accuracy
[Input Data]
Topic: "{topic}"
Audience: "{audience}"
[Format]
Output: "{output_format}"

IMPORTANT: Your response must be a valid JSON object matching the exact structure provided."""

    user_prompt = f"""Generate configuration parameters for a {dataset_type} dataset with the following requirements:
{requirements}

IMPORTANT: Respond with ONLY a valid JSON object, no other text. The JSON must match this exact structure:
{{
    "sampling_parameters": {{
        "temperature_range": [min, max],
        "top_p": 0.9
    }},
    "prompt_structure": {{
        "user_prompt_template": "string",
        "language_style": "string"
    }},
    "required_criteria": ["string"],
    "advanced_options": {{
        "strict_mode": true
    }},
    "dataset_parameters": {{
        "output_format": "string",
        "task_description": "string",
        "max_tokens": 300,
        "seed_value": 42,
        "validation_rules": ["string"]
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 4096,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            config_json = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    config_json = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse JSON from response")
            else:
                raise ValueError("No valid JSON found in response")
        
        # Create Pydantic model from JSON
        try:
            config = Configuration(
                sampling_parameters=SamplingParameters(**config_json["sampling_parameters"]),
                prompt_structure=PromptStructure(**config_json["prompt_structure"]),
                required_criteria=config_json["required_criteria"],
                advanced_options=AdvancedOptions(**config_json["advanced_options"]),
                dataset_parameters=DatasetParameters(**config_json["dataset_parameters"])
            )
            return config
        except Exception as e:
            logging.error(f"Error creating configuration model: {str(e)}")
            raise
        
    except Exception as e:
        logging.error(f"Error generating smart config: {str(e)}")
        raise

def create_default_config() -> Configuration:
    """Create a default configuration using Pydantic models."""
    return Configuration(
        sampling_parameters=SamplingParameters(
            temperature_range=(0.5, 0.7),
            top_p=0.9
        ),
        prompt_structure=PromptStructure(
            user_prompt_template="""[Task] {task_description}

[Requirements]
- Respond in {output_format}
- Keep responses under {max_tokens} tokens
- Use {language_style} tone
- Each JSON object must contain a single array field called messages
- Each message must have a role ("system", "user", or "assistant") and content
- System message (if present) must be first
- Messages must alternate between user and assistant roles

[Quality Checks]
- Verify response format
- Check length constraints
- Ensure appropriate tone
- Validate message sequence format

[Input Data]
"{raw_input}" """,
            language_style="professional"
        ),
        required_criteria=[
            "Output must match the expected data format",
            "No hallucinations about real-world facts",
            "Responses must address all parts of the input prompt",
            "Maintain consistency with previous context (if applicable)",
            "Messages must follow the required format with proper role alternation"
        ],
        advanced_options=AdvancedOptions(strict_mode=True),
        dataset_parameters=DatasetParameters(
            output_format="json",
            task_description="You are an AI assistant. Generate appropriate responses based on the input.",
            max_tokens=200,
            seed_value=42,
            validation_rules=[
                "Check output format validity",
                "Ensure response length matches token limits",
                "Validate message sequence format"
            ],
            message_format=MessageSequence(
                messages=[
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content="What color is the sky?"),
                    Message(role="assistant", content="blue")
                ]
            )
        )
    )

def update_config_section(config: Configuration, section: str, value: Any) -> None:
    """Update a specific section of the configuration."""
    if hasattr(config, section):
        setattr(config, section, value)
    else:
        # Handle nested updates using dot notation (e.g., "dataset_parameters.output_format")
        parts = section.split('.')
        current = config
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Invalid section: {part}")
        setattr(current, parts[-1], value)

def save_config(config: Configuration, output_path: str) -> None:
    """Save the configuration to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def validate_config_structure(config: Configuration) -> bool:
    """Validate the configuration structure and content."""
    try:
        # Validate message format
        if not config.dataset_parameters.message_format:
            raise ValueError("Message format is required")
        
        # Validate message sequence
        messages = config.dataset_parameters.message_format.messages
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Check if system message is first (if present)
        if messages[0].role == "system":
            messages = messages[1:]
        
        # Check alternating user/assistant pattern
        for i, msg in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg.role != expected_role:
                raise ValueError(f"Message sequence must alternate between user and assistant roles. Found {msg.role} when expecting {expected_role}")
        
        # Pydantic models handle other validation automatically
        return True
    except Exception as e:
        logging.error(f"Configuration validation error: {str(e)}")
        return False

def enhance_config_with_context(config: Configuration, dataset_type: str, requirements: str) -> Configuration:
    """Enhance the configuration with context-specific improvements using LLM."""
    system_prompt = """You are a configuration enhancement expert.
Given a configuration and context, suggest appropriate enhancements to optimize it for the specific use case.
Focus on making the configuration more effective and contextually appropriate.

Consider:
1. Language style and tone
2. Validation rules specific to the domain
3. Appropriate token limits
4. Required metadata
5. Domain-specific parameters

For technical documentation, consider:
- Code example requirements
- Deployment strategy depth
- Structural requirements
- Technical depth level
- Architecture focus areas
- Implementation details
- Best practices
- Security considerations
- Scalability factors
- Monitoring requirements
- Testing strategies

Respond with ONLY a valid JSON object containing the enhancements."""

    user_prompt = f"""Enhance this configuration for a {dataset_type} dataset with these requirements:
{requirements}

Current configuration:
{json.dumps(config.model_dump(), indent=2, cls=DateTimeEncoder)}

IMPORTANT: Respond with ONLY a valid JSON object containing the enhancements. The JSON should include:
{{
    "metadata": {{
        "generated_at": "timestamp",
        "dataset_type": "string",
        "requirements_summary": "string",
        "domain_specific_metadata": {{}}
    }},
    "enhancements": {{
        "language_style": "string",
        "validation_rules": ["string"],
        "max_tokens": number,
        "domain_specific_parameters": {{
            "code_example_requirements": ["string"],
            "deployment_strategy_depth": "string",
            "structural_requirements": ["string"],
            "technical_depth": "string",
            "architecture_focus": ["string"],
            "implementation_details": {{}},
            "best_practices": ["string"],
            "security_considerations": ["string"],
            "scalability_factors": ["string"],
            "monitoring_requirements": ["string"],
            "testing_strategies": ["string"]
        }}
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            enhancements = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    enhancements = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logging.error("Failed to parse enhancement response as JSON")
                    return config
            else:
                logging.error("No valid JSON found in enhancement response")
                return config
        
        # Apply the enhancements
        if "metadata" in enhancements:
            try:
                config.metadata = Metadata(**enhancements["metadata"])
            except Exception as e:
                logging.error(f"Error applying metadata: {str(e)}")
        
        if "enhancements" in enhancements:
            enh = enhancements["enhancements"]
            
            # Update language style if provided
            if "language_style" in enh:
                config.prompt_structure.language_style = enh["language_style"]
            
            # Update validation rules if provided
            if "validation_rules" in enh:
                config.dataset_parameters.validation_rules.extend(enh["validation_rules"])
            
            # Update max_tokens if provided
            if "max_tokens" in enh:
                config.dataset_parameters.max_tokens = enh["max_tokens"]
            
            # Add domain-specific parameters
            if "domain_specific_parameters" in enh:
                domain_params = enh["domain_specific_parameters"]
                for key, value in domain_params.items():
                    try:
                        if hasattr(config.dataset_parameters, key):
                            setattr(config.dataset_parameters, key, value)
                        else:
                            logging.warning(f"Unknown domain parameter: {key}")
                    except Exception as e:
                        logging.error(f"Error setting domain parameter {key}: {str(e)}")
        
        return config
    except Exception as e:
        logging.error(f"Error enhancing config: {str(e)}")
        return config

def format_yaml_output(config: Configuration, output_path: str) -> None:
    """Format and save the YAML output with proper structure and comments."""
    # Create a formatted YAML string with comments
    yaml_content = f"""# Configuration generated for {config.metadata.dataset_type if config.metadata else 'unknown'} dataset
# Generated at: {config.metadata.generated_at if config.metadata else 'unknown'}

# Sampling parameters for controlling response generation
sampling_parameters:
  temperature_range: {config.sampling_parameters.temperature_range}  # Controls response randomness
  top_p: {config.sampling_parameters.top_p}  # Nucleus sampling parameter

# Structure for generating prompts
prompt_structure:
  user_prompt_template: |
{chr(10).join('    ' + line for line in config.prompt_structure.user_prompt_template.split(chr(10)))}
  language_style: {config.prompt_structure.language_style}

# Required criteria for response validation
required_criteria:
{chr(10).join('  - ' + criterion for criterion in config.required_criteria)}

# Advanced configuration options
advanced_options:
  strict_mode: {config.advanced_options.strict_mode}

# Dataset-specific parameters
dataset_parameters:
  output_format: {config.dataset_parameters.output_format}
  task_description: {config.dataset_parameters.task_description}
  max_tokens: {config.dataset_parameters.max_tokens}
  seed_value: {config.dataset_parameters.seed_value}
  validation_rules:
{chr(10).join('    - ' + rule for rule in config.dataset_parameters.validation_rules)}

# Quality control parameters
quality_controls:
  response_validation:
    min_length: {config.quality_controls.response_validation['min_length'] if config.quality_controls else 50}
    max_length: {config.quality_controls.response_validation['max_length'] if config.quality_controls else 2000}
    required_elements:
{chr(10).join('      - ' + element for element in (config.quality_controls.response_validation['required_elements'] if config.quality_controls else []))}
    forbidden_elements:
{chr(10).join('      - ' + element for element in (config.quality_controls.response_validation['forbidden_elements'] if config.quality_controls else []))}
  diversity_controls:
    min_unique_words: {config.quality_controls.diversity_controls['min_unique_words'] if config.quality_controls else 10}
    max_repetition: {config.quality_controls.diversity_controls['max_repetition'] if config.quality_controls else 0.2}
    style_variation: {config.quality_controls.diversity_controls['style_variation'] if config.quality_controls else 0.7}
  consistency_checks:
    context_window: {config.quality_controls.consistency_checks['context_window'] if config.quality_controls else 5}
    style_consistency: {config.quality_controls.consistency_checks['style_consistency'] if config.quality_controls else 0.8}
    fact_consistency: {config.quality_controls.consistency_checks['fact_consistency'] if config.quality_controls else 0.9}

# Diversity parameters
diversity_parameters:
  variation_controls:
    temperature_variation: {config.diversity_parameters.variation_controls['temperature_variation'] if config.diversity_parameters else 0.2}
    style_variation: {config.diversity_parameters.variation_controls['style_variation'] if config.diversity_parameters else 0.3}
    complexity_variation: {config.diversity_parameters.variation_controls['complexity_variation'] if config.diversity_parameters else 0.4}
  sampling_strategy:
    method: {config.diversity_parameters.sampling_strategy['method'] if config.diversity_parameters else 'adaptive'}
    min_unique_ratio: {config.diversity_parameters.sampling_strategy['min_unique_ratio'] if config.diversity_parameters else 0.7}
    max_similarity: {config.diversity_parameters.sampling_strategy['max_similarity'] if config.diversity_parameters else 0.3}
  content_balancing:
    topic_distribution: {config.diversity_parameters.content_balancing['topic_distribution'] if config.diversity_parameters else 'uniform'}
    difficulty_levels:
{chr(10).join('      - ' + level for level in (config.diversity_parameters.content_balancing['difficulty_levels'] if config.diversity_parameters else ['beginner', 'intermediate', 'advanced']))}
    style_distribution:
{chr(10).join('      - ' + style for style in (config.diversity_parameters.content_balancing['style_distribution'] if config.diversity_parameters else ['formal', 'casual', 'technical']))}

# Evaluation metrics
evaluation_metrics:
  quality_metrics:
    coherence_score: {config.evaluation_metrics.quality_metrics['coherence_score'] if config.evaluation_metrics else 0.8}
    relevance_score: {config.evaluation_metrics.quality_metrics['relevance_score'] if config.evaluation_metrics else 0.9}
    completeness_score: {config.evaluation_metrics.quality_metrics['completeness_score'] if config.evaluation_metrics else 0.85}
  diversity_metrics:
    vocabulary_richness: {config.evaluation_metrics.diversity_metrics['vocabulary_richness'] if config.evaluation_metrics else 0.7}
    syntactic_diversity: {config.evaluation_metrics.diversity_metrics['syntactic_diversity'] if config.evaluation_metrics else 0.6}
    semantic_diversity: {config.evaluation_metrics.diversity_metrics['semantic_diversity'] if config.evaluation_metrics else 0.75}
  consistency_metrics:
    factual_accuracy: {config.evaluation_metrics.consistency_metrics['factual_accuracy'] if config.evaluation_metrics else 0.95}
    style_consistency: {config.evaluation_metrics.consistency_metrics['style_consistency'] if config.evaluation_metrics else 0.85}
    context_relevance: {config.evaluation_metrics.consistency_metrics['context_relevance'] if config.evaluation_metrics else 0.9}

# Example inputs for reference
example_inputs:
{chr(10).join('  - ' + example for example in config.example_inputs)}
"""
    
    # Save the formatted YAML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)

def add_quality_controls(config: Configuration, dataset_type: str) -> Configuration:
    """Add quality control parameters for dataset generation using LLM."""
    system_prompt = """You are a quality control expert for AI training data generation.
Given a dataset type and context, determine appropriate quality control parameters.
Focus on ensuring high-quality, consistent, and appropriate content.

Consider:
1. Required elements based on content type
2. Forbidden elements to avoid
3. Length constraints
4. Style requirements
5. Domain-specific quality criteria

Respond with ONLY a valid JSON object containing the quality control parameters."""

    user_prompt = f"""Generate quality control parameters for a {dataset_type} dataset.

Current configuration:
{json.dumps(config.model_dump(), indent=2, cls=DateTimeEncoder)}

IMPORTANT: Respond with ONLY a valid JSON object containing:
{{
    "response_validation": {{
        "min_length": number,
        "max_length": number,
        "required_elements": ["string"],
        "forbidden_elements": ["string"]
    }},
    "diversity_controls": {{
        "min_unique_words": number,
        "max_repetition": number,
        "style_variation": number
    }},
    "consistency_checks": {{
        "context_window": number,
        "style_consistency": number,
        "fact_consistency": number
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            quality_params = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    quality_params = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Failed to parse quality control response as JSON, using defaults")
                    quality_params = {
                        "response_validation": {
                            "min_length": 50,
                            "max_length": 2000,
                            "required_elements": [],
                            "forbidden_elements": []
                        },
                        "diversity_controls": {
                            "min_unique_words": 10,
                            "max_repetition": 0.2,
                            "style_variation": 0.7
                        },
                        "consistency_checks": {
                            "context_window": 5,
                            "style_consistency": 0.8,
                            "fact_consistency": 0.9
                        }
                    }
            else:
                logging.warning("No valid JSON found in quality control response, using defaults")
                quality_params = {
                    "response_validation": {
                        "min_length": 50,
                        "max_length": 2000,
                        "required_elements": [],
                        "forbidden_elements": []
                    },
                    "diversity_controls": {
                        "min_unique_words": 10,
                        "max_repetition": 0.2,
                        "style_variation": 0.7
                    },
                    "consistency_checks": {
                        "context_window": 5,
                        "style_consistency": 0.8,
                        "fact_consistency": 0.9
                    }
                }
        
        # Create and set quality controls
        quality_controls = QualityControls(
            response_validation=quality_params["response_validation"],
            diversity_controls=quality_params["diversity_controls"],
            consistency_checks=quality_params["consistency_checks"]
        )
        
        config.quality_controls = quality_controls
        return config
    except Exception as e:
        logging.error(f"Error adding quality controls: {str(e)}")
        return config

def add_diversity_controls(config: Configuration, dataset_type: str) -> Configuration:
    """Add controls for ensuring dataset diversity using LLM."""
    system_prompt = """You are a diversity control expert for AI training data generation.
Given a dataset type and context, determine appropriate diversity parameters.
Focus on ensuring varied, balanced, and representative content.

Consider:
1. Variation in style and tone
2. Complexity distribution
3. Topic coverage
4. Sampling strategy
5. Content balancing requirements

Respond with ONLY a valid JSON object containing the diversity parameters."""

    user_prompt = f"""Generate diversity parameters for a {dataset_type} dataset.

Current configuration:
{json.dumps(config.model_dump(), indent=2, cls=DateTimeEncoder)}

IMPORTANT: Respond with ONLY a valid JSON object containing:
{{
    "variation_controls": {{
        "temperature_variation": number,
        "style_variation": number,
        "complexity_variation": number
    }},
    "sampling_strategy": {{
        "method": "string",
        "min_unique_ratio": number,
        "max_similarity": number
    }},
    "content_balancing": {{
        "topic_distribution": "string",
        "difficulty_levels": ["string"],
        "style_distribution": ["string"]
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            diversity_params = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    diversity_params = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Failed to parse diversity control response as JSON, using defaults")
                    diversity_params = {
                        "variation_controls": {
                            "temperature_variation": 0.2,
                            "style_variation": 0.3,
                            "complexity_variation": 0.4
                        },
                        "sampling_strategy": {
                            "method": "adaptive",
                            "min_unique_ratio": 0.7,
                            "max_similarity": 0.3
                        },
                        "content_balancing": {
                            "topic_distribution": "uniform",
                            "difficulty_levels": ["beginner", "intermediate", "advanced"],
                            "style_distribution": ["formal", "casual", "technical"]
                        }
                    }
            else:
                logging.warning("No valid JSON found in diversity control response, using defaults")
                diversity_params = {
                    "variation_controls": {
                        "temperature_variation": 0.2,
                        "style_variation": 0.3,
                        "complexity_variation": 0.4
                    },
                    "sampling_strategy": {
                        "method": "adaptive",
                        "min_unique_ratio": 0.7,
                        "max_similarity": 0.3
                    },
                    "content_balancing": {
                        "topic_distribution": "uniform",
                        "difficulty_levels": ["beginner", "intermediate", "advanced"],
                        "style_distribution": ["formal", "casual", "technical"]
                    }
                }
        
        # Create and set diversity parameters
        diversity_parameters = DiversityParameters(
            variation_controls=diversity_params["variation_controls"],
            sampling_strategy=diversity_params["sampling_strategy"],
            content_balancing=diversity_params["content_balancing"]
        )
        
        config.diversity_parameters = diversity_parameters
        return config
    except Exception as e:
        logging.error(f"Error adding diversity controls: {str(e)}")
        return config

def add_evaluation_metrics(config: Configuration, dataset_type: str) -> Configuration:
    """Add metrics for evaluating dataset quality using LLM."""
    system_prompt = """You are an evaluation metrics expert for AI training data generation.
Given a dataset type and context, determine appropriate evaluation metrics.
Focus on measuring quality, diversity, and consistency.

Consider:
1. Quality metrics (coherence, relevance, completeness)
2. Diversity metrics (vocabulary, syntactic, semantic)
3. Consistency metrics (factual accuracy, style, context)
4. Domain-specific evaluation criteria
5. Appropriate thresholds for each metric

Respond with ONLY a valid JSON object containing the evaluation metrics."""

    user_prompt = f"""Generate evaluation metrics for a {dataset_type} dataset.

Current configuration:
{json.dumps(config.model_dump(), indent=2, cls=DateTimeEncoder)}

IMPORTANT: Respond with ONLY a valid JSON object containing:
{{
    "quality_metrics": {{
        "coherence_score": number,
        "relevance_score": number,
        "completeness_score": number
    }},
    "diversity_metrics": {{
        "vocabulary_richness": number,
        "syntactic_diversity": number,
        "semantic_diversity": number
    }},
    "consistency_metrics": {{
        "factual_accuracy": number,
        "style_consistency": number,
        "context_relevance": number
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            metric_params = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    metric_params = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logging.warning("Failed to parse evaluation metrics response as JSON, using defaults")
                    metric_params = {
                        "quality_metrics": {
                            "coherence_score": 0.8,
                            "relevance_score": 0.9,
                            "completeness_score": 0.85
                        },
                        "diversity_metrics": {
                            "vocabulary_richness": 0.7,
                            "syntactic_diversity": 0.6,
                            "semantic_diversity": 0.75
                        },
                        "consistency_metrics": {
                            "factual_accuracy": 0.95,
                            "style_consistency": 0.85,
                            "context_relevance": 0.9
                        }
                    }
            else:
                logging.warning("No valid JSON found in evaluation metrics response, using defaults")
                metric_params = {
                    "quality_metrics": {
                        "coherence_score": 0.8,
                        "relevance_score": 0.9,
                        "completeness_score": 0.85
                    },
                    "diversity_metrics": {
                        "vocabulary_richness": 0.7,
                        "syntactic_diversity": 0.6,
                        "semantic_diversity": 0.75
                    },
                    "consistency_metrics": {
                        "factual_accuracy": 0.95,
                        "style_consistency": 0.85,
                        "context_relevance": 0.9
                    }
                }
        
        # Create and set evaluation metrics
        evaluation_metrics = EvaluationMetrics(
            quality_metrics=metric_params["quality_metrics"],
            diversity_metrics=metric_params["diversity_metrics"],
            consistency_metrics=metric_params["consistency_metrics"]
        )
        
        config.evaluation_metrics = evaluation_metrics
        return config
    except Exception as e:
        logging.error(f"Error adding evaluation metrics: {str(e)}")
        return config

def determine_dataset_type(requirements: str, examples: List[str]) -> str:
    """Use LLM to determine the appropriate dataset type based on requirements and examples."""
    system_prompt = """You are a dataset type classification expert.
Given requirements and examples, determine the most appropriate dataset type.
Consider the content style, technical complexity, and intended use case.

Analyze the task requirements and examples to determine:
1. The primary domain (e.g., technical, creative, business, etc.)
2. The complexity level (e.g., beginner, intermediate, advanced)
3. The specific use case (e.g., documentation, analysis, storytelling, etc.)
4. Any special requirements or constraints

Respond with a JSON object containing:
{
    "primary_type": "string",  # The main category (e.g., technical, creative, business)
    "subtype": "string",      # More specific type (e.g., code, documentation, story)
    "complexity": "string",   # Complexity level
    "use_case": "string",     # Specific use case
    "special_requirements": ["string"]  # Any special requirements
}

The primary_type should be one of:
- technical: For code, documentation, and technical content
- creative: For creative writing, storytelling, and imaginative content
- business: For business analysis, reports, and professional content
- educational: For educational content and explanations
- conversational: For dialogue and interactive content
- analytical: For data analysis and insights
- legal: For legal documents and compliance
- medical: For medical and healthcare content
- scientific: For scientific research and papers
- artistic: For artistic and design content
- social: For social media and community content
- gaming: For game-related content
- research: For research and academic content
- marketing: For marketing and advertising content
- financial: For financial and economic content
- environmental: For environmental and sustainability content
- political: For political and policy content
- cultural: For cultural and historical content
- sports: For sports and athletic content
- entertainment: For entertainment and media content

But you can also suggest new types if none of these fit well."""

    user_prompt = f"""Determine the appropriate dataset type for these requirements and examples:

Requirements:
{requirements}

Examples:
{chr(10).join(f"- {example}" for example in examples)}

IMPORTANT: Respond with ONLY a valid JSON object containing the type information."""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 500,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Parse the response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            logging.error("Failed to parse dataset type response as JSON")
            return "technical"  # Default to technical if parsing fails
        
        # Extract the primary type
        dataset_type = result.get("primary_type", "technical").lower()
        
        # Store additional metadata in the configuration
        metadata = {
            "subtype": result.get("subtype", ""),
            "complexity": result.get("complexity", "intermediate"),
            "use_case": result.get("use_case", ""),
            "special_requirements": result.get("special_requirements", [])
        }
        
        # Store the metadata for later use
        global task_metadata
        task_metadata = metadata
        
        return dataset_type
    except Exception as e:
        logging.error(f"Error determining dataset type: {str(e)}")
        return "technical"  # Default to technical if there's an error

def generate_output_filename(dataset_type: str, requirements: str) -> str:
    """Generate an appropriate output filename based on dataset type and requirements."""
    # Clean the requirements to create a meaningful filename
    clean_requirements = requirements.lower()
    # Remove special characters and replace spaces with underscores
    clean_requirements = ''.join(c if c.isalnum() or c.isspace() else '_' for c in clean_requirements)
    clean_requirements = '_'.join(clean_requirements.split())
    
    # Truncate if too long
    if len(clean_requirements) > 50:
        clean_requirements = clean_requirements[:50]
    
    # Generate the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_type}_{clean_requirements}_{timestamp}.yaml"
    
    return filename

def generate_task_config(task: str) -> tuple[Configuration, str, str, List[str]]:
    """Generate a complete configuration based on a task description."""
    system_prompt = """You are a configuration generation expert.
Given a task, generate appropriate requirements, examples, and configuration parameters.
Focus on creating a comprehensive setup that will help generate high-quality training data.

IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text, no explanations, no markdown formatting.
The JSON must be properly formatted and parseable.

The prompt template MUST include these required sections:
- [Task]: The main task description
- [Requirements]: Specific requirements and constraints
- [Input Data]: Input format and placeholders
- [Format]: Output format requirements

Consider:
1. What specific requirements are needed for this task?
2. What kind of examples would be helpful?
3. What dataset type is most appropriate?
4. What configuration parameters would optimize the output?

Example valid response format:
{
    "dataset_type": "technical",
    "requirements": "Generate Python code with proper error handling and documentation",
    "examples": ["How to implement error handling?", "What's the best practice for logging?"],
    "config": {
        "sampling_parameters": {
            "temperature_range": [0.2, 0.4],
            "top_p": 0.9
        },
        "prompt_structure": {
            "user_prompt_template": "[Task] {task_description}\n[Requirements]\n- {requirements}\n[Input Data]\nTopic: {topic}\n[Format]\nOutput: {output_format}",
            "language_style": "technical"
        },
        "required_criteria": ["code quality", "documentation"],
        "advanced_options": {
            "strict_mode": true
        },
        "dataset_parameters": {
            "output_format": "code",
            "task_description": "Generate code examples",
            "max_tokens": 500,
            "seed_value": 42,
            "validation_rules": ["PEP 8 compliance", "type hints"],
            "message_format": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the best way to handle errors in Python?"},
                    {"role": "assistant", "content": "Here are some best practices for error handling in Python..."}
                ]
            }
        }
    }
}"""

    user_prompt = f"""Generate a complete configuration for this task:
{task}

IMPORTANT: Respond with ONLY a valid JSON object, no other text. The JSON must match this exact structure:
{{
    "dataset_type": "string",  # The most appropriate type for this task
    "requirements": "string",  # Detailed requirements for the task
    "examples": ["string"],    # Relevant example inputs
    "config": {{
        "sampling_parameters": {{
            "temperature_range": [min, max],
            "top_p": 0.9
        }},
        "prompt_structure": {{
            "user_prompt_template": "string",  # Must include [Task], [Requirements], [Input Data], and [Format] sections
            "language_style": "string"
        }},
        "required_criteria": ["string"],
        "advanced_options": {{
            "strict_mode": true
        }},
        "dataset_parameters": {{
            "output_format": "string",
            "task_description": "string",
            "max_tokens": 300,
            "seed_value": 42,
            "validation_rules": ["string"],
            "message_format": {{
                "messages": [
                    {{"role": "system", "content": "string"}},
                    {{"role": "user", "content": "string"}},
                    {{"role": "assistant", "content": "string"}}
                ]
            }}
        }}
    }}
}}"""

    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3-0324",
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post(url, headers=get_fireworks_headers(), json=payload)
        response.raise_for_status()
        
        # Get the response content
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Try to find JSON in the response if it's wrapped in other text
        try:
            # First try direct JSON parsing
            result = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to find JSON-like content between triple backticks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse JSON from response")
            else:
                raise ValueError("No valid JSON found in response")
        
        # Validate required fields
        required_fields = ["dataset_type", "requirements", "examples", "config"]
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"Response missing required fields: {', '.join(missing_fields)}")
        
        # Extract components
        dataset_type = result["dataset_type"]
        requirements = result["requirements"]
        examples = result["examples"]
        config_json = result["config"]
        
        # Ensure prompt template has required sections
        if "prompt_structure" in config_json:
            template = config_json["prompt_structure"]["user_prompt_template"]
            required_sections = ["[Task]", "[Requirements]", "[Input Data]", "[Format]"]
            missing_sections = [section for section in required_sections if section not in template]
            if missing_sections:
                # Add missing sections with default content
                if "[Task]" not in template:
                    template = f"[Task] {{task_description}}\n{template}"
                if "[Requirements]" not in template:
                    template = template.replace("[Input Data]", "[Requirements]\n- {requirements}\n[Input Data]")
                if "[Input Data]" not in template:
                    template = f"{template}\n[Input Data]\nTopic: {{topic}}"
                if "[Format]" not in template:
                    template = f"{template}\n[Format]\nOutput: {{output_format}}"
                config_json["prompt_structure"]["user_prompt_template"] = template
        
        # Create Pydantic model from config JSON
        config = Configuration(
            sampling_parameters=SamplingParameters(**config_json["sampling_parameters"]),
            prompt_structure=PromptStructure(**config_json["prompt_structure"]),
            required_criteria=config_json["required_criteria"],
            advanced_options=AdvancedOptions(**config_json["advanced_options"]),
            dataset_parameters=DatasetParameters(
                output_format=config_json["dataset_parameters"]["output_format"],
                task_description=config_json["dataset_parameters"]["task_description"],
                max_tokens=config_json["dataset_parameters"]["max_tokens"],
                seed_value=config_json["dataset_parameters"]["seed_value"],
                validation_rules=config_json["dataset_parameters"]["validation_rules"],
                message_format=MessageSequence(**config_json["dataset_parameters"]["message_format"])
            )
        )
        
        # Enhance the configuration with context
        config = enhance_config_with_context(config, dataset_type, requirements)
        
        return (config, dataset_type, requirements, examples)
    except Exception as e:
        logging.error(f"Error generating task config: {str(e)}")
        raise

def main():
    """Main function to generate and save the configuration."""
    parser = argparse.ArgumentParser(description='Generate a YAML configuration file for dataset generation.')
    parser.add_argument('--task', type=str, required=True, help='Task description')
    parser.add_argument('--output', type=str, help='Output YAML file path (optional)')
    args = parser.parse_args()

    try:
        # Generate complete configuration from task
        config, dataset_type, requirements, examples = generate_task_config(args.task)
        
        # Add examples to config
        config.example_inputs = examples
        
        # Add quality controls
        config = add_quality_controls(config, dataset_type)
        
        # Add diversity parameters
        config = add_diversity_controls(config, dataset_type)
        
        # Add evaluation metrics
        config = add_evaluation_metrics(config, dataset_type)
        
        # Generate output filename if not provided
        output_path = args.output if args.output else generate_output_filename(dataset_type, requirements)
        
        # Format and save the YAML output
        format_yaml_output(config, output_path)
        
        logging.info(f"Configuration saved to {output_path}")
        logging.info(f"Dataset type: {dataset_type}")
        logging.info(f"Generated {len(examples)} examples")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 