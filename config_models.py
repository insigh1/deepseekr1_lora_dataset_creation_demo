from typing import Dict, Any, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime

class SamplingParameters(BaseModel):
    temperature_range: Tuple[float, float] = Field(..., description="Temperature range for sampling [min, max]")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")

    @validator('temperature_range')
    def validate_temperature_range(cls, v):
        min_temp, max_temp = v
        if min_temp < 0 or max_temp > 1:
            raise ValueError("Temperature range must be between 0 and 1")
        if min_temp > max_temp:
            raise ValueError("Minimum temperature must be less than maximum temperature")
        return v

class PromptStructure(BaseModel):
    user_prompt_template: str = Field(..., description="Template for user prompts")
    language_style: str = Field(..., description="Style of language to use")

    @validator('user_prompt_template')
    def validate_template_sections(cls, v):
        required_sections = ["[Task]"]
        missing_sections = [section for section in required_sections if section not in v]
        if missing_sections:
            raise ValueError(f"Prompt template missing required sections: {', '.join(missing_sections)}")
        return v

class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class MessageSequence(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")

    @validator('messages')
    def validate_message_sequence(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        # Check if system message is first (if present)
        if v[0].role == "system":
            messages = v[1:]
        else:
            messages = v
        
        # Check alternating user/assistant pattern
        for i, msg in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg.role != expected_role:
                raise ValueError(f"Message sequence must alternate between user and assistant roles. Found {msg.role} when expecting {expected_role}")
        
        return v

class DatasetParameters(BaseModel):
    output_format: str = Field(..., description="Format of the output")
    task_description: str = Field(..., description="Description of the task")
    max_tokens: int = Field(300, gt=0, description="Maximum number of tokens")
    seed_value: int = Field(42, description="Random seed value")
    validation_rules: List[str] = Field(default_factory=list, description="Rules for validation")
    message_format: MessageSequence = Field(..., description="Required message format for the dataset")
    
    # Domain-specific parameters
    code_example_requirements: Optional[List[str]] = Field(default_factory=list, description="Requirements for code examples")
    deployment_strategy_depth: Optional[str] = Field(default="comprehensive", description="Depth of deployment strategy coverage")
    structural_requirements: Optional[List[str]] = Field(default_factory=list, description="Structural requirements for the content")
    technical_depth: Optional[str] = Field(default="intermediate", description="Technical depth level")
    architecture_focus: Optional[List[str]] = Field(default_factory=list, description="Specific architecture aspects to focus on")
    implementation_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Implementation-specific details")
    best_practices: Optional[List[str]] = Field(default_factory=list, description="Best practices to include")
    security_considerations: Optional[List[str]] = Field(default_factory=list, description="Security aspects to cover")
    scalability_factors: Optional[List[str]] = Field(default_factory=list, description="Scalability considerations")
    monitoring_requirements: Optional[List[str]] = Field(default_factory=list, description="Monitoring and observability requirements")
    testing_strategies: Optional[List[str]] = Field(default_factory=list, description="Testing approaches to include")

class AdvancedOptions(BaseModel):
    strict_mode: bool = Field(True, description="Whether to use strict mode")

class QualityControls(BaseModel):
    response_validation: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_length": 50,
            "max_length": 2000,
            "required_elements": [],
            "forbidden_elements": []
        }
    )
    diversity_controls: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_unique_words": 10,
            "max_repetition": 0.2,
            "style_variation": 0.7
        }
    )
    consistency_checks: Dict[str, Any] = Field(
        default_factory=lambda: {
            "context_window": 5,
            "style_consistency": 0.8,
            "fact_consistency": 0.9
        }
    )

class DiversityParameters(BaseModel):
    variation_controls: Dict[str, float] = Field(
        default_factory=lambda: {
            "temperature_variation": 0.2,
            "style_variation": 0.3,
            "complexity_variation": 0.4
        }
    )
    sampling_strategy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "method": "adaptive",
            "min_unique_ratio": 0.7,
            "max_similarity": 0.3
        }
    )
    content_balancing: Dict[str, Any] = Field(
        default_factory=lambda: {
            "topic_distribution": "uniform",
            "difficulty_levels": ["beginner", "intermediate", "advanced"],
            "style_distribution": ["formal", "casual", "technical"]
        }
    )

class EvaluationMetrics(BaseModel):
    quality_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "coherence_score": 0.8,
            "relevance_score": 0.9,
            "completeness_score": 0.85
        }
    )
    diversity_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "vocabulary_richness": 0.7,
            "syntactic_diversity": 0.6,
            "semantic_diversity": 0.75
        }
    )
    consistency_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "factual_accuracy": 0.95,
            "style_consistency": 0.85,
            "context_relevance": 0.9
        }
    )

class Metadata(BaseModel):
    generated_at: datetime = Field(default_factory=datetime.now)
    dataset_type: str
    requirements_summary: str
    domain_specific_metadata: Dict[str, Any] = Field(default_factory=dict)

class Configuration(BaseModel):
    sampling_parameters: SamplingParameters
    prompt_structure: PromptStructure
    required_criteria: List[str] = Field(default_factory=list)
    advanced_options: AdvancedOptions
    dataset_parameters: DatasetParameters
    quality_controls: Optional[QualityControls] = None
    diversity_parameters: Optional[DiversityParameters] = None
    evaluation_metrics: Optional[EvaluationMetrics] = None
    metadata: Optional[Metadata] = None
    example_inputs: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True 