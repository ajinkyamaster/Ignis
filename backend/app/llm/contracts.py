from typing import Literal
from pydantic import BaseModel, Field, model_validator

class NumberedFeature(BaseModel):
    id: int = Field(ge=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

class AnalyzerOutput(BaseModel):
    problem: str = Field(min_length=1)
    target_users: str = Field(min_length=1)
    core_workflow: str = Field(min_length=5, pattern=r'.+→.+→.+',
        description='A single line with at least two literal Unicode arrows, e.g. video → transcription and concept extraction → flashcards')
    features: list[NumberedFeature] = Field(min_length=1, max_length=12)
    synonyms: list[str]
    github_queries: list[str] = Field(min_length=4, max_length=6)

    @model_validator(mode='after')
    def contiguous(self):
        if [f.id for f in self.features] != list(range(1, len(self.features) + 1)):
            raise ValueError('feature IDs must be contiguous from 1')
        if '→' not in self.core_workflow:
            raise ValueError('workflow must use arrows')
        return self

class CandidateJudgment(BaseModel):
    candidate_id: str
    label: Literal['Strong', 'Related', 'No match']
    confidence: Literal['High', 'Medium', 'Low']
    one_line_reason: str
    covered_feature_ids: list[int]
    addons: list[str]

class ReviewerBatchOutput(BaseModel):
    judgments: list[CandidateJudgment]

class AddonCluster(BaseModel):
    theme: str
    features: list[str]
    seen_in_projects: list[str]

class SynthesizerOutput(BaseModel):
    addon_clusters: list[AddonCluster]
    verdict_direction: Literal['Proceed', 'Differentiate', 'Build on Existing']
    crowdedness_summary: str
    differentiation_strategy: list[str]

class ExplainerOutput(BaseModel):
    scope_disclaimer: str
    workflow_explanation: str
    core_concepts: list[str]
    folder_roles: list[str]
