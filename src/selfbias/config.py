"""Experiment configuration schemas + loaders.

Everything experiment-shaped is a field here and comes from YAML - nothing is hardcoded
in pipeline code. The pipeline must handle N models, N domains, N length bins, N
disclosure arms, N paradigms (Phase discipline: later phases are config changes).

Secrets live in ``.env`` and are read via :class:`Settings` (pydantic-settings). The
program starts and the dashboard renders with missing keys - see ``keys_present``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schemas import Disclosure, Paradigm, Provider, ReferenceSource, stable_hash

# ---------------------------------------------------------------------------
# Secrets (never logged, never committed)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """API keys from ``.env`` / environment. Optional so the app runs keyless.

    Keys are resolved by *environment-variable name* so any provider/key works - a model
    can name its own ``api_key_env`` (e.g. ``OPENROUTER_API_KEY``). We load ``.env`` into
    the process environment so those arbitrary names resolve too.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def __init__(self, **kwargs):
        # Populate os.environ from .env so arbitrary key names (not just the three
        # declared below) resolve via resolve_key(). Existing env vars win.
        try:
            from dotenv import load_dotenv

            load_dotenv(".env", override=False)
        except Exception:  # pragma: no cover - dotenv always present via deps
            pass
        super().__init__(**kwargs)

    # Declared for convenience/back-compat; resolution below is by env-var name.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    def resolve_key(self, provider: Provider, api_key_env: str | None = None) -> str | None:
        """Resolve a model's API key. ``mock`` never needs one; ``openai_compatible``
        may legitimately have none (e.g. a local Ollama server)."""

        if provider == Provider.mock:
            return "mock-key"
        env_name = api_key_env or DEFAULT_KEY_ENV.get(provider)
        if env_name:
            val = os.getenv(env_name)
            if val:
                return val
        # Local OpenAI-compatible servers often need no key.
        return None

    def key_status(self, roster: list[RosterModel]) -> list[dict]:
        """Per-model key presence, for the dashboard/CLI 'keys' view."""

        out = []
        for m in roster:
            needs = m.provider not in (Provider.mock, Provider.openai_compatible)
            present = bool(self.resolve_key(m.provider, m.key_env()))
            out.append(
                {
                    "slot": m.slot,
                    "model": m.model,
                    "provider": m.provider.value,
                    "key_env": m.key_env(),
                    "present": present,
                    "required": needs,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Experiment config (mirrors config/experiment.example.yaml 1:1)
# ---------------------------------------------------------------------------


class RunMeta(BaseModel):
    name: str
    seed: int
    budget_usd: float = Field(gt=0)
    phase: int = 0


DEFAULT_KEY_ENV: dict[Provider, str] = {
    Provider.anthropic: "ANTHROPIC_API_KEY",
    Provider.google: "GOOGLE_API_KEY",
    Provider.openai: "OPENAI_API_KEY",
}


class RosterModel(BaseModel):
    """One model in the roster. Any N >= 2, any mix of providers and families.

    ``provider`` is the adapter kind; ``family`` is a free-form label used only to group
    siblings for HSPP-R_fam (e.g. "anthropic", "qwen", "llama"). ``base_url`` targets a
    custom OpenAI-compatible endpoint (Ollama/vLLM/OpenRouter/Together/...). ``api_key_env``
    names the environment variable holding this model's key; it defaults per provider.
    """

    slot: str  # short unique label for this roster entry (e.g. "judge_a", "qwen")
    provider: Provider
    model: str
    family: str
    base_url: str | None = None
    api_key_env: str | None = None

    def key_env(self) -> str | None:
        if self.api_key_env:
            return self.api_key_env
        return DEFAULT_KEY_ENV.get(self.provider)

    @model_validator(mode="after")
    def _endpoint_sane(self) -> RosterModel:
        if self.provider == Provider.openai_compatible and not self.base_url:
            raise ValueError(
                f"roster model '{self.slot}' uses provider 'openai_compatible' and must "
                "set base_url (e.g. http://localhost:11434/v1 for Ollama)"
            )
        return self


class RubricSpec(BaseModel):
    source: str = "llm_drafted_human_reviewed"
    per_task_min: int = 4
    per_task_max: int = 10


class DomainConfig(BaseModel):
    name: str
    n_prompts: int = Field(gt=0)
    reference: ReferenceSource
    rubrics: RubricSpec | None = None


class PromptsConfig(BaseModel):
    """Where task prompts come from.

    ``builtin`` - the curated seed set (default, reproducible, free).
    ``excel``   - a user-supplied ``.xlsx``/``.csv`` prompt library (``excel_path``).
    ``llm_generated`` - a roster model drafts a fresh set (W4; adds cost).
    """

    source: Literal["builtin", "excel", "llm_generated"] = "builtin"
    excel_path: str | None = None
    generator_model: str | None = None  # llm_generated: which roster model drafts prompts
    n_per_domain: int | None = None  # llm_generated: how many prompts per domain

    @model_validator(mode="after")
    def _source_sane(self) -> PromptsConfig:
        if self.source == "excel" and not self.excel_path:
            raise ValueError("prompts.source='excel' requires prompts.excel_path")
        return self


class LengthConfig(BaseModel):
    target_bins_tokens: list[int]
    tolerance_pct: int = 20
    max_retries: int = 3
    truncation_series: bool = True

    @field_validator("target_bins_tokens")
    @classmethod
    def _nonempty_sorted(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("target_bins_tokens must be non-empty")
        return sorted(v)


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    persist_raw: bool = True


class JudgingConfig(BaseModel):
    paradigms: list[Paradigm]
    disclosure_arms: list[Disclosure]
    order_swap_pwc: bool = True
    temperature: float = 0.0
    elicit_confidence: bool = True
    judge_isolation: bool = True

    @model_validator(mode="after")
    def _integrity(self) -> JudgingConfig:
        # Integrity rules that must never be "optimized" away outside explicit studies.
        if Paradigm.pwc in self.paradigms and not self.order_swap_pwc:
            raise ValueError("order_swap_pwc must be true whenever PWC is enabled")
        if not self.judge_isolation:
            raise ValueError("judge_isolation must remain true outside the future contagion study")
        return self


class ProbesConfig(BaseModel):
    pairwise_recognition: bool = True
    single_recognition: bool = False
    on_truncation_series: bool = True


class DiagnosticsConfig(BaseModel):
    repeatability_sample_pct: float = 2.0
    repeatability_n: int = 5


class AnalysisConfig(BaseModel):
    bootstrap_iterations: int = 1000
    attribution_baseline: str = "tfidf"
    mixed_effects: bool = True


class FingerprintConfig(BaseModel):
    enabled: bool = False
    discovery_pool_prompts: int = 60
    feature_response_types: list[str] = Field(default_factory=list)
    dedup_cosine_threshold: float = 0.85


class ExperimentConfig(BaseModel):
    run: RunMeta
    roster: list[RosterModel]
    domains: list[DomainConfig]
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    lengths: LengthConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judging: JudgingConfig
    probes: ProbesConfig = Field(default_factory=ProbesConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)

    @model_validator(mode="after")
    def _roster_sane(self) -> ExperimentConfig:
        if len(self.roster) < 2:
            raise ValueError(
                "roster needs at least 2 models so a judge always has non-self outputs "
                "to compare against"
            )
        slots = [m.slot for m in self.roster]
        if len(slots) != len(set(slots)):
            raise ValueError("roster slots must be unique")
        # Model string IS the generator/judge identity throughout the pipeline
        # (gen ids, judge cells, lookups). Two slots sharing a model string would
        # collapse into one identity, so require uniqueness.
        models = [m.model for m in self.roster]
        if len(models) != len(set(models)):
            raise ValueError(
                "roster model strings must be unique (model string is the identity used "
                "across generation ids, judge cells, and lookups)"
            )
        # llm_generated: a named generator model must be in the roster.
        gen_model = self.prompts.generator_model
        if self.prompts.source == "llm_generated" and gen_model and gen_model not in models:
            raise ValueError(
                f"prompts.generator_model '{gen_model}' is not in the roster; "
                "it must be one of the roster model strings"
            )
        return self

    def prompt_generator(self) -> RosterModel:
        """The roster model that drafts prompts in llm_generated mode (defaults to the
        first roster entry when ``prompts.generator_model`` is unset)."""

        if self.prompts.generator_model:
            m = self.model_by_string(self.prompts.generator_model)
            if m:
                return m
        return self.roster[0]

    def run_id(self) -> str:
        """Deterministic run id from name + seed + a hash of the frozen config."""

        return stable_hash(self.run.name, self.run.seed, self.model_dump(mode="json"))

    def model_by_string(self, model: str) -> RosterModel | None:
        for m in self.roster:
            if m.model == model:
                return m
        return None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class Pricing(BaseModel):
    default: ModelPrice
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    def price_for(self, model: str) -> tuple[ModelPrice, bool]:
        """Return (price, found). ``found`` is False when falling back to default.

        Any ``mock-*`` model string is treated as free (the mock provider never spends),
        so all-mock Phase 0 / test runs estimate and account at exactly $0.
        """

        if model in self.models:
            return self.models[model], True
        if model.startswith("mock"):
            return ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0), True
        return self.default, False


def load_pricing(path: str | Path) -> Pricing:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Pricing.model_validate(raw)
