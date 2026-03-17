"""Cost tracking for API usage during annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Pricing per 1M tokens (Standard tier)
GPT_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.00},
    "gpt-5.1-2025-11-13": {"input": 1.25, "output": 10.00},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.00},
    "gpt-5-nano-2025-08-07": {"input": 0.15, "output": 0.60},  # GPT-5 Nano is cheaper than mini
}

CLAUDE_PRICING = {
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    image_tokens: int = 0  # Estimated image tokens (for GPT vision)
    
    def total_input(self) -> int:
        return self.input_tokens + self.image_tokens


@dataclass
class CostTracker:
    """Track API costs across annotation runs."""
    
    gpt_calls: int = 0
    claude_calls: int = 0
    qwen_calls: int = 0
    template_calls: int = 0
    
    gpt_tokens: Dict[str, TokenUsage] = field(default_factory=dict)  # model -> usage
    claude_tokens: Dict[str, TokenUsage] = field(default_factory=dict)  # model -> usage
    
    def add_gpt_usage(self, model: str, input_tokens: int, output_tokens: int, image_tokens: int = 0):
        """Add GPT API usage."""
        self.gpt_calls += 1
        if model not in self.gpt_tokens:
            self.gpt_tokens[model] = TokenUsage()
        self.gpt_tokens[model].input_tokens += input_tokens
        self.gpt_tokens[model].output_tokens += output_tokens
        self.gpt_tokens[model].image_tokens += image_tokens
    
    def add_claude_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add Claude API usage."""
        self.claude_calls += 1
        if model not in self.claude_tokens:
            self.claude_tokens[model] = TokenUsage()
        self.claude_tokens[model].input_tokens += input_tokens
        self.claude_tokens[model].output_tokens += output_tokens
    
    def add_qwen_call(self):
        """Track Qwen call (local, no API cost)."""
        self.qwen_calls += 1
    
    def add_template_call(self):
        """Track template call (no API cost)."""
        self.template_calls += 1
    
    def calculate_cost(self) -> Dict[str, float]:
        """Calculate total cost in USD."""
        total = 0.0
        breakdown = {}
        
        # GPT costs
        for model, usage in self.gpt_tokens.items():
            pricing = GPT_PRICING.get(model, GPT_PRICING.get("gpt-4o"))
            input_cost = (usage.total_input() / 1_000_000) * pricing["input"]
            output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
            model_cost = input_cost + output_cost
            total += model_cost
            breakdown[f"GPT-{model}"] = model_cost
        
        # Claude costs
        for model, usage in self.claude_tokens.items():
            pricing = CLAUDE_PRICING.get(model, CLAUDE_PRICING.get("claude-3-7-sonnet-20250219"))
            input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
            output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]
            model_cost = input_cost + output_cost
            total += model_cost
            breakdown[f"Claude-{model}"] = model_cost
        
        breakdown["TOTAL"] = total
        return breakdown
    
    def get_summary(self) -> str:
        """Get formatted cost summary."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("COST SUMMARY")
        lines.append("=" * 70)
        
        # API calls
        lines.append(f"\nAPI Calls:")
        lines.append(f"  GPT:     {self.gpt_calls}")
        lines.append(f"  Claude:  {self.claude_calls}")
        lines.append(f"  Qwen:    {self.qwen_calls} (local, no cost)")
        lines.append(f"  Template:{self.template_calls} (no cost)")
        
        # Token usage
        if self.gpt_tokens:
            lines.append(f"\nGPT Token Usage:")
            for model, usage in self.gpt_tokens.items():
                lines.append(f"  {model}:")
                lines.append(f"    Input:  {usage.input_tokens:,} tokens")
                lines.append(f"    Images: {usage.image_tokens:,} tokens")
                lines.append(f"    Output: {usage.output_tokens:,} tokens")
                lines.append(f"    Total:  {usage.total_input() + usage.output_tokens:,} tokens")
        
        if self.claude_tokens:
            lines.append(f"\nClaude Token Usage:")
            for model, usage in self.claude_tokens.items():
                lines.append(f"  {model}:")
                lines.append(f"    Input:  {usage.input_tokens:,} tokens")
                lines.append(f"    Output: {usage.output_tokens:,} tokens")
                lines.append(f"    Total:  {usage.input_tokens + usage.output_tokens:,} tokens")
        
        # Costs
        costs = self.calculate_cost()
        lines.append(f"\nCost Breakdown:")
        for key, cost in costs.items():
            if key != "TOTAL":
                lines.append(f"  {key}: ${cost:.4f}")
        lines.append(f"\n  {'TOTAL COST':<20} ${costs['TOTAL']:.4f}")
        
        # Per-image estimate
        total_calls = self.gpt_calls + self.claude_calls
        if total_calls > 0:
            avg_cost = costs["TOTAL"] / total_calls
            lines.append(f"\n  Average per image: ${avg_cost:.4f}")
        
        lines.append("=" * 70 + "\n")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all counters."""
        self.gpt_calls = 0
        self.claude_calls = 0
        self.qwen_calls = 0
        self.template_calls = 0
        self.gpt_tokens.clear()
        self.claude_tokens.clear()
