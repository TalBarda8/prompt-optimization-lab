# Cost Analysis & Budget Management

**Project:** Prompt Optimization & Evaluation System
**Author:** Tal Barda
**Course:** LLMs in Multi-Agent Environments
**Assignment:** #6 - Prompt Engineering Optimization
**Document Version:** 1.0
**Last Updated:** December 13, 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Token Usage Analysis](#2-token-usage-analysis)
3. [Cost Breakdown by Provider](#3-cost-breakdown-by-provider)
4. [Experiment Cost Summary](#4-experiment-cost-summary)
5. [Budget Optimization Strategies](#5-budget-optimization-strategies)
6. [ROI Analysis](#6-roi-analysis)
7. [Cost Projection for Production](#7-cost-projection-for-production)
8. [Monitoring & Tracking](#8-monitoring--tracking)
9. [Recommendations](#9-recommendations)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Total Project Costs

**Experiment Phase:**
- **Total API Calls:** 660 (110 samples × 6 techniques)
- **Total Input Tokens:** ~103,000 tokens
- **Total Output Tokens:** ~215,000 tokens
- **Total Tokens:** ~318,000 tokens

**Estimated Costs by Provider:**

| Provider | Model | Input Cost | Output Cost | Total Cost |
|----------|-------|------------|-------------|------------|
| **OpenAI** | GPT-4 Turbo | $1.03 | $3.23 | **$4.26** |
| **OpenAI** | GPT-3.5 Turbo | $0.05 | $0.11 | **$0.16** |
| **Anthropic** | Claude 3.5 Sonnet | $0.31 | $0.65 | **$0.96** |
| **Anthropic** | Claude 3 Haiku | $0.03 | $0.04 | **$0.07** |
| **Ollama** | Llama 3.2 (Local) | $0.00 | $0.00 | **$0.00** ✅ |

**Actual Implementation:** Used **Ollama (local)** → **$0 API cost**

### 1.2 Key Findings

1. **Local deployment saved ~$0.96-4.26** compared to commercial APIs
2. **ReAct technique** has highest cost/query but best quality
3. **Role-Based** offers best cost-efficiency ratio (4× better than ReAct)
4. **Fast Mode** reduces costs by **75-80%** with minimal quality loss
5. **Production scale (1M queries):** $1,450-$40,000 depending on technique and provider

---

## 2. Token Usage Analysis

### 2.1 Token Consumption by Technique

Based on experimental results with 110 test samples:

| Technique | Avg Input Tokens | Avg Output Tokens | Avg Total Tokens | vs Baseline |
|-----------|------------------|-------------------|------------------|-------------|
| **Baseline** | 45.2 | 111.0 | **156.2** | — |
| **Role-Based** | 87.3 | 111.0 | **198.3** | +27% |
| **Few-Shot** | 156.9 | 111.0 | **267.9** | +72% |
| **CoT** | 201.7 | 111.0 | **312.7** | +100% |
| **ReAct** | 278.4 | 111.0 | **389.4** | +149% |
| **CoT++** | 317.3 | 111.0 | **428.3** | +174% |
| **ToT** | 401.8 | 111.0 | **512.8** | +228% |

**Insights:**

- **Output tokens remain constant** (~111 tokens) - answer length doesn't vary much
- **Input tokens scale dramatically** - complex prompts add significant overhead
- **ToT uses 3.3× more tokens** than baseline for same accuracy
- **Diminishing returns** - ToT's 228% token increase yields only 17.5% quality improvement

### 2.2 Token Distribution Analysis

**Breakdown of Total Experiment Tokens (318K):**

```
Input Tokens (32.4%):   ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 103,086
Output Tokens (67.6%):  ████████████████████████████████████░░░░░░░ 214,914
```

**By Technique (Total Tokens):**

| Technique | Total Tokens | Percentage | Cumulative |
|-----------|--------------|------------|------------|
| ToT | 56,408 | 17.7% | 17.7% |
| CoT++ | 47,113 | 14.8% | 32.5% |
| ReAct | 42,834 | 13.5% | 46.0% |
| CoT | 34,397 | 10.8% | 56.8% |
| Few-Shot | 29,469 | 9.3% | 66.1% |
| Role-Based | 21,813 | 6.9% | 73.0% |
| Baseline | 17,182 | 5.4% | 78.4% |
| *Fast Mode (all)* | 68,784 | 21.6% | 100.0% |

**Key Observation:** The top 3 techniques (ToT, CoT++, ReAct) consume 46% of all tokens despite representing only 3/7 techniques.

### 2.3 Token Efficiency Metrics

**Quality per Token (Lower Loss per Token is Better):**

| Technique | Loss | Avg Tokens | Quality/Token (×10³) |
|-----------|------|------------|----------------------|
| **ReAct** | 1.326 | 389.4 | **3.40** ⭐ |
| **CoT** | 1.395 | 312.7 | **4.46** |
| **Role-Based** | 1.489 | 198.3 | **7.51** 💰 |
| CoT++ | 1.368 | 428.3 | 3.19 |
| ToT | 1.342 | 512.8 | 2.62 |
| Few-Shot | 1.412 | 267.9 | 5.27 |
| Baseline | 1.627 | 156.2 | 10.41 |

**Interpretation:**
- **ReAct**: Best quality-per-token ratio among advanced techniques
- **Role-Based**: 2.2× better efficiency than ReAct, great budget choice
- **ToT**: Poorest efficiency - high cost for marginal quality gain

---

## 3. Cost Breakdown by Provider

### 3.1 OpenAI Pricing (December 2025)

**GPT-4 Turbo (gpt-4-turbo-preview):**
- Input: $0.01 / 1K tokens
- Output: $0.03 / 1K tokens

**GPT-3.5 Turbo (gpt-3.5-turbo):**
- Input: $0.0005 / 1K tokens
- Output: $0.0015 / 1K tokens

**Cost Calculation (Full Experiment - 318K tokens):**

```
GPT-4 Turbo:
  Input:  103,086 tokens × $0.01 / 1,000 = $1.03
  Output: 214,914 tokens × $0.03 / 1,000 = $6.45
  Total: $7.48

GPT-3.5 Turbo:
  Input:  103,086 tokens × $0.0005 / 1,000 = $0.05
  Output: 214,914 tokens × $0.0015 / 1,000 = $0.32
  Total: $0.37
```

### 3.2 Anthropic Pricing (December 2025)

**Claude 3.5 Sonnet:**
- Input: $0.003 / 1K tokens
- Output: $0.015 / 1K tokens

**Claude 3 Haiku (most economical):**
- Input: $0.00025 / 1K tokens
- Output: $0.00125 / 1K tokens

**Cost Calculation (Full Experiment - 318K tokens):**

```
Claude 3.5 Sonnet:
  Input:  103,086 tokens × $0.003 / 1,000 = $0.31
  Output: 214,914 tokens × $0.015 / 1,000 = $3.22
  Total: $3.53

Claude 3 Haiku:
  Input:  103,086 tokens × $0.00025 / 1,000 = $0.03
  Output: 214,914 tokens × $0.00125 / 1,000 = $0.27
  Total: $0.30
```

### 3.3 Ollama (Local Deployment)

**Hardware Requirements:**
- **Model:** Llama 3.2 (3B parameters)
- **RAM:** 8 GB minimum
- **Storage:** ~2 GB for model weights
- **GPU:** Optional (3-5× speedup with CUDA)

**Direct Costs:**
- **API Fees:** $0.00
- **Electricity:** ~$0.002-0.005 per experiment run (~1 hour @ 200W GPU)

**Indirect Costs:**
- **Initial Setup:** 2-4 hours (one-time)
- **Maintenance:** ~1 hour/month

**Total Estimated Cost (Experiment):** $0.00 - $0.01

**Advantages:**
- ✅ No per-token costs
- ✅ Unlimited experimentation
- ✅ Data privacy (no external API)
- ✅ No rate limits
- ✅ Offline capability

**Disadvantages:**
- ❌ Lower quality than GPT-4/Claude (for some tasks)
- ❌ Hardware requirements
- ❌ Maintenance overhead

---

## 4. Experiment Cost Summary

### 4.1 Cost Per Technique (GPT-4 Turbo Pricing)

| Technique | Queries | Avg Tokens | Total Tokens | Input Cost | Output Cost | Total Cost |
|-----------|---------|------------|--------------|------------|-------------|------------|
| ToT | 110 | 512.8 | 56,408 | $0.18 | $1.02 | **$1.20** |
| CoT++ | 110 | 428.3 | 47,113 | $0.15 | $0.85 | **$1.00** |
| ReAct | 110 | 389.4 | 42,834 | $0.14 | $0.77 | **$0.91** |
| CoT | 110 | 312.7 | 34,397 | $0.11 | $0.62 | **$0.73** |
| Few-Shot | 110 | 267.9 | 29,469 | $0.10 | $0.53 | **$0.63** |
| Role-Based | 110 | 198.3 | 21,813 | $0.07 | $0.39 | **$0.46** |
| Baseline | 110 | 156.2 | 17,182 | $0.06 | $0.31 | **$0.37** |

**Total Experiment Cost (GPT-4 Turbo):** $5.30

### 4.2 Cost Per Query (Average)

| Provider | Model | Cost/Query (Baseline) | Cost/Query (ReAct) | Cost/Query (ToT) |
|----------|-------|----------------------|-------------------|------------------|
| OpenAI | GPT-4 Turbo | $0.0034 | $0.0083 | $0.0109 |
| OpenAI | GPT-3.5 Turbo | $0.0002 | $0.0004 | $0.0005 |
| Anthropic | Claude 3.5 Sonnet | $0.0019 | $0.0046 | $0.0060 |
| Anthropic | Claude 3 Haiku | $0.0002 | $0.0004 | $0.0005 |
| Ollama | Llama 3.2 | $0.0000 | $0.0000 | $0.0000 |

### 4.3 Cumulative Cost Over Time

**Scenario: Academic Research Project (1,000 queries/month)**

| Month | Baseline | CoT | ReAct | ToT | Total |
|-------|----------|-----|-------|-----|-------|
| Month 1 | $3.40 | $7.30 | $8.30 | $10.90 | $30 |
| Month 2 | $6.80 | $14.60 | $16.60 | $21.80 | $60 |
| Month 3 | $10.20 | $21.90 | $24.90 | $32.70 | $90 |
| **Year 1** | **$40.80** | **$87.60** | **$99.60** | **$130.80** | **$359** |

Using **GPT-4 Turbo** pricing.

---

## 5. Budget Optimization Strategies

### 5.1 Strategy 1: Fast Mode Implementation

**Concept:** Use minimal prompts for simple queries, full prompts only when needed.

**Implementation:**
```python
def optimize_prompt(question, quality_threshold=0.9):
    # Try fast mode first
    result = generate(question, technique=CoT, fast_mode=True)

    if result.confidence > quality_threshold:
        return result  # Fast mode sufficient
    else:
        # Retry with standard mode
        return generate(question, technique=CoT, fast_mode=False)
```

**Expected Savings:**

| Metric | Standard Mode | Fast Mode | Savings |
|--------|---------------|-----------|---------|
| Avg Tokens | 324.7 | 78.3 | **75.9%** |
| Cost/Query (GPT-4) | $0.0065 | $0.0016 | **75.4%** |
| Monthly (1K queries) | $6.50 | $1.60 | **$4.90** |

**Annual Savings (1,000 queries/month):** ~$59/year per technique

---

### 5.2 Strategy 2: Technique Selection by Complexity

**Concept:** Route queries to appropriate technique based on complexity.

**Classification Rules:**

```python
def select_technique(question):
    complexity = assess_complexity(question)

    if complexity == "simple":
        return Baseline  # Factual, arithmetic
    elif complexity == "moderate":
        return CoT  # Multi-step reasoning
    elif complexity == "complex":
        return ReAct  # Advanced problems
    else:
        return ToT  # Strategic decisions only
```

**Expected Distribution (typical application):**
- Simple: 60% → Baseline (156 tokens/query)
- Moderate: 30% → CoT (313 tokens/query)
- Complex: 9% → ReAct (389 tokens/query)
- Strategic: 1% → ToT (513 tokens/query)

**Weighted Average:** 213 tokens/query (vs 324 for uniform CoT)

**Savings:** 34% reduction in token usage

---

### 5.3 Strategy 3: Caching & Deduplication

**Concept:** Cache responses for identical or similar queries.

**Implementation:**

```python
from functools import lru_cache
import hashlib

def cache_key(question, technique):
    return hashlib.md5(f"{question}|{technique}".encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_generate(question, technique):
    return llm_client.generate(question, technique)
```

**Expected Hit Rate:** 15-30% for typical applications

**Savings:**
- 20% hit rate → 20% cost reduction
- Monthly (1K queries, CoT): $7.30 → $5.84 (-$1.46/month)

---

### 5.4 Strategy 4: Hybrid Local + Cloud

**Concept:** Use local Ollama for development/testing, cloud APIs for production.

**Implementation:**

```python
if environment == "development":
    client = OllamaClient()  # $0 cost
elif environment == "production":
    client = OpenAIClient(model="gpt-3.5-turbo")  # Low cost
```

**Cost Breakdown:**

| Phase | Queries | Provider | Cost |
|-------|---------|----------|------|
| Development (70%) | 700 | Ollama | $0.00 |
| Testing (20%) | 200 | GPT-3.5 | $0.13 |
| Production (10%) | 100 | GPT-4 | $0.65 |
| **Total** | **1,000** | **Mixed** | **$0.78** |

**Savings vs Pure GPT-4:** 88% reduction ($6.50 → $0.78)

---

### 5.5 Strategy 5: Batch Processing

**Concept:** Batch similar queries to reduce overhead.

**API Batch Pricing (OpenAI):**
- Standard API: $0.03/1K output tokens
- Batch API (50% discount): $0.015/1K output tokens

**Implementation:**
```python
# Batch 100 queries together
batch_results = openai.Batch.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

**Savings:** 50% on output tokens (input tokens unchanged)

**Monthly Savings (1K queries, ReAct):**
- Standard: $8.30
- Batch: $5.90 (-$2.40/month)

---

## 6. ROI Analysis

### 6.1 Quality Improvement vs Cost Increase

**Baseline (Loss = 1.627) as Reference:**

| Technique | Loss Reduction | Quality Gain | Token Increase | Cost Increase | ROI Score |
|-----------|----------------|--------------|----------------|---------------|-----------|
| **Role-Based** | -8.5% | 8.5% | +27% | +27% | **0.31** 🥇 |
| **CoT** | -14.3% | 14.3% | +100% | +100% | **0.14** 🥈 |
| **Few-Shot** | -13.2% | 13.2% | +72% | +72% | **0.18** 🥉 |
| ReAct | -18.5% | 18.5% | +149% | +149% | 0.12 |
| CoT++ | -15.9% | 15.9% | +174% | +174% | 0.09 |
| ToT | -17.5% | 17.5% | +228% | +228% | 0.08 |

**ROI Score Formula:** Quality Gain ÷ Cost Increase

**Insights:**
- **Role-Based** provides best return on investment (3× better ROI than ReAct)
- **CoT** offers balanced quality and cost
- **ToT** has poorest ROI despite good quality

### 6.2 Break-Even Analysis

**Question:** When does ReAct's superior quality justify its 149% cost premium?

**Scenario:** Financial trading application where decisions impact revenue.

**Assumptions:**
- Avg decision value: $1,000
- Baseline error rate: 5%
- ReAct error rate: 2.5% (hypothetical)

**Expected Value:**

```
Baseline:
  Correct decisions (95%): $1,000 × 0.95 = $950
  Wrong decisions (5%): -$200 × 0.05 = -$10
  Net EV: $940
  Cost: $0.0034/query
  Profit: $939.997

ReAct:
  Correct decisions (97.5%): $1,000 × 0.975 = $975
  Wrong decisions (2.5%): -$200 × 0.025 = -$5
  Net EV: $970
  Cost: $0.0083/query
  Profit: $969.992

Additional Profit: $30 per query
ROI: 3,614× return on $0.0083 investment
```

**Conclusion:** In high-stakes applications, ReAct's quality premium easily justifies cost.

### 6.3 Use Case Recommendations

**Budget-Conscious (≤$50/month):**
- **Primary:** Role-Based (70% of queries)
- **Secondary:** CoT (25%)
- **Rare:** ReAct (5% for critical queries)
- **Provider:** GPT-3.5 Turbo or Claude Haiku

**Balanced ($50-200/month):**
- **Primary:** CoT (60%)
- **Secondary:** ReAct (30%)
- **Rare:** ToT (10% for strategic)
- **Provider:** Claude 3.5 Sonnet

**Quality-Critical ($200+/month):**
- **Primary:** ReAct (50%)
- **Secondary:** CoT++ (30%)
- **Strategic:** ToT (20%)
- **Provider:** GPT-4 Turbo

---

## 7. Cost Projection for Production

### 7.1 Scale Estimates

**Production Scenarios:**

| Scale | Queries/Month | Technique | Provider | Monthly Cost | Annual Cost |
|-------|---------------|-----------|----------|--------------|-------------|
| **Small** | 1,000 | CoT | GPT-3.5 | $0.20 | $2.40 |
| **Medium** | 10,000 | CoT | GPT-3.5 | $2.00 | $24.00 |
| **Medium** | 10,000 | ReAct | Claude Sonnet | $46.00 | $552.00 |
| **Large** | 100,000 | CoT | GPT-4 | $730.00 | $8,760.00 |
| **Large** | 100,000 | ReAct | GPT-4 | $830.00 | $9,960.00 |
| **Enterprise** | 1,000,000 | CoT | GPT-3.5 | $2,000.00 | $24,000.00 |
| **Enterprise** | 1,000,000 | ReAct | GPT-4 | $8,300.00 | $99,600.00 |

### 7.2 Budget Planning Template

**Monthly Budget Allocation:**

```
Total Budget: $500/month

Distribution:
├─ Development & Testing (20%): $100
│  └─ Use Ollama (free) + occasional API tests
├─ Production Queries (60%): $300
│  ├─ Simple (50% of prod): $75 (GPT-3.5 Baseline)
│  ├─ Moderate (35% of prod): $105 (GPT-3.5 CoT)
│  └─ Complex (15% of prod): $120 (Claude ReAct)
├─ Monitoring & Analytics (10%): $50
│  └─ Usage tracking, error analysis
└─ Reserve Buffer (10%): $50
   └─ Unexpected spikes, experimentation
```

**Expected Query Volume:** ~35,000 queries/month

### 7.3 Cost Control Mechanisms

**1. Rate Limiting:**
```python
# Limit to 1,000 queries/day
if daily_count > 1000:
    switch_to_queuing_mode()
```

**2. Budget Alerts:**
```python
# Alert when 80% of budget consumed
if monthly_spend > BUDGET * 0.8:
    send_alert(team, "Budget threshold reached")
```

**3. Auto-Downgrade:**
```python
# Switch to cheaper model when budget low
if monthly_spend > BUDGET * 0.9:
    model = "gpt-3.5-turbo"  # Downgrade from GPT-4
```

---

## 8. Monitoring & Tracking

### 8.1 Key Metrics to Track

**Real-Time Metrics:**
1. **Tokens per request** (input + output)
2. **Cost per request**
3. **Requests per hour/day/month**
4. **Cumulative monthly spend**
5. **Budget utilization percentage**

**Aggregated Analytics:**
1. **Cost by technique**
2. **Cost by user/team**
3. **Cost trends over time**
4. **Token efficiency (quality/token)**
5. **Error rate vs cost**

### 8.2 Monitoring Implementation

**Token Usage Logging:**

```python
import logging
from datetime import datetime

def log_token_usage(request_id, technique, input_tokens, output_tokens, cost):
    """Log token usage for cost tracking."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "technique": technique,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": cost,
    }

    # Write to database or log file
    cost_logger.info(json.dumps(log_entry))

    # Update cumulative counters
    update_monthly_totals(technique, input_tokens, output_tokens, cost)
```

**Dashboard Queries:**

```sql
-- Monthly cost by technique
SELECT
    technique,
    SUM(total_tokens) AS total_tokens,
    SUM(estimated_cost) AS total_cost,
    COUNT(*) AS num_requests,
    AVG(estimated_cost) AS avg_cost_per_request
FROM token_usage_logs
WHERE timestamp >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY technique
ORDER BY total_cost DESC;
```

### 8.3 Alerting Rules

**Budget Alerts:**

```python
# Alert thresholds
BUDGET_ALERTS = {
    0.5: "info",     # 50% consumed
    0.75: "warning", # 75% consumed
    0.9: "critical", # 90% consumed
    1.0: "emergency" # Budget exceeded
}

def check_budget_alerts():
    utilization = monthly_spend / MONTHLY_BUDGET

    for threshold, severity in BUDGET_ALERTS.items():
        if utilization >= threshold and not alerted[threshold]:
            send_alert(
                severity=severity,
                message=f"Budget {utilization:.0%} consumed (${monthly_spend:.2f}/${MONTHLY_BUDGET:.2f})"
            )
            alerted[threshold] = True
```

---

## 9. Recommendations

### 9.1 For This Project (Academic)

✅ **Current Approach (Ollama Local):** Optimal choice
- Zero API costs
- Unlimited experimentation
- Full data privacy
- Sufficient quality for research purposes

**Recommendation:** Continue with Ollama for all academic work.

### 9.2 For Production Deployment

**Phase 1 - MVP (Months 1-3):**
- **Primary:** GPT-3.5 Turbo with CoT (balance of cost/quality)
- **Budget:** $50-100/month (covers ~5,000-10,000 queries)
- **Focus:** Validate product-market fit before scaling costs

**Phase 2 - Growth (Months 4-12):**
- **Hybrid:** GPT-3.5 (simple) + Claude Sonnet (complex)
- **Budget:** $200-500/month (20,000-50,000 queries)
- **Optimization:** Implement Fast Mode, caching, technique routing

**Phase 3 - Scale (Year 2+):**
- **Enterprise:** Custom model fine-tuning or self-hosted
- **Budget:** $1,000-5,000/month (100,000-500,000 queries)
- **Strategy:** Negotiate volume discounts, consider Azure/GCP credits

### 9.3 Cost-Saving Checklist

- [x] Use local models (Ollama) for development
- [ ] Implement Fast Mode for 60%+ of queries
- [ ] Add caching for duplicate/similar queries (15-30% savings)
- [ ] Route simple queries to baseline/role-based techniques
- [ ] Use batch API for non-time-sensitive requests (50% savings)
- [ ] Monitor token usage and set budget alerts
- [ ] Compress prompts (remove verbose examples)
- [ ] Use cheaper models (GPT-3.5, Claude Haiku) where acceptable
- [ ] Negotiate volume discounts with providers
- [ ] Consider reserved capacity for predictable workloads

---

## 10. Appendices

### Appendix A: Pricing Reference (December 2025)

**OpenAI:**
| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| GPT-4 Turbo | $10.00 | $30.00 |
| GPT-4 | $30.00 | $60.00 |
| GPT-3.5 Turbo | $0.50 | $1.50 |

**Anthropic:**
| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| Claude 3 Opus | $15.00 | $75.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |

**Google (Vertex AI):**
| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| Gemini 1.5 Pro | $3.50 | $10.50 |
| Gemini 1.5 Flash | $0.35 | $1.05 |

---

### Appendix B: Token Counting Tools

**Python Implementation:**

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Input text
        model: Model name (for tokenizer selection)

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# Example usage
prompt = "Explain quantum computing in simple terms."
token_count = count_tokens(prompt)
print(f"Tokens: {token_count}")
```

**Online Tools:**
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- Tiktoken Playground: https://tiktokenizer.vercel.app/

---

### Appendix C: Cost Calculation Formulas

**Total Cost Formula:**

```
Total Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)

where:
  Input Tokens = Σ(system_prompt_tokens + user_prompt_tokens)
  Output Tokens = Σ(response_tokens)
  Prices are in $/1K tokens or $/1M tokens
```

**Cost per Query:**

```
Cost/Query = Total Cost ÷ Number of Queries
```

**Monthly Projection:**

```
Monthly Cost = Cost/Query × Expected Monthly Queries
```

**ROI Score:**

```
ROI = Quality Improvement (%) ÷ Cost Increase (%)

where:
  Quality Improvement = (Baseline Loss - Technique Loss) / Baseline Loss
  Cost Increase = (Technique Tokens - Baseline Tokens) / Baseline Tokens
```

---

### Appendix D: Budget Template (Spreadsheet)

```csv
Technique,Avg_Input_Tokens,Avg_Output_Tokens,Total_Tokens,Queries,Input_Cost,Output_Cost,Total_Cost
Baseline,45.2,111.0,156.2,110,$0.45,$1.11,$1.56
Role-Based,87.3,111.0,198.3,110,$0.87,$1.11,$1.98
CoT,201.7,111.0,312.7,110,$2.02,$1.11,$3.13
ReAct,278.4,111.0,389.4,110,$2.78,$1.11,$3.89
CoT++,317.3,111.0,428.3,110,$3.17,$1.11,$4.28
ToT,401.8,111.0,512.8,110,$4.02,$1.11,$5.13
Few-Shot,156.9,111.0,267.9,110,$1.57,$1.11,$2.68
```

*(Prices shown for GPT-4 Turbo @ $10/1M input, $30/1M output)*

---

## References

1. **OpenAI Pricing:** https://openai.com/pricing
2. **Anthropic Pricing:** https://www.anthropic.com/pricing
3. **Tiktoken Documentation:** https://github.com/openai/tiktoken
4. **Token Optimization Guide:** https://platform.openai.com/docs/guides/optimization
5. **Batch API Documentation:** https://platform.openai.com/docs/guides/batch

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-13 | Initial cost analysis and budget guide | Tal Barda |

---

**End of Cost Analysis & Budget Management Documentation**
