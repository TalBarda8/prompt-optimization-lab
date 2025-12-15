# C4 Context Diagram - Prompt Optimization System

```mermaid
C4Context
    title System Context - Prompt Optimization & Evaluation System

    Person(researcher, "Academic Researcher", "Studies prompt engineering techniques and evaluates LLM performance")
    Person(developer, "NLP Engineer", "Optimizes LLM applications using experimental results")

    System(promptSystem, "Prompt Optimization System", "Evaluates & compares 7 prompt engineering techniques using mathematical metrics and statistical validation")

    System_Ext(openai, "OpenAI API", "GPT-4, GPT-3.5-turbo models for text generation")
    System_Ext(anthropic, "Anthropic API", "Claude models for text generation")
    System_Ext(ollama, "Ollama", "Local LLM runtime (llama3.2, phi3, etc.) with auto-download")

    System_Ext(filesystem, "File System", "Dataset storage (JSON), configuration (YAML), results persistence (JSON)")

    Rel(researcher, promptSystem, "Runs experiments, analyzes results")
    Rel(developer, promptSystem, "Uses for prompt optimization research")

    Rel(promptSystem, openai, "Sends prompts, receives completions + logprobs", "HTTPS/REST")
    Rel(promptSystem, anthropic, "Sends prompts, receives completions", "HTTPS/REST")
    Rel(promptSystem, ollama, "Sends prompts, receives completions", "Local subprocess")

    Rel(promptSystem, filesystem, "Reads datasets, writes results/visualizations", "File I/O")

    UpdateRelStyle(promptSystem, openai, $textColor="blue", $lineColor="blue")
    UpdateRelStyle(promptSystem, anthropic, $textColor="green", $lineColor="green")
    UpdateRelStyle(promptSystem, ollama, $textColor="orange", $lineColor="orange")
```

**External Systems:**
- **OpenAI API**: Provides GPT-4 and GPT-3.5-turbo models with log probabilities for information-theoretic metrics
- **Anthropic API**: Provides Claude models (Opus, Sonnet, Haiku)
- **Ollama**: Local LLM runtime with automatic model downloading
- **File System**: Stores JSON datasets (140 samples), YAML configurations, JSON results, and PNG/SVG visualizations

**Users:**
- **Academic Researchers**: Primary users conducting prompt engineering research
- **NLP Engineers**: Secondary users optimizing production LLM applications
