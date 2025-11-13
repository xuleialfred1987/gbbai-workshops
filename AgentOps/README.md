# Azure AI Agent Operations

A comprehensive guide to building, deploying, and operating production-ready AI agents with Azure AI Foundry.

## 📋 Overview

This repository demonstrates enterprise-grade agent operations including governance, monitoring, evaluation, integration, and version management. Each module provides hands-on notebooks with best practices for deploying AI agents at scale.

## 🗂️ Repository Structure

### 📁 `01_agent/` - Agent Foundations
Core agent development patterns and management utilities.

- **`01_foundry_agent.ipynb`** - Create and configure Azure AI Foundry agents
- **`02_agent_manager.ipynb`** - Agent lifecycle management with AgentManager utility
- **`03_agent_db_extended.ipynb`** - Agent metadata storage with Azure Cosmos DB

**Key Concepts**: Agent creation, thread management, function calling, metadata persistence

---

### 📁 `02_governance/` - AI Governance Framework
Operationalize responsible AI principles and compliance requirements.

- **`01_catalog.ipynb`** - Agent catalog and discovery system
- **`02_ethical_responsible.ipynb`** - Bias testing and safety guardrails
- **`03_compliance_regulatory.ipynb`** - Regulatory compliance workflows
- **`04_deployment_monitoring.ipynb`** - Production deployment controls
- **`05_auditability_traceability.ipynb`** - Audit trail generation
- **`06_stakeholder_engagement.ipynb`** - Review and approval processes
- **`07_technical_safeguards.ipynb`** - Content filtering and safety layers

**Artifacts**: `governance_exports/` - Audit logs and compliance evidence

**Key Concepts**: Responsible AI, bias detection, compliance, audit trails, stakeholder reviews

---

### 📁 `03_monitoring/` - Observability & Tracing
Complete observability for agent operations with OpenTelemetry.

- **`01_azure_foundry_tracing.ipynb`** - Automatic tracing with Azure Monitor
- **`02_local_tracing.ipynb`** - Local tracing for development
- **`03_decorator_tracing_walkthrough.ipynb`** - Custom tracing with decorators
- **`tracing_decorators.py`** - Reusable tracing decorator utilities
- **`tracing_utils.py`** - Helper functions for trace analysis

**Key Concepts**: OpenTelemetry, Application Insights, distributed tracing, performance monitoring

---

### 📁 `04_integration/` - External Integrations
Connect agents with external services and knowledge sources.

- **`01_bing_integration.ipynb`** - Web search with Bing Grounding
- **`02_mcp_integration.ipynb`** - Model Context Protocol integration
- **`03_knowledge_base_integration.ipynb`** - RAG with Azure AI Search

**Data**: `data/` - Sample documents for knowledge base integration

**Key Concepts**: Bing Grounding, MCP, RAG, Azure AI Search, vector search

---

### 📁 `05_evaluation/` - Agent Quality Assurance
Comprehensive evaluation frameworks for agent performance.

- **`01_genai_evaluation.ipynb`** - LLM-based quality metrics (relevance, coherence, fluency)
- **`02_simulator_eval.ipynb`** - Conversational simulation testing
- **`03_rag_evaluation.ipynb`** - RAG-specific metrics (groundedness, retrieval quality)
- **`04_agent_evaluation.ipynb`** - End-to-end agent evaluation
- **`05_cloud_based_evaluation.ipynb`** - Cloud-based evaluation at scale

**Data**: `data/` - Evaluation datasets and results (JSONL format)

**Key Concepts**: Quality metrics, simulators, RAG evaluation, cloud evaluation, JSONL datasets

---

### 📁 `06_versioning/` - Version Control & CI/CD
Agent versioning and continuous deployment workflows.

- **`01_agent_versioning.ipynb`** - Version management patterns
- **`02_version_performance_demo.ipynb`** - Performance comparison across versions
- **`agent_version_manager.py`** - Versioning utility class
- **`agent_cicd_cli.py`** - CLI tool for CI/CD pipelines
- **`github_actions_example.yml`** - Sample GitHub Actions workflow

**Config**: `example_config.json`, `example_test_queries.json` - Version configuration samples

**Data**: `data/` - Version performance metrics

**Key Concepts**: Semantic versioning, A/B testing, CI/CD, automated deployment, rollback strategies

---

### 📁 `utils/` - Shared Utilities
Reusable utility modules for all notebooks.

- **`agent_utils.py`** - `AgentManager` class for agent operations
- **`agent_db.py`** - `AgentDB` class for Cosmos DB metadata storage
- **`README_agent_utils.md`** - AgentManager documentation
- **`README_DB.md`** - AgentDB documentation
- **`QUICK_REFERENCE.md`** - Quick reference guide

**Key Concepts**: Reusable components, agent lifecycle management, metadata storage

---

## 🚀 Getting Started

### Prerequisites

1. **Azure Resources**:
   - Azure AI Foundry project
   - Azure OpenAI deployment (GPT-4o or similar)
   - Azure Cosmos DB account (optional, for metadata storage)
   - Application Insights (optional, for monitoring)

2. **Authentication**:
   ```bash
   az login
   ```

3. **Python Environment**:
   ```bash
   pip install azure-ai-projects azure-identity azure-ai-evaluation
   pip install azure-monitor-opentelemetry opentelemetry-api
   pip install python-dotenv pandas
   ```

### Configuration

1. Copy the sample environment file:
   ```bash
   cp .env.sample .env
   ```

2. Edit `.env` with your Azure credentials:
   ```properties
   AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
   AZURE_OPENAI_API_KEY_GPT_4o=your-api-key
   AZURE_OPENAI_ENDPOINT_GPT_4o=https://your-openai.openai.azure.com/
   BING_CONNECTION_ID=/subscriptions/.../connections/your-connection
   ```

3. Start with the foundational notebooks:
   - `01_agent/01_foundry_agent.ipynb` - Learn agent basics
   - `01_agent/02_agent_manager.ipynb` - Use AgentManager utility

---

## 📚 Learning Paths

### 🎯 Path 1: Agent Development Basics
**Goal**: Build and deploy your first agent

1. `01_agent/01_foundry_agent.ipynb` - Agent creation fundamentals
2. `01_agent/02_agent_manager.ipynb` - Lifecycle management
3. `04_integration/01_bing_integration.ipynb` - Add web search
4. `03_monitoring/01_azure_foundry_tracing.ipynb` - Add observability

### 🎯 Path 2: Production Operations
**Goal**: Deploy production-ready agents with governance

1. `01_agent/03_agent_db_extended.ipynb` - Metadata persistence
2. `02_governance/01_catalog.ipynb` - Agent discovery system
3. `02_governance/02_ethical_responsible.ipynb` - Safety guardrails
4. `03_monitoring/01_azure_foundry_tracing.ipynb` - Monitor in production
5. `06_versioning/01_agent_versioning.ipynb` - Version management

### 🎯 Path 3: Quality & Evaluation
**Goal**: Ensure agent quality at scale

1. `05_evaluation/01_genai_evaluation.ipynb` - Basic quality metrics
2. `05_evaluation/02_simulator_eval.ipynb` - Conversation testing
3. `05_evaluation/03_rag_evaluation.ipynb` - RAG-specific evaluation
4. `05_evaluation/05_cloud_based_evaluation.ipynb` - Scale evaluation

### 🎯 Path 4: Advanced Integration
**Goal**: Build sophisticated multi-capability agents

1. `04_integration/01_bing_integration.ipynb` - Web search
2. `04_integration/02_mcp_integration.ipynb` - MCP tools
3. `04_integration/03_knowledge_base_integration.ipynb` - RAG with Azure AI Search
4. `01_agent/01_foundry_agent.ipynb` - Custom function calling

---

## 🤝 Contributing

When adding new notebooks or utilities:

1. Follow the existing structure and naming conventions
2. Include comprehensive markdown documentation
3. Add error handling and logging
4. Update this README with new content
5. Test with multiple Azure OpenAI deployments
6. Document prerequisites and dependencies

---

## 📖 Additional Resources

### Azure Documentation
- [Azure AI Foundry](https://learn.microsoft.com/azure/ai-studio/)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [Azure AI Evaluation](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-sdk)
- [OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-enable)

---

## 📝 License

This project is for educational and demonstration purposes. Review Azure service terms and pricing before production use.

---

## ✉️ Support

For issues or questions:
1. Review notebook documentation and comments
2. Consult Azure documentation links above
3. Review error messages and Application Insights logs

---

**Last Updated**: November 13, 2025  
**Version**: 1.0.0
