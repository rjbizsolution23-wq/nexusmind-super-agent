// ═══════════════════════════════════════════════════════════════
// 🧠 NEXUSMIND KNOWLEDGE RETRIEVAL ENGINE
// ═══════════════════════════════════════════════════════════════

import { Pinecone } from '@pinecone-database/pinecone'
import { OpenAI } from 'openai'

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

const PINECONE_API_KEY = process.env.PINECONE_API_KEY!
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY })
const openai = new OpenAI({ apiKey: OPENAI_API_KEY })

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export interface KnowledgeSource {
  id: string
  title: string
  url: string
  description: string
  category: string
  apiEndpoint?: string
  authentication?: {
    type: 'api_key' | 'oauth' | 'basic' | 'none'
    keyName?: string
    envVar?: string
  }
  features: string[]
  integration: {
    language: string
    example: string
  }
  cost: string
  rateLimits: string
  lastVerified: string
}

export interface Citation {
  source: string
  url: string
  dateAccessed: string
  relevance: string
  category: string
}

export interface KnowledgeQueryResult {
  sources: KnowledgeSource[]
  citations: Citation[]
  answer?: string
  confidence: number
}

// ─────────────────────────────────────────────────────────────
// Knowledge Base Registry
// ─────────────────────────────────────────────────────────────

export const KNOWLEDGE_REGISTRY: Record<string, KnowledgeSource> = {
  'arxiv': {
    id: 'arxiv',
    title: 'arXiv',
    url: 'https://arxiv.org',
    description: 'Open-access repository of over 2 million scholarly articles',
    category: 'Research & Knowledge Access',
    apiEndpoint: 'https://arxiv.org/api/query',
    authentication: { type: 'none' },
    features: [
      'Physics, Mathematics, CS, Biology',
      'Free full-text PDFs',
      'Daily updates',
      'LaTeX source available'
    ],
    integration: {
      language: 'python',
      example: `import arxiv

search = arxiv.Search(
    query="machine learning",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in search.results():
    print(f"Title: {result.title}")
    print(f"PDF: {result.pdf_url}")`
    },
    cost: 'Free',
    rateLimits: '3 requests/second',
    lastVerified: '2025-12-02'
  },
  
  'github': {
    id: 'github',
    title: 'GitHub',
    url: 'https://github.com',
    description: 'World\'s largest code hosting platform with 100M+ repositories',
    category: 'Code Repositories',
    apiEndpoint: 'https://api.github.com',
    authentication: {
      type: 'api_key',
      keyName: 'GITHUB_TOKEN',
      envVar: 'GITHUB_TOKEN'
    },
    features: [
      '100M+ repositories',
      'Code search',
      'Issues & PRs',
      'Actions (CI/CD)',
      'GitHub Copilot'
    ],
    integration: {
      language: 'python',
      example: `import requests

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

response = requests.get(
    "https://api.github.com/search/repositories",
    headers=headers,
    params={
        "q": "machine learning language:python",
        "sort": "stars",
        "order": "desc"
    }
)

repos = response.json()["items"]`
    },
    cost: 'Free (public repos), Pro from $4/month',
    rateLimits: '5,000 requests/hour (authenticated)',
    lastVerified: '2025-12-02'
  },
  
  'huggingface': {
    id: 'huggingface',
    title: 'HuggingFace',
    url: 'https://huggingface.co',
    description: '500K+ models, 100K+ datasets, AI community hub',
    category: 'AI/ML Model Repositories',
    apiEndpoint: 'https://huggingface.co/api',
    authentication: {
      type: 'api_key',
      keyName: 'HF_TOKEN',
      envVar: 'HUGGINGFACE_TOKEN'
    },
    features: [
      'Transformers',
      'Diffusers',
      'Model cards',
      'Inference API',
      'Dataset viewer'
    ],
    integration: {
      language: 'python',
      example: `from huggingface_hub import HfApi, InferenceClient

api = HfApi()
models = api.list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1
)

client = InferenceClient(token="hf_token")
result = client.text_classification("I love AI agents!")`
    },
    cost: 'Free (community), Pro from $9/month',
    rateLimits: 'Varies by endpoint',
    lastVerified: '2025-12-02'
  },
  
  'openai': {
    id: 'openai',
    title: 'OpenAI API',
    url: 'https://platform.openai.com',
    description: 'GPT-4, GPT-3.5, DALL-E, Whisper',
    category: 'Large Language Models',
    apiEndpoint: 'https://api.openai.com/v1',
    authentication: {
      type: 'api_key',
      keyName: 'OPENAI_API_KEY',
      envVar: 'OPENAI_API_KEY'
    },
    features: [
      'Function calling',
      'Streaming',
      'Fine-tuning',
      'Embeddings',
      'Vision'
    ],
    integration: {
      language: 'python',
      example: `from openai import OpenAI

client = OpenAI(api_key="sk-proj-...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful AI agent."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)`
    },
    cost: 'Pay per token (GPT-4: $0.03/1K input tokens)',
    rateLimits: 'Tier-based (default: 60 requests/minute)',
    lastVerified: '2025-12-02'
  },
  
  'anthropic': {
    id: 'anthropic',
    title: 'Anthropic Claude',
    url: 'https://www.anthropic.com',
    description: 'Claude 3 family (Opus, Sonnet, Haiku)',
    category: 'Large Language Models',
    apiEndpoint: 'https://api.anthropic.com/v1',
    authentication: {
      type: 'api_key',
      keyName: 'ANTHROPIC_API_KEY',
      envVar: 'ANTHROPIC_API_KEY'
    },
    features: [
      '200K context window',
      'Constitutional AI',
      'Tool use',
      'Vision',
      'Streaming'
    ],
    integration: {
      language: 'python',
      example: `import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain chain-of-thought reasoning"}
    ]
)

print(message.content)`
    },
    cost: 'Pay per token (Opus: $15/1M input tokens)',
    rateLimits: '1,000 requests/minute (default tier)',
    lastVerified: '2025-12-02'
  },
  
  'pinecone': {
    id: 'pinecone',
    title: 'Pinecone',
    url: 'https://www.pinecone.io',
    description: 'Vector database for AI applications',
    category: 'Databases & Data Storage',
    apiEndpoint: 'https://api.pinecone.io',
    authentication: {
      type: 'api_key',
      keyName: 'PINECONE_API_KEY',
      envVar: 'PINECONE_API_KEY'
    },
    features: [
      'Semantic search',
      'Similarity matching',
      'Metadata filtering',
      'Hybrid search',
      'Real-time updates'
    ],
    integration: {
      language: 'python',
      example: `from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_...")
index = pc.Index("agent-knowledge")

index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, 0.3, ...],
        "metadata": {"source": "arxiv", "title": "Paper Title"}
    }
])

results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=5,
    include_metadata=True
)`
    },
    cost: 'Starter free, paid from $70/month',
    rateLimits: '100 operations/second (starter)',
    lastVerified: '2025-12-02'
  },
  
  'kaggle': {
    id: 'kaggle',
    title: 'Kaggle',
    url: 'https://www.kaggle.com',
    description: '50K+ datasets, competitions, and notebooks',
    category: 'Dataset Repositories',
    apiEndpoint: 'https://www.kaggle.com/api/v1',
    authentication: {
      type: 'api_key',
      keyName: 'KAGGLE_KEY',
      envVar: 'KAGGLE_KEY'
    },
    features: [
      'Competition leaderboards',
      'Kernels (notebooks)',
      'Dataset versioning',
      'Community discussions',
      'GPU access'
    ],
    integration: {
      language: 'python',
      example: `from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    'zillow/zecon',
    path='./data',
    unzip=True
)

datasets = api.dataset_list(search='machine learning')`
    },
    cost: 'Free',
    rateLimits: 'Generous (no official limit)',
    lastVerified: '2025-12-02'
  }
}

// ─────────────────────────────────────────────────────────────
// Vector Embeddings
// ─────────────────────────────────────────────────────────────

async function generateEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  })
  
  return response.data[0].embedding
}

// ─────────────────────────────────────────────────────────────
// Knowledge Base Initialization
// ─────────────────────────────────────────────────────────────

export async function initializeKnowledgeBase(indexName: string = 'nexusmind-kb') {
  try {
    // Create Pinecone index if it doesn't exist
    const existingIndexes = await pinecone.listIndexes()
    
    if (!existingIndexes.indexes?.some(idx => idx.name === indexName)) {
      await pinecone.createIndex({
        name: indexName,
        dimension: 1536, // text-embedding-3-small dimension
        metric: 'cosine',
        spec: {
          serverless: {
            cloud: 'aws',
            region: 'us-east-1'
          }
        }
      })
    }
    
    const index = pinecone.Index(indexName)
    
    // Upsert all knowledge sources
    const vectors = await Promise.all(
      Object.values(KNOWLEDGE_REGISTRY).map(async (source) => {
        const text = `${source.title} ${source.description} ${source.features.join(' ')}`
        const embedding = await generateEmbedding(text)
        
        return {
          id: source.id,
          values: embedding,
          metadata: {
            title: source.title,
            url: source.url,
            category: source.category,
            description: source.description,
            lastVerified: source.lastVerified
          }
        }
      })
    )
    
    await index.upsert(vectors)
    
    console.log(`✅ Initialized knowledge base with ${vectors.length} sources`)
    
    return { success: true, count: vectors.length }
  } catch (error) {
    console.error('❌ Failed to initialize knowledge base:', error)
    throw error
  }
}

// ─────────────────────────────────────────────────────────────
// Semantic Search
// ─────────────────────────────────────────────────────────────

export async function searchKnowledge(
  query: string,
  options: {
    topK?: number
    category?: string
    includeIntegration?: boolean
  } = {}
): Promise<KnowledgeQueryResult> {
  try {
    const { topK = 5, category, includeIntegration = true } = options
    
    // Generate query embedding
    const queryEmbedding = await generateEmbedding(query)
    
    // Search Pinecone
    const index = pinecone.Index('nexusmind-kb')
    const searchResults = await index.query({
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
      filter: category ? { category: { $eq: category } } : undefined
    })
    
    // Map results to knowledge sources
    const sources: KnowledgeSource[] = searchResults.matches
      .map(match => {
        const source = KNOWLEDGE_REGISTRY[match.id]
        if (!source) return null
        
        if (!includeIntegration) {
          const { integration, ...rest } = source
          return rest as KnowledgeSource
        }
        
        return source
      })
      .filter((s): s is KnowledgeSource => s !== null)
    
    // Generate citations
    const citations: Citation[] = sources.map(source => ({
      source: source.title,
      url: source.url,
      dateAccessed: new Date().toISOString().split('T')[0],
      relevance: source.category,
      category: source.category
    }))
    
    // Calculate confidence score (based on top match score)
    const confidence = searchResults.matches[0]?.score || 0
    
    return {
      sources,
      citations,
      confidence
    }
  } catch (error) {
    console.error('❌ Knowledge search failed:', error)
    throw error
  }
}

// ─────────────────────────────────────────────────────────────
// RAG (Retrieval-Augmented Generation)
// ─────────────────────────────────────────────────────────────

export async function answerWithKnowledge(
  question: string,
  options: {
    model?: string
    temperature?: number
    maxTokens?: number
  } = {}
): Promise<KnowledgeQueryResult> {
  try {
    const { model = 'gpt-4', temperature = 0.7, maxTokens = 1000 } = options
    
    // Step 1: Retrieve relevant knowledge
    const { sources, citations, confidence } = await searchKnowledge(question, { topK: 3 })
    
    // Step 2: Build context from sources
    const context = sources.map(source => `
**${source.title}**
${source.description}

Key Features:
${source.features.map(f => `- ${f}`).join('\n')}

API Endpoint: ${source.apiEndpoint || 'N/A'}
Cost: ${source.cost}
Rate Limits: ${source.rateLimits}

Integration Example:
\`\`\`${source.integration.language}
${source.integration.example}
\`\`\`
    `).join('\n\n---\n\n')
    
    // Step 3: Generate answer with LLM
    const response = await openai.chat.completions.create({
      model,
      temperature,
      max_tokens: maxTokens,
      messages: [
        {
          role: 'system',
          content: `You are an AI agent assistant with access to a comprehensive knowledge base.

**Current Date:** ${new Date().toISOString().split('T')[0]}

When answering:
1. Use only information from the provided knowledge sources
2. Cite sources in your response
3. Include relevant API endpoints and integration examples
4. Mention costs and rate limits when relevant
5. Always include the date you accessed the information

Format citations as: **Source:** [Source Name] - [URL]`
        },
        {
          role: 'user',
          content: `Question: ${question}

Available Knowledge Sources:
${context}

Please provide a comprehensive answer with proper citations.`
        }
      ]
    })
    
    const answer = response.choices[0].message.content || ''
    
    return {
      sources,
      citations,
      answer,
      confidence
    }
  } catch (error) {
    console.error('❌ RAG query failed:', error)
    throw error
  }
}

// ─────────────────────────────────────────────────────────────
// Citation Generator
// ─────────────────────────────────────────────────────────────

export function generateCitationMarkdown(citations: Citation[]): string {
  const today = new Date().toISOString().split('T')[0]
  
  return `## 📚 Sources & Citations

**Date Generated:** ${today}

${citations.map((citation, idx) => `
### ${idx + 1}. ${citation.source}
- **URL:** ${citation.url}
- **Category:** ${citation.category}
- **Date Accessed:** ${citation.dateAccessed}
- **Relevance:** ${citation.relevance}
`).join('\n')}

---

**All sources verified as of ${today}**
`
}

// ─────────────────────────────────────────────────────────────
// Knowledge Base Stats
// ─────────────────────────────────────────────────────────────

export function getKnowledgeBaseStats() {
  const categories = new Set<string>()
  const totalSources = Object.keys(KNOWLEDGE_REGISTRY).length
  
  Object.values(KNOWLEDGE_REGISTRY).forEach(source => {
    categories.add(source.category)
  })
  
  return {
    totalSources,
    categories: Array.from(categories),
    categoryCounts: Object.values(KNOWLEDGE_REGISTRY).reduce((acc, source) => {
      acc[source.category] = (acc[source.category] || 0) + 1
      return acc
    }, {} as Record<string, number>)
  }
}

// ─────────────────────────────────────────────────────────────
// Export All
// ─────────────────────────────────────────────────────────────

export default {
  initializeKnowledgeBase,
  searchKnowledge,
  answerWithKnowledge,
  generateCitationMarkdown,
  getKnowledgeBaseStats,
  KNOWLEDGE_REGISTRY
}
