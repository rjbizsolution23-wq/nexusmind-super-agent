// ═══════════════════════════════════════════════════════════════
// 🔍 KNOWLEDGE API ROUTES
// ═══════════════════════════════════════════════════════════════

import { Hono } from 'hono'
import { 
  searchKnowledge, 
  answerWithKnowledge,
  generateCitationMarkdown,
  getKnowledgeBaseStats,
  KNOWLEDGE_REGISTRY
} from '../../../src/knowledge-engine/knowledge-retrieval'

const app = new Hono()

// ─────────────────────────────────────────────────────────────
// GET /knowledge/search - Semantic search across knowledge base
// ─────────────────────────────────────────────────────────────

app.post('/search', async (c) => {
  try {
    const { query, topK, category, includeIntegration } = await c.req.json()
    
    if (!query) {
      return c.json({ error: 'Query is required' }, 400)
    }
    
    const result = await searchKnowledge(query, {
      topK: topK || 5,
      category,
      includeIntegration: includeIntegration !== false
    })
    
    return c.json({
      success: true,
      query,
      timestamp: new Date().toISOString(),
      ...result
    })
  } catch (error: any) {
    return c.json({
      success: false,
      error: error.message
    }, 500)
  }
})

// ─────────────────────────────────────────────────────────────
// POST /knowledge/ask - RAG-powered Q&A
// ─────────────────────────────────────────────────────────────

app.post('/ask', async (c) => {
  try {
    const { question, model, temperature, maxTokens } = await c.req.json()
    
    if (!question) {
      return c.json({ error: 'Question is required' }, 400)
    }
    
    const result = await answerWithKnowledge(question, {
      model: model || 'gpt-4',
      temperature: temperature || 0.7,
      maxTokens: maxTokens || 1000
    })
    
    // Generate citation markdown
    const citationMarkdown = generateCitationMarkdown(result.citations)
    
    return c.json({
      success: true,
      question,
      answer: result.answer,
      sources: result.sources,
      citations: result.citations,
      citationMarkdown,
      confidence: result.confidence,
      timestamp: new Date().toISOString(),
      model: model || 'gpt-4'
    })
  } catch (error: any) {
    return c.json({
      success: false,
      error: error.message
    }, 500)
  }
})

// ─────────────────────────────────────────────────────────────
// GET /knowledge/sources - List all available sources
// ─────────────────────────────────────────────────────────────

app.get('/sources', (c) => {
  const { category } = c.req.query()
  
  let sources = Object.values(KNOWLEDGE_REGISTRY)
  
  if (category) {
    sources = sources.filter(s => s.category === category)
  }
  
  return c.json({
    success: true,
    count: sources.length,
    sources,
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// GET /knowledge/source/:id - Get specific source details
// ─────────────────────────────────────────────────────────────

app.get('/source/:id', (c) => {
  const { id } = c.req.param()
  
  const source = KNOWLEDGE_REGISTRY[id]
  
  if (!source) {
    return c.json({
      success: false,
      error: `Source '${id}' not found`
    }, 404)
  }
  
  return c.json({
    success: true,
    source,
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// GET /knowledge/categories - List all categories
// ─────────────────────────────────────────────────────────────

app.get('/categories', (c) => {
  const stats = getKnowledgeBaseStats()
  
  return c.json({
    success: true,
    ...stats,
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// POST /knowledge/integration - Get integration code for source
// ─────────────────────────────────────────────────────────────

app.post('/integration', async (c) => {
  const { sourceId, language } = await c.req.json()
  
  if (!sourceId) {
    return c.json({ error: 'sourceId is required' }, 400)
  }
  
  const source = KNOWLEDGE_REGISTRY[sourceId]
  
  if (!source) {
    return c.json({
      success: false,
      error: `Source '${sourceId}' not found`
    }, 404)
  }
  
  // Return integration example
  return c.json({
    success: true,
    source: source.title,
    language: source.integration.language,
    code: source.integration.example,
    authentication: source.authentication,
    apiEndpoint: source.apiEndpoint,
    rateLimits: source.rateLimits,
    cost: source.cost,
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// POST /knowledge/cite - Generate citations for used sources
// ─────────────────────────────────────────────────────────────

app.post('/cite', async (c) => {
  const { sourceIds } = await c.req.json()
  
  if (!sourceIds || !Array.isArray(sourceIds)) {
    return c.json({ error: 'sourceIds array is required' }, 400)
  }
  
  const citations = sourceIds
    .map(id => KNOWLEDGE_REGISTRY[id])
    .filter(Boolean)
    .map(source => ({
      source: source.title,
      url: source.url,
      dateAccessed: new Date().toISOString().split('T')[0],
      relevance: source.category,
      category: source.category
    }))
  
  const citationMarkdown = generateCitationMarkdown(citations)
  
  return c.json({
    success: true,
    citations,
    markdown: citationMarkdown,
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// GET /knowledge/stats - Knowledge base statistics
// ─────────────────────────────────────────────────────────────

app.get('/stats', (c) => {
  const stats = getKnowledgeBaseStats()
  
  return c.json({
    success: true,
    ...stats,
    lastUpdated: '2025-12-02',
    timestamp: new Date().toISOString()
  })
})

// ─────────────────────────────────────────────────────────────
// POST /knowledge/recommend - Recommend sources for a task
// ─────────────────────────────────────────────────────────────

app.post('/recommend', async (c) => {
  try {
    const { task, count } = await c.req.json()
    
    if (!task) {
      return c.json({ error: 'task description is required' }, 400)
    }
    
    // Use semantic search to find relevant sources
    const result = await searchKnowledge(task, {
      topK: count || 5,
      includeIntegration: true
    })
    
    return c.json({
      success: true,
      task,
      recommendations: result.sources.map((source, idx) => ({
        rank: idx + 1,
        source: source.title,
        url: source.url,
        reason: source.description,
        integration: source.integration,
        cost: source.cost,
        rateLimits: source.rateLimits
      })),
      timestamp: new Date().toISOString()
    })
  } catch (error: any) {
    return c.json({
      success: false,
      error: error.message
    }, 500)
  }
})

export default app
