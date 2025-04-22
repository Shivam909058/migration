// deep-research/src/search/index.ts

import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import { SerpAPI } from 'langchain/tools';
import { compact, uniqBy } from 'lodash-es';
import { generateObject } from 'ai';
import { z } from 'zod';

import { getGroundingModel, getModel, trimPrompt } from '../ai/providers';
import { systemPrompt } from '../prompt';

// Initialize search tools
export const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

export const serpapi = process.env.SERPAPI_KEY 
  ? new SerpAPI(process.env.SERPAPI_KEY, {
      location: 'United States',
      hl: 'en',
      gl: 'us',
    })
  : null;

// Gemini grounding search function
export async function geminiGroundingSearch(query: string, maxResults: number = 10): Promise<any[]> {
  try {
    // Use the grounding model to perform a web search
    const res = await generateObject({
      model: getGroundingModel(),
      system: `You are a web search assistant. Search the web for information on the given query and return detailed search results.`,
      prompt: `Search the web for information about: ${query}
      
      Present your findings as a JSON list of the ${maxResults} most relevant articles or resources, with each item containing:
      1. The title of the page
      2. A comprehensive summary of the content (at least 200 words)
      3. The complete URL
      
      Return only the JSON data with no additional explanations.`,
      schema: z.array(z.object({
        title: z.string().describe('The title of the resource'),
        content: z.string().describe('A comprehensive summary of the content (at least 200 words)'),
        url: z.string().describe('The complete URL of the resource'),
      })),
    });
    
    return res.object.slice(0, maxResults);
  } catch (error) {
    console.error('Error in Gemini grounding search:', error);
    return [];
  }
}

// Combined search using all available tools
export async function combinedSearch(query: string, maxResults: number = 10): Promise<any[]> {
  const results: any[] = [];
  
  // Try Gemini grounding search first
  try {
    const geminiResults = await geminiGroundingSearch(query, maxResults);
    results.push(...geminiResults);
  } catch (e) {
    console.error('Gemini grounding search failed:', e);
  }
  
  // If we don't have enough results, try FireCrawl
  if (results.length < maxResults) {
    try {
      const firecrawlResults = await firecrawl.search(query, {
        timeout: 15000,
        limit: maxResults - results.length,
        scrapeOptions: { formats: ['markdown'] },
      });
      
      const formatted = compact(firecrawlResults.data.map(item => ({
        title: item.title || 'Unknown Title',
        content: item.markdown || 'No content available',
        url: item.url || '',
      })));
      
      results.push(...formatted);
    } catch (e) {
      console.error('FireCrawl search failed:', e);
    }
  }
  
  // If we still don't have enough results and SERPAPI is available, try it
  if (results.length < maxResults && serpapi) {
    try {
      const serpResult = await serpapi.call(query);
      const serpData = JSON.parse(serpResult);
      
      // Format SERPAPI results
      if (serpData.organic_results) {
        const serpFormatted = serpData.organic_results.slice(0, maxResults - results.length).map((item: any) => ({
          title: item.title || 'Unknown Title',
          content: item.snippet || 'No content available',
          url: item.link || '',
        }));
        
        results.push(...serpFormatted);
      }
    } catch (e) {
      console.error('SERPAPI search failed:', e);
    }
  }
  
  // Remove duplicates based on URL
  return uniqBy(results, 'url').slice(0, maxResults);
}

// Extract insights from search results
export async function extractInsightsFromResults(query: string, results: any[], maxInsights: number = 5): Promise<any> {
  const contents = compact(results.map(item => item.content)).map(content =>
    trimPrompt(content, 25_000),
  );
  
  console.log(`Ran ${query}, found ${contents.length} contents`);

  const res = await generateObject({
    model: getModel(),
    system: systemPrompt(),
    prompt: trimPrompt(
      `Given the following contents from a search for the query <query>${query}</query>, generate a list of learnings from the contents. Return a maximum of ${maxInsights} learnings, but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.\n\n<contents>${contents
        .map(content => `<content>\n${content}\n</content>`)
        .join('\n')}</contents>`,
    ),
    schema: z.object({
      learnings: z.array(z.string()).describe(`List of learnings, max of ${maxInsights}`),
      followUpQuestions: z
        .array(z.string())
        .describe(
          `List of follow-up questions to research the topic further, max of ${maxInsights}`,
        ),
    }),
  });

  return res.object;
}