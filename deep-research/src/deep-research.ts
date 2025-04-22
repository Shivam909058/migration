import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import { generateObject } from 'ai';

import { getModel, trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';
import { combinedSearch, extractInsightsFromResults } from './search';

function log(...args: any[]) {
  console.log(...args);
}

export type ResearchProgress = {
  currentDepth: number;
  totalDepth: number;
  currentBreadth: number;
  totalBreadth: number;
  currentQuery?: string;
  totalQueries: number;
  completedQueries: number;
};

type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

// increase this if you have higher API rate limits
const ConcurrencyLimit = Number(process.env.CONCURRENCY_LIMIT) || 2;

// take en user query, return a list of SERP queries
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
}: {
  query: string;
  numQueries?: number;

  // optional, if provided, the research will continue from the last learning
  learnings?: string[];
}) {
  const res = await generateObject({
    model: getModel(),
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, generate a list of search queries to research the topic. Return a maximum of ${numQueries} queries, but feel free to return less if the original prompt is clear. Make sure each query is unique and not similar to each other: <prompt>${query}</prompt>\n\n${
      learnings
        ? `Here are some learnings from previous research, use them to generate more specific queries: ${learnings.join(
            '\n',
          )}`
        : ''
    }`,
    schema: z.object({
      queries: z
        .array(
          z.object({
            query: z.string().describe('The search query'),
            researchGoal: z
              .string()
              .describe(
                'First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions.',
              ),
          }),
        )
        .describe(`List of search queries, max of ${numQueries}`),
    }),
  });
  log(`Created ${res.object.queries.length} queries`, res.object.queries);

  return res.object.queries.slice(0, numQueries);
}

export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}) {
  const learningsString = learnings
    .map(learning => `<learning>\n${learning}\n</learning>`)
    .join('\n');

  const res = await generateObject({
    model: getModel(),
    system: systemPrompt(),
    prompt: trimPrompt(
      `Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>`,
    ),
    schema: z.object({
      reportMarkdown: z.string().describe('Final report on the topic in Markdown'),
    }),
  });

  // Append the visited URLs section to the report
  const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
  return res.object.reportMarkdown + urlsSection;
}

export async function writeFinalAnswer({
  prompt,
  learnings,
}: {
  prompt: string;
  learnings: string[];
}) {
  const learningsString = learnings
    .map(learning => `<learning>\n${learning}\n</learning>`)
    .join('\n');

  const res = await generateObject({
    model: getModel(),
    system: systemPrompt(),
    prompt: trimPrompt(
      `Given the following prompt from the user, write a final answer on the topic using the learnings from research. Follow the format specified in the prompt. Do not yap or babble or include any other text than the answer besides the format specified in the prompt. Keep the answer as concise as possible - usually it should be just a few words or maximum a sentence. Try to follow the format specified in the prompt (for example, if the prompt is using Latex, the answer should be in Latex. If the prompt gives multiple answer choices, the answer should be one of the choices).\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from research on the topic that you can use to help answer the prompt:\n\n<learnings>\n${learningsString}\n</learnings>`,
    ),
    schema: z.object({
      exactAnswer: z
        .string()
        .describe('The final answer, make it short and concise, just the answer, no other text'),
    }),
  });

  return res.object.exactAnswer;
}

export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
  onProgress,
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
  onProgress?: (progress: ResearchProgress) => void;
}): Promise<ResearchResult> {
  const progress: ResearchProgress = {
    currentDepth: depth,
    totalDepth: depth,
    currentBreadth: breadth,
    totalBreadth: breadth,
    totalQueries: 0,
    completedQueries: 0,
  };

  const reportProgress = (update: Partial<ResearchProgress>) => {
    Object.assign(progress, update);
    onProgress?.(progress);
  };

  const serpQueries = await generateSerpQueries({
    query,
    learnings,
    numQueries: breadth,
  });

  reportProgress({
    totalQueries: serpQueries.length,
    currentQuery: serpQueries[0]?.query,
  });

  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          // Use our new combined search function
          const searchResults = await combinedSearch(serpQuery.query, 5);
          
          // Collect URLs from this search
          const newUrls = compact(searchResults.map(item => item.url));
          const newBreadth = Math.ceil(breadth / 2);
          const newDepth = depth - 1;

          // Extract insights from search results
          const newLearnings = await extractInsightsFromResults(
            serpQuery.query, 
            searchResults, 
            newBreadth
          );
          
          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          if (newDepth > 0) {
            log(`Researching deeper, breadth: ${newBreadth}, depth: ${newDepth}`);

            reportProgress({
              currentDepth: newDepth,
              currentBreadth: newBreadth,
              completedQueries: progress.completedQueries + 1,
              currentQuery: serpQuery.query,
            });

            // If we have follow-up questions, use them as queries for the next round
            const nextQueries = newLearnings.followUpQuestions?.length
              ? newLearnings.followUpQuestions.slice(0, newBreadth).map(question => ({
                  query: question,
                  researchGoal: `Further research based on initial findings about "${serpQuery.query}"`,
                }))
              : await generateSerpQueries({
                  query: serpQuery.query,
                  numQueries: newBreadth,
                  learnings: allLearnings,
                });

            // Recurse with derived queries
            const nextResults = await Promise.all(
              nextQueries.map(nextQuery =>
                deepResearch({
                  query: nextQuery.query,
                  breadth: newBreadth,
                  depth: newDepth,
                  learnings: allLearnings,
                  visitedUrls: allUrls,
                  onProgress,
                }),
              ),
            );

            // Combine results from all branches
            return nextResults.reduce(
              (acc, result) => ({
                learnings: [...acc.learnings, ...result.learnings],
                visitedUrls: [...acc.visitedUrls, ...result.visitedUrls],
              }),
              { learnings: allLearnings, visitedUrls: allUrls },
            );
          }

          return { learnings: allLearnings, visitedUrls: allUrls };
        } catch (error) {
          console.error(`Error researching query: ${serpQuery.query}`, error);
          return { learnings, visitedUrls };
        } finally {
          reportProgress({
            completedQueries: progress.completedQueries + 1,
          });
        }
      }),
    ),
  );

  // Combine results from all branches
  const combinedResults = results.reduce(
    (acc, result) => ({
      learnings: [...acc.learnings, ...result.learnings],
      visitedUrls: [...acc.visitedUrls, ...result.visitedUrls],
    }),
    { learnings: [], visitedUrls: [] },
  );

  // Deduplicate
  return {
    learnings: [...new Set(combinedResults.learnings)],
    visitedUrls: [...new Set(combinedResults.visitedUrls)],
  };
}
