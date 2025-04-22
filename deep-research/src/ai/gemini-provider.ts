// deep-research/src/ai/gemini-provider.ts

import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import { LanguageModelV1, wrapLanguageModel, extractReasoningMiddleware } from 'ai';
import { getEncoding } from 'js-tiktoken';

// Initialize the API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Gemini 2.0 Flash model initialization with sensible defaults
export function createGeminiModel(): LanguageModelV1 {
  const model = genAI.getGenerativeModel({
    model: 'gemini-2.0-flash',
    safetySettings: [
      {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
    ],
    generationConfig: {
      temperature: 0.2,
      topP: 0.8, 
      topK: 40,
      maxOutputTokens: 8192,
    },
  });

  // Create a wrapper for the Gemini model that adapts it to the ai SDK's LanguageModelV1 interface
  const geminiWrapper: LanguageModelV1 = {
    modelId: 'gemini-2.0-flash',
    
    async complete({ prompt, system, maxTokens, temperature, stopSequences }) {
      const geminiPrompt = system ? `${system}\n\n${prompt}` : prompt;
      
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: geminiPrompt }] }],
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature: temperature,
          stopSequences: stopSequences,
        },
      });
      
      const response = result.response;
      return { text: response.text() };
    },
    
    async completeStructured({ prompt, system, schema, maxTokens, temperature, stopSequences }) {
      // Add schema as part of the prompt for structured output
      const schemaPrompt = `${prompt}\n\nPlease provide your response in this JSON schema:\n${JSON.stringify(schema, null, 2)}`;
      
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: system ? `${system}\n\n${schemaPrompt}` : schemaPrompt }] }],
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature: temperature,
          stopSequences: stopSequences,
        },
      });
      
      const response = result.response;
      // Extract JSON from the response text
      const text = response.text();
      let jsonMatch = text.match(/```json\n([\s\S]*?)\n```/) || 
                      text.match(/```\n([\s\S]*?)\n```/) || 
                      text.match(/{[\s\S]*?}/);
      
      let jsonString = jsonMatch ? jsonMatch[0] : text;
      // If we matched a code block, extract just the JSON part
      if (jsonString.startsWith('```')) {
        jsonString = jsonMatch![1];
      }
      
      try {
        const data = JSON.parse(jsonString);
        return { 
          data,
          text: response.text()
        };
      } catch (e) {
        console.error('Failed to parse JSON from Gemini response:', e);
        throw new Error('Failed to parse structured output from Gemini');
      }
    }
  };

  return wrapLanguageModel({
    model: geminiWrapper,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
}

// Create a grounding-enabled model for web search
export function createGeminiGroundingModel(): LanguageModelV1 {
  const model = genAI.getGenerativeModel({
    model: 'gemini-2.0-flash',
    safetySettings: [
      {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
      {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      },
    ],
    generationConfig: {
      temperature: 0.2,
      topP: 0.8,
      topK: 40,
      maxOutputTokens: 8192,
    },
    tools: [{
      googleSearch: {}
    }]
  });

  // Create a wrapper for the Gemini Grounding model
  const geminiGroundingWrapper: LanguageModelV1 = {
    modelId: 'gemini-2.0-flash-grounding',
    
    async complete({ prompt, system, maxTokens, temperature, stopSequences }) {
      const geminiPrompt = system ? `${system}\n\n${prompt}` : prompt;
      
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: geminiPrompt }] }],
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature: temperature,
          stopSequences: stopSequences,
        },
      });
      
      const response = result.response;
      return { text: response.text() };
    },
    
    async completeStructured({ prompt, system, schema, maxTokens, temperature, stopSequences }) {
      // Add schema as part of the prompt for structured output
      const schemaPrompt = `${prompt}\n\nPlease provide your response in this JSON schema:\n${JSON.stringify(schema, null, 2)}`;
      
      const result = await model.generateContent({
        contents: [{ role: 'user', parts: [{ text: system ? `${system}\n\n${schemaPrompt}` : schemaPrompt }] }],
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature: temperature,
          stopSequences: stopSequences,
        },
      });
      
      const response = result.response;
      // Extract JSON from the response text
      const text = response.text();
      let jsonMatch = text.match(/```json\n([\s\S]*?)\n```/) || 
                      text.match(/```\n([\s\S]*?)\n```/) || 
                      text.match(/{[\s\S]*?}/);
      
      let jsonString = jsonMatch ? jsonMatch[0] : text;
      // If we matched a code block, extract just the JSON part
      if (jsonString.startsWith('```')) {
        jsonString = jsonMatch![1];
      }
      
      try {
        const data = JSON.parse(jsonString);
        return { 
          data,
          text: response.text()
        };
      } catch (e) {
        console.error('Failed to parse JSON from Gemini response:', e);
        throw new Error('Failed to parse structured output from Gemini');
      }
    }
  };

  return wrapLanguageModel({
    model: geminiGroundingWrapper,
    middleware: extractReasoningMiddleware({ tagName: 'think' }),
  });
}