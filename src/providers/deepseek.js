import OpenAI from 'openai';
import { BaseProvider } from './base-provider.js';

export class DeepSeekProvider extends BaseProvider {
    constructor(apiKey, modelName, systemMessage, allTools, logger, initialConversationHistory = []) {
        super(apiKey, modelName, systemMessage, allTools, logger);
        this.llmClient = new OpenAI({ 
            apiKey: this.apiKey, 
            baseURL: "https://api.deepseek.com/v1" // Standard DeepSeek API Base URL
        });
        this.initialize(initialConversationHistory);
    }

    async initialize(initialConversationHistory = []) {
        this.logger.info(`[DeepSeekProvider] Initialized for model: ${this.modelName}. Client configured.`);
        // No specific async initialization needed for DeepSeek client itself using OpenAI SDK.
    }

    async reconfigure(newSystemMessage, newAllTools) {
        await super.reconfigure(newSystemMessage, newAllTools);
        this.logger.info(`[DeepSeekProvider] Reconfigured. New system message and tools will be used on next API call.`);
    }

    getToolDeclarations() {
        this.logger.debug(`[DeepSeekProvider.getToolDeclarations]this.allTools count: ${this.allTools ? this.allTools.length : 'null or undefined'}`);
        if (!this.allTools || this.allTools.length === 0) {
            return undefined; // OpenAI/DeepSeek expects undefined if no tools
        }
        const declarations = this.allTools.map(tool => ({
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description || `Tool from ${tool.serverName}`,
                parameters: tool.inputSchema || { type: "object", properties: {} }
            }
        }));
        this.logger.debug('[DeepSeekProvider.getToolDeclarations] Generated declarations:', JSON.stringify(declarations, null, 2));
        return declarations;
    }
    
    _convertToolResultForDeepSeek(mcpToolResult, toolName) {
        // DeepSeek (like OpenAI) expects a stringified result for tool calls.
        this.logger.debug(`[DeepSeekProvider._convertToolResult] Raw MCP Result for ${toolName}:`, JSON.stringify(mcpToolResult, (key, value) => {
            if ((key === 'base64Data' || key === 'data') && typeof value === 'string' && value.length > 100) return `[base64_len:${value.length}]`;
            return value;
        }, 2));

        if (typeof mcpToolResult === 'string') return mcpToolResult;

        if (mcpToolResult && mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
            let firstTextPartContent = null;
            for (const part of mcpToolResult.content) {
                if (part.type === 'text' && typeof part.text === 'string') {
                    if (firstTextPartContent === null) firstTextPartContent = part.text;
                    try {
                        JSON.parse(part.text);
                        this.logger.debug(`[DeepSeekProvider._convertToolResult] Found parsable JSON in text part for ${toolName}.`);
                        return part.text; // Return the JSON string
                    } catch (e) { /* Is not JSON */ }
                }
            }
            const dataPart = mcpToolResult.content.find(p => p.type === 'data' && typeof p.data !== 'undefined');
            if (dataPart) {
                this.logger.debug(`[DeepSeekProvider._convertToolResult] Using data part for ${toolName}, stringifying.`);
                return JSON.stringify(dataPart.data);
            }
            if (firstTextPartContent !== null) {
                this.logger.debug(`[DeepSeekProvider._convertToolResult] Using first non-JSON text part as summary for ${toolName}.`);
                return firstTextPartContent;
            }
        }
        this.logger.debug(`[DeepSeekProvider._convertToolResult] Fallback: Stringifying entire toolResult for ${toolName}.`);
        return JSON.stringify(mcpToolResult);
    }

    formatConversation(conversationHistory) {
        const formattedMessages = [];
        if (this.systemMessage && (conversationHistory.length === 0 || conversationHistory[0].role !== 'system')) {
            formattedMessages.push({ role: 'system', content: this.systemMessage });
        }

        for (const message of conversationHistory) {
            if (message.role === 'system') {
                if (formattedMessages.length > 0 && formattedMessages[0].role === 'system' && formattedMessages[0].content === message.content) continue;
                continue;
            }
            if (message.role === 'user') {
                let userContent = message.content;
                // For DeepSeek (deepseek-chat), assuming no direct image input in messages for now.
                // If content is an array (e.g., from another provider's image message), convert to text summary.
                if (Array.isArray(message.content)) {
                    let textSummary = message.content
                        .filter(part => part.type === 'text' && typeof part.text === 'string')
                        .map(part => part.text)
                        .join('\n');
                    if (message.content.some(part => part.type === 'image_mcp' || part.type === 'image_url' || part.type === 'image')) {
                        textSummary += (textSummary ? '\n' : '') + "[Image data was present in history but not sent to DeepSeek provider]";
                    }
                    userContent = textSummary || "[Unsupported multi-part user message content]";
                }
                formattedMessages.push({ role: 'user', content: userContent });
            } else if (message.role === 'assistant') {
                const assistantMessagePayload = { role: 'assistant' };
                if (message.content && typeof message.content === 'string' && message.content.trim() !== '') {
                    assistantMessagePayload.content = message.content;
                } else {
                    assistantMessagePayload.content = null; 
                }

                // Filter out 'reasoning_content' if it ever appears from a DeepSeek model response in history
                // Though `deepseek-chat` might not produce it, this is good hygiene.
                // The DeepSeek docs state: "if the reasoning_content field is included in the sequence of input messages, the API will return a 400 error."
                // This check applies to the 'content' of an assistant message being *added* to history.
                // For *formatting* history, we need to ensure the `message` object itself doesn't have reasoning_content.
                // The example shows `messages.append(response.choices[0].message)` which could include it.
                // So, when formatting, if `message.reasoning_content` exists, we should not pass it.
                // The `OpenAI.Message` type doesn't have `reasoning_content`, so this would be a custom property.
                // Let's assume `MCPClient`'s conversation store for assistant messages only has `content` and `tool_calls`.

                if (message.tool_calls && message.tool_calls.length > 0) {
                    assistantMessagePayload.tool_calls = message.tool_calls.map(tc => ({
                        id: tc.id,
                        type: 'function',
                        function: {
                            name: tc.name,
                            arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args || {})
                        }
                    }));
                }
                if (assistantMessagePayload.content || assistantMessagePayload.tool_calls) {
                    formattedMessages.push(assistantMessagePayload);
                }
            } else if (message.role === 'tool') {
                formattedMessages.push({
                    role: 'tool',
                    tool_call_id: message.tool_call_id,
                    content: this._convertToolResultForDeepSeek(message.content, message.name) 
                });
            }
        }
        this.logger.debug('[DeepSeekProvider.formatConversation] Formatted messages:', JSON.stringify(formattedMessages, null, 2));
        return formattedMessages;
    }

    async sendMessage(formattedMessages) {
        if (!this.llmClient) {
            this.logger.error("[DeepSeekProvider.sendMessage] DeepSeek client not initialized!");
            throw new Error("DeepSeekProvider client not initialized.");
        }

        const toolDeclarations = this.getToolDeclarations();
        const requestToLLM = {
            model: this.modelName, // e.g., "deepseek-chat"
            messages: formattedMessages,
            temperature: 0.7, 
            max_tokens: 4096, 
        };

        if (toolDeclarations && toolDeclarations.length > 0) {
            requestToLLM.tools = toolDeclarations;
            requestToLLM.tool_choice = "auto";
        }
        
        this.logger.debug(`[DeepSeekProvider.sendMessage] Sending request to DeepSeek (model: ${this.modelName}):`, JSON.stringify(requestToLLM, null, 2));
        try {
            const deepSeekResponse = await this.llmClient.chat.completions.create(requestToLLM);
            this.logger.debug(`[DeepSeekProvider.sendMessage] Received raw response object from DeepSeek API call: ${JSON.stringify(deepSeekResponse)}`);
            
            if (!deepSeekResponse || !deepSeekResponse.choices || !Array.isArray(deepSeekResponse.choices) || deepSeekResponse.choices.length === 0 || !deepSeekResponse.choices[0].message) {
                this.logger.error("[DeepSeekProvider.sendMessage] DeepSeek API returned an unexpected, malformed, or empty response:", JSON.stringify(deepSeekResponse));
                let errorMessage = "DeepSeek API returned an invalid, empty, or malformed response.";
                // Attempt to find an error message within the DeepSeek response, if structured that way
                if (deepSeekResponse && typeof deepSeekResponse.error === 'object' && deepSeekResponse.error !== null && typeof deepSeekResponse.error.message === 'string') {
                    errorMessage += ` API Error: ${deepSeekResponse.error.message}`;
                } else if (deepSeekResponse && typeof deepSeekResponse.error === 'string') {
                    errorMessage += ` API Error: ${deepSeekResponse.error}`;
                }
                throw new Error(errorMessage);
            }
            return deepSeekResponse;
        } catch (error) {
            this.logger.error(`[DeepSeekProvider.sendMessage] Error during DeepSeek API call or response processing: ${error.message}`, error.stack);
            throw error; 
        }
    }

    extractTextContent(llmResponse) {
        // Standard OpenAI response structure for chat.completions
        // For deepseek-reasoner, the docs say response.choices[0].message.content is the final answer
        // and response.choices[0].message.reasoning_content is the CoT.
        // For deepseek-chat, it should just be response.choices[0].message.content.
        if (llmResponse.choices && llmResponse.choices[0] && llmResponse.choices[0].message) {
            const message = llmResponse.choices[0].message;
            let mainContent = message.content || '';
            
            // If this were deepseek-reasoner and we wanted to display CoT:
            // if (message.reasoning_content) {
            //    this.logger.info(`[DeepSeekProvider] Reasoning Content: ${message.reasoning_content}`);
            //    // Decide if/how to prepend or show this to user. For now, just log.
            // }
            return mainContent;
        }
        return '';
    }

    extractToolCalls(llmResponse) {
        // Standard OpenAI response structure for tool_calls
        if (llmResponse.choices && llmResponse.choices[0] && llmResponse.choices[0].message && llmResponse.choices[0].message.tool_calls) {
            const validToolCalls = [];
            for (const tc of llmResponse.choices[0].message.tool_calls) {
                if (tc.type === 'function') {
                    try {
                        const parsedArgs = JSON.parse(tc.function.arguments);
                        validToolCalls.push({
                            id: tc.id,
                            name: tc.function.name,
                            args: parsedArgs 
                        });
                    } catch (e) {
                        this.logger.error(`[DeepSeekProvider.extractToolCalls] Failed to parse arguments for tool call '${tc.function.name}' (id: ${tc.id}). Error: ${e.message}. Arguments string: "${tc.function.arguments}"`);
                    }
                }
            }
            return validToolCalls.length > 0 ? validToolCalls : null;
        }
        return null;
    }
    
    convertToolResult(mcpToolResult, toolName) {
        return this._convertToolResultForDeepSeek(mcpToolResult, toolName);
    }

    // For deepseek-chat, we assume no direct image input via messages for now.
    // So, these methods related to preparing image messages for sending are not strictly needed
    // unless deepseek-chat API evolves to support it like gpt-4o.
    extractImageFromToolResult(toolCallResult) {
        this.logger.warn("[DeepSeekProvider.extractImageFromToolResult] Image extraction from tool result called, but DeepSeek provider does not currently support sending images to the LLM.");
        // This method is on BaseProvider. If a tool *called by DeepSeek* returns an image,
        // MCPClient's _extractImageFromMcpToolResult will still work to get the image for history.
        // This provider method is for when the *provider itself* needs to process an image from a tool it called.
        return { base64ImageData: null, mimeType: null };
    }

    prepareImageMessageContent(base64ImageData, mimeType, textPrompt) {
        this.logger.warn("[DeepSeekProvider.prepareImageMessageContent] Image preparation called, but DeepSeek provider does not currently support sending images to the LLM.");
        // Returns a text-only representation or an error/warning.
        return [{ type: 'text', text: textPrompt + " [Image content was intended here but not sent to DeepSeek provider]" }];
    }
} 