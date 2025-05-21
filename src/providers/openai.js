import OpenAI from 'openai';
import { BaseProvider } from './base-provider.js';
// import { logger as defaultLogger } from '../utils/logging.js'; // If needed directly

export class OpenAIProvider extends BaseProvider {
    constructor(apiKey, modelName, systemMessage, allTools, logger, initialConversationHistory = []) {
        super(apiKey, modelName, systemMessage, allTools, logger);
        this.llmClient = new OpenAI({ apiKey: this.apiKey });
        // OpenAI is mostly stateless for chat history in the same way Google's ChatSession is stateful.
        // History is passed with each request.
        this.initialize(initialConversationHistory);
    }

    async initialize(initialConversationHistory = []) {
        this.logger.info(`[OpenAIProvider] Initialized for model: ${this.modelName}. Client configured.`);
        // No specific async initialization needed for OpenAI client itself usually.
    }

    async reconfigure(newSystemMessage, newAllTools) {
        await super.reconfigure(newSystemMessage, newAllTools);
        this.logger.info(`[OpenAIProvider] Reconfigured. New system message and tools will be used on next API call.`);
    }

    getToolDeclarations() {
        this.logger.debug(`[OpenAIProvider.getToolDeclarations]this.allTools count: ${this.allTools ? this.allTools.length : 'null or undefined'}`);
        if (!this.allTools || this.allTools.length === 0) {
            return undefined; // OpenAI expects undefined if no tools, not an empty array
        }
        const declarations = this.allTools.map(tool => ({
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description || `Tool from ${tool.serverName}`,
                parameters: tool.inputSchema || { type: "object", properties: {} } // Ensure parameters is an object schema
            }
        }));
        this.logger.debug('[OpenAIProvider.getToolDeclarations] Generated declarations:', JSON.stringify(declarations, null, 2));
        return declarations;
    }
    
    // Internal helper for converting MCP tool result to OpenAI's expected string format
    _convertToolResultForOpenAI(mcpToolResult, toolName) {
        this.logger.debug(`[OpenAIProvider._convertToolResult] Raw MCP Result for ${toolName}:`, JSON.stringify(mcpToolResult, (key, value) => {
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
                        this.logger.debug(`[OpenAIProvider._convertToolResult] Found parsable JSON in text part for ${toolName}.`);
                        return part.text;
                    } catch (e) { /* Is not JSON */ }
                }
            }
            const dataPart = mcpToolResult.content.find(p => p.type === 'data' && typeof p.data !== 'undefined');
            if (dataPart) {
                this.logger.debug(`[OpenAIProvider._convertToolResult] Using data part for ${toolName}, stringifying.`);
                return JSON.stringify(dataPart.data);
            }
            if (firstTextPartContent !== null) {
                this.logger.debug(`[OpenAIProvider._convertToolResult] Using first non-JSON text part as summary for ${toolName}.`);
                return firstTextPartContent;
            }
        }
        this.logger.debug(`[OpenAIProvider._convertToolResult] Fallback: Stringifying entire toolResult for ${toolName}.`);
        return JSON.stringify(mcpToolResult);
    }

    /**
     * Extracts image data from a tool call result if present.
     * @param {Object} toolCallResult The raw result from `client.callTool()`.
     * @returns {{base64ImageData: string|null, mimeType: string|null}}
     */
    extractImageFromToolResult(toolCallResult) {
        let base64ImageData = null;
        let mimeType = null;

        if (toolCallResult && typeof toolCallResult === 'object' && toolCallResult !== null) {
            // Check for direct image property (common in MCP screenshot tool)
            if (toolCallResult.image && typeof toolCallResult.image.base64Data === 'string' && toolCallResult.image.mimeType) {
                base64ImageData = toolCallResult.image.base64Data;
                mimeType = toolCallResult.image.mimeType;
            }
            // Check for image in a content array (common in MCP SDK ToolCallResponse)
            else if (toolCallResult.content && Array.isArray(toolCallResult.content)) {
                const imageContentPart = toolCallResult.content.find(
                    p => p.type === 'image' && p.data && typeof p.data === 'string' && p.mimeType
                );
                if (imageContentPart) {
                    base64ImageData = imageContentPart.data;
                    mimeType = imageContentPart.mimeType;
                }
            }
        }

        if (base64ImageData && mimeType) {
            // Strip data URI prefix if present, OpenAI expects raw base64
            const base64PrefixMatch = /^data:[a-zA-Z0-9\\/\\+]+;base64,/;
            if (base64PrefixMatch.test(base64ImageData)) {
                base64ImageData = base64ImageData.replace(base64PrefixMatch, '');
            }
            this.logger.debug(`[OpenAIProvider.extractImageFromToolResult] Extracted image: ${mimeType}, data length: ${base64ImageData.length}`);
        } else {
            this.logger.debug('[OpenAIProvider.extractImageFromToolResult] No image found in tool result.');
        }
        return { base64ImageData, mimeType };
    }

    /**
     * Prepares the content array for an OpenAI message that includes an image.
     * @param {string} base64ImageData The base64 encoded image data (without data URI prefix).
     * @param {string} mimeType The mime type of the image.
     * @param {string} textPrompt The text prompt to accompany the image.
     * @returns {Array} The content array for the OpenAI message.
     */
    prepareImageMessageContent(base64ImageData, mimeType, textPrompt) {
        return [
            { type: 'text', text: textPrompt },
            {
                type: 'image_url',
                image_url: {
                    url: `data:${mimeType};base64,${base64ImageData}`,
                },
            },
        ];
    }

    formatConversation(conversationHistory) {
        // Provider should not trim global history; MCPClient handles that.
        // This local trim is if the provider itself imposes a token limit different from MCPClient's message count.
        // For now, assume MCPClient's trimming is sufficient.
        // this._trimConversationHistory(conversationHistory); 

        const formattedMessages = [];
        // Add system message if not already at the start and if defined
        if (this.systemMessage && (conversationHistory.length === 0 || conversationHistory[0].role !== 'system')) {
            formattedMessages.push({ role: 'system', content: this.systemMessage });
        }

        for (const message of conversationHistory) {
            if (message.role === 'system') {
                // If system message is already handled or should be unique at the start, skip others.
                // Assuming the first one added (if any) is authoritative.
                if (formattedMessages.length > 0 && formattedMessages[0].role === 'system' && formattedMessages[0].content === message.content) continue;
                // Or, if allowing multiple system messages (uncommon for OpenAI), ensure it's only role and content.
                // formattedMessages.push({ role: 'system', content: message.content });
                continue; // Generally, only one system message at the beginning is standard.
            }
            if (message.role === 'user') {
                let processedUserContent = message.content;
                if (Array.isArray(message.content)) {
                    const newContentArray = [];
                    for (const part of message.content) {
                        if (part.type === 'text' && typeof part.text === 'string') {
                            newContentArray.push(part);
                        } else if (part.type === 'image_url' && part.image_url && typeof part.image_url.url === 'string') {
                            newContentArray.push(part);
                        } else if (part.type === 'image' && part.source && part.source.type === 'base64' && part.source.media_type && part.source.data) {
                            newContentArray.push({
                                type: 'image_url',
                                image_url: {
                                    url: `data:${part.source.media_type};base64,${part.source.data}`,
                                },
                            });
                            this.logger.debug(`[OpenAIProvider.formatConversation] Converted Anthropic image (type: ${part.source.media_type}) to OpenAI image_url.`);
                        } else if (part.type === 'image_mcp' && part.source && part.source.media_type && part.source.data) {
                            // OpenAIProvider should IGNORE image_mcp, as its images are primarily handled by 
                            // MCPClient.continueConversation's specific OpenAI image path which creates an image_url directly.
                            // This image_mcp in history is for other providers if a switch occurs.
                            this.logger.debug(`[OpenAIProvider.formatConversation] Ignoring image_mcp part; OpenAI handles its tool images via a dedicated flow.`);
                            // No push to newContentArray for image_mcp
                        } else {
                            this.logger.warn(`[OpenAIProvider.formatConversation] Skipping unknown/incomplete part in user message content array: ${JSON.stringify(part).substring(0,100)}`);
                        }
                    }
                    processedUserContent = newContentArray.length > 0 ? newContentArray : '';
                }

                if (typeof processedUserContent === 'string' || (Array.isArray(processedUserContent) && processedUserContent.length > 0)) {
                    // Ensure only role and content are passed for user messages
                    formattedMessages.push({ role: 'user', content: processedUserContent });
                } else if (Array.isArray(processedUserContent) && processedUserContent.length === 0 && Array.isArray(message.content) && message.content.length > 0) {
                    this.logger.warn(`[OpenAIProvider.formatConversation] User message content array resulted in empty processed content. Original had ${message.content.length} parts.`);
                } else if (typeof processedUserContent !== 'string' && !Array.isArray(processedUserContent)){
                     this.logger.warn(`[OpenAIProvider.formatConversation] User message content is neither string nor array after processing, skipping: ${JSON.stringify(processedUserContent).substring(0,100)}`);
                }
            } else if (message.role === 'assistant') {
                // Ensure only role, content, and tool_calls are passed for assistant messages
                const assistantMessagePayload = { role: 'assistant' };
                if (message.content && typeof message.content === 'string' && message.content.trim() !== '') {
                    assistantMessagePayload.content = message.content;
                } else {
                    assistantMessagePayload.content = null; // Explicitly null if no text content, as per OpenAI spec
                }

                if (message.tool_calls && message.tool_calls.length > 0) {
                    assistantMessagePayload.tool_calls = message.tool_calls.map(tc => ({
                        id: tc.id,
                        type: 'function',
                        function: {
                            name: tc.name,
                            arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args || {}) // Arguments must be a string
                        }
                    }));
                }
                // Only push if there's content or tool_calls, to avoid empty assistant messages unless intended
                if (assistantMessagePayload.content || assistantMessagePayload.tool_calls) {
                    formattedMessages.push(assistantMessagePayload);
                }
            } else if (message.role === 'tool') {
                // Ensure only role, tool_call_id, name, and content are passed for tool messages
                formattedMessages.push({
                    role: 'tool',
                    tool_call_id: message.tool_call_id,
                    // name: message.name, // OpenAI's 'tool' role does not use 'name' at the top level here; it's part of the preceding assistant tool_call
                    content: this._convertToolResultForOpenAI(message.content, message.name) 
                });
            }
        }
        this.logger.debug('[OpenAIProvider.formatConversation] Formatted messages:', JSON.stringify(formattedMessages, null, 2));
        return formattedMessages;
    }

    async sendMessage(formattedMessages) {
        if (!this.llmClient) {
            this.logger.error("[OpenAIProvider.sendMessage] OpenAI client not initialized!");
            throw new Error("OpenAIProvider client not initialized.");
        }

        const toolDeclarations = this.getToolDeclarations(); // Gets tools in OpenAI format
        const requestToLLM = {
            model: this.modelName,
            max_tokens: 4096, // Consider making configurable
            temperature: 0.7, // Consider making configurable
            messages: formattedMessages,
        };
        if (toolDeclarations) { // Only add if tools are defined
            requestToLLM.tools = toolDeclarations;
            requestToLLM.tool_choice = "auto"; // or specific tool choice logic if needed
        }

        // this.logger.debug('[OpenAIProvider.sendMessage] Tool declarations prepared:', JSON.stringify(toolDeclarations, null, 2)); // Covered by getToolDeclarations log
        this.logger.debug('[OpenAIProvider.sendMessage] Sending request to OpenAI (messages count: ${requestToLLM.messages.length}, tools count: ${requestToLLM.tools ? requestToLLM.tools.length : 0}):', JSON.stringify(requestToLLM, null, 2));
        return this.llmClient.chat.completions.create(requestToLLM);
    }

    extractTextContent(llmResponse) {
        // llmResponse is the raw response from openai.chat.completions.create()
        if (llmResponse.choices && llmResponse.choices.length > 0 && llmResponse.choices[0].message) {
            return llmResponse.choices[0].message.content || '';
        }
        return '';
    }

    extractToolCalls(llmResponse) {
        // llmResponse is the raw response from openai.chat.completions.create()
        if (llmResponse.choices && llmResponse.choices.length > 0 && llmResponse.choices[0].message && llmResponse.choices[0].message.tool_calls) {
            const validToolCalls = [];
            for (const tc of llmResponse.choices[0].message.tool_calls) {
                try {
                    const parsedArgs = JSON.parse(tc.function.arguments);
                    validToolCalls.push({
                        id: tc.id,
                        name: tc.function.name,
                        args: parsedArgs 
                    });
                } catch (e) {
                    this.logger.error(`[OpenAIProvider.extractToolCalls] Failed to parse arguments for tool call '${tc.function.name}' (id: ${tc.id}). Error: ${e.message}. Arguments string: "${tc.function.arguments}"`);
                    // Optionally, push a tool call with an error or skip it.
                    // Skipping for now to prevent further issues downstream with malformed args.
                }
            }
            return validToolCalls.length > 0 ? validToolCalls : null;
        }
        return null;
    }
    
    // convertToolResult is effectively _convertToolResultForOpenAI and is used internally by formatConversation.
    // If it needs to be exposed on the interface, it can be, but typically not called directly by MCPClient for OpenAI.
    // For the BaseProvider interface, if we want a public `convertToolResult`:
    convertToolResult(mcpToolResult, toolName) {
        return this._convertToolResultForOpenAI(mcpToolResult, toolName);
    }
} 