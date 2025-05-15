import Anthropic from '@anthropic-ai/sdk';
import { BaseProvider } from './base-provider.js';
// Assuming logger might be needed, or passed via constructor as this.logger
// import { logger } from '../utils/logging.js'; 

export class AnthropicProvider extends BaseProvider {
    constructor(apiKey, modelName, systemMessage, allTools, logger, initialConversationHistory = []) {
        super(apiKey, modelName, systemMessage, allTools, logger);
        this.llmClient = new Anthropic({ apiKey: this.apiKey, defaultHeaders: { 'anthropic-version': '2023-06-01' } });
        // Anthropic is stateless for chat history in the same way Google's ChatSession is stateful.
        // History is passed with each request.
        // We can still use this.initialize for any setup if needed, or to validate.
        this.initialize(initialConversationHistory);
    }

    async initialize(initialConversationHistory = []) {
        // For Anthropic, initialization might just be validating the client or model.
        // It doesn't have a stateful chat session to initialize like Google.
        // We store the initial history if needed for the first call, but it's mainly managed by MCPClient's conversation.
        this.logger.info(`[AnthropicProvider] Initialized for model: ${this.modelName}. Client configured.`);
        // If we needed to make an initial API call to verify, this would be the place.
        // For now, constructor handles SDK client setup.
    }

    async reconfigure(newSystemMessage, newAllTools) {
        await super.reconfigure(newSystemMessage, newAllTools);
        // Anthropic doesn't need to re-initialize a chat session for system message changes.
        // The new system message and tools will be used in subsequent sendMessage calls.
        this.logger.info(`[AnthropicProvider] Reconfigured. New system message and tools will be used on next API call.`);
    }

    getToolDeclarations() {
        // Anthropic's tool format is slightly different from Google's FunctionDeclaration.
        // It expects an array of tool specifications directly.
        if (!this.allTools || this.allTools.length === 0) {
            return [];
        }
        return this.allTools.map(tool => ({
            name: tool.name,
            description: tool.description || `Tool from ${tool.serverName}`, // Anthropic uses 'description'
            input_schema: tool.inputSchema || { type: "object", properties: {} } // Anthropic uses 'input_schema'
        }));
    }

    formatConversation(conversationHistory, forTrimmingOnly = false) {
        // This method adapts the generic conversation history from MCPClient
        // to the format Anthropic's API expects.
        // The `_trimConversationHistory` call is expected to be done by MCPClient before calling this.
        // However, if this provider needs its own trimming logic, it could be added.
        // For now, this provider assumes MCPClient handles trimming.
        
        this.logger.debug('[AnthropicProvider.formatConversation] Formatting history:', JSON.stringify(conversationHistory, null, 2));
        
        const formattedMessages = [];
        const toolResultsMap = {};

        // First pass: collect all tool results and map them by their call ID.
        // The `convertToolResult` method here will ensure they are in Anthropic's desired format.
        for (const message of conversationHistory) {
            if (message.role === 'tool' && message.tool_call_id) {
                toolResultsMap[message.tool_call_id] = this.convertToolResult(message.content, message.name);
            }
        }

        let pendingToolResults = []; // Stores tool_result blocks to be attached to the next user message
                                   // or sent as a standalone user message if they are last.

        for (let i = 0; i < conversationHistory.length; i++) {
            const message = conversationHistory[i];

            if (message.role === 'user') {
                if (pendingToolResults.length > 0) {
                    formattedMessages.push({ role: 'user', content: [...pendingToolResults] });
                    pendingToolResults = [];
                }
                // Pass user message content through our conversion helper
                const processedUserContent = this._convertUserContentForAnthropic(message.content);
                if (processedUserContent && processedUserContent.length > 0) {
                    formattedMessages.push({ role: 'user', content: processedUserContent });
                } else if (typeof message.content === 'string' && (!processedUserContent || processedUserContent.length === 0)){
                    // If original was a string and conversion yielded nothing, push the original string.
                    formattedMessages.push({ role: 'user', content: message.content });
                } else if (processedUserContent && processedUserContent.length === 0 && typeof message.content !== 'string'){
                    this.logger.warn(`[AnthropicProvider.formatConversation] User message (originally not a simple string) resulted in empty processed content. Original content preview: ${JSON.stringify(message.content).substring(0, 100)}...`);
                    // Optionally, push an empty text block or skip if appropriate for Anthropic when content becomes empty
                    // For now, let's not push anything if it was complex and became empty, to avoid sending empty user turns.
                }
            } else if (message.role === 'assistant') {
                const assistantContent = [];
                if (message.content && typeof message.content === 'string' && message.content.trim() !== '') {
                    assistantContent.push({ type: 'text', text: message.content });
                }

                if (message.tool_calls && message.tool_calls.length > 0) {
                    for (const toolCall of message.tool_calls) {
                        assistantContent.push({
                            type: 'tool_use',
                            id: toolCall.id,
                            name: toolCall.name,
                            input: toolCall.args || {} // Ensure args is an object
                        });
                        // After an assistant makes a tool_use call, the corresponding tool_result
                        // should appear in the subsequent 'user' message.
                        const toolResultForThisCall = toolResultsMap[toolCall.id];
                        if (toolResultForThisCall) { // toolResultForThisCall is already formatted
                            pendingToolResults.push({
                                type: 'tool_result',
                                tool_use_id: toolCall.id,
                                content: toolResultForThisCall, 
                                // Anthropic expects `content` to be an array of blocks or a string for tool_result
                                // `this.convertToolResult` should ensure this.
                            });
                        } else {
                            // This case (tool_call made but no result found in history yet) is normal if history is mid-sequence.
                            // If the tool call is the *last* thing in the history, MCPClient will call the tool and then continueConversation.
                            // If history is being formatted *after* a tool was called and result added, toolResultForThisCall should be found.
                            this.logger.warn(`[AnthropicProvider.formatConversation] No tool result found in history for tool_call_id: ${toolCall.id} (name: ${toolCall.name}). This might be okay if the call is very recent and result is pending.`);
                        }
                    }
                }
                
                if (assistantContent.length > 0) {
                    formattedMessages.push({ role: 'assistant', content: assistantContent });
                }

                // If this assistant message had tool_calls and thus generated pendingToolResults,
                // and it's either the last message in history or the next message isn't a user message
                // (which would naturally cause a flush), then we need to flush the tool results now.
                if (pendingToolResults.length > 0 && 
                    (i === conversationHistory.length - 1 || conversationHistory[i+1]?.role !== 'user')) {
                    this.logger.debug('[AnthropicProvider.formatConversation] Flushing pending tool results immediately after assistant turn.');
                    formattedMessages.push({ role: 'user', content: [...pendingToolResults] });
                    pendingToolResults = [];
                }
            }
            // 'tool' role messages from MCPClient's history are processed in the first pass to populate toolResultsMap.
            // They don't directly map to a top-level message in Anthropic's format, but their content is used.
        }

        // If there are any remaining pending tool results (e.g., if the conversation ends with assistant tool_calls)
        // they must be sent as a final 'user' message.
        if (pendingToolResults.length > 0) {
            formattedMessages.push({ role: 'user', content: [...pendingToolResults] });
        }
        
        this.logger.debug('[AnthropicProvider.formatConversation] Formatted messages:', JSON.stringify(formattedMessages, null, 2));
        return formattedMessages;
    }

    async sendMessage(formattedMessages) {
        if (!this.llmClient) {
            this.logger.error("[AnthropicProvider.sendMessage] Anthropic client not initialized!");
            throw new Error("AnthropicProvider client not initialized.");
        }

        const toolDeclarations = this.getToolDeclarations();
        const requestToLLM = {
            model: this.modelName,
            max_tokens: 4096, // Consider making this configurable
            temperature: 0.7, // Consider making this configurable
            system: this.systemMessage,
            messages: formattedMessages,
            tools: toolDeclarations.length > 0 ? toolDeclarations : undefined,
        };

        this.logger.debug('[AnthropicProvider.sendMessage] Sending request to Anthropic:', JSON.stringify(requestToLLM, null, 2));
        return this.llmClient.messages.create(requestToLLM);
    }

    extractTextContent(llmResponse) {
        if (!llmResponse || !llmResponse.content) {
            return '';
        }
        return llmResponse.content
            .filter(contentBlock => contentBlock.type === 'text')
            .map(textBlock => textBlock.text)
            .join('\n');
    }

    extractToolCalls(llmResponse) {
        if (!llmResponse || !llmResponse.content) {
            return null;
        }
        const toolCalls = llmResponse.content
            .filter(contentBlock => contentBlock.type === 'tool_use')
            .map(toolUseBlock => ({
                id: toolUseBlock.id,
                name: toolUseBlock.name,
                args: toolUseBlock.input || {} // Ensure args is an object
            }));
        return toolCalls.length > 0 ? toolCalls : null;
    }

    convertToolResult(mcpToolResult, toolName) {
        // Converts MCP tool result (which can be complex object with 'content' array or direct image)
        // into Anthropic's expected format for the 'content' field of a 'tool_result' block.
        // This can be a string or an array of content blocks (e.g., text, image).
        
        this.logger.debug(`[AnthropicProvider.convertToolResult] Raw MCP Result for ${toolName}:`, JSON.stringify(mcpToolResult, (key, value) => {
            if ((key === 'base64Data' || key === 'data') && typeof value === 'string' && value.length > 100) return `[base64_len:${value.length}]`;
            return value;
          }, 2));

        if (typeof mcpToolResult === 'string') {
            return mcpToolResult; // Simple string result
        }

        let anthropicContent = [];

        // Case 1: mcpToolResult has a 'content' array (standard MCP multi-part)
        if (mcpToolResult && mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
            for (const part of mcpToolResult.content) {
                if (part.type === 'text' && typeof part.text === 'string') {
                    anthropicContent.push({ type: 'text', text: part.text });
                } else if (part.type === 'image' && part.data && part.mimeType) {
                    let base64Data = part.data;
                    const base64PrefixMatch = /^data:[a-zA-Z0-9\/+]+;base64,/;
                    if (base64PrefixMatch.test(base64Data)) {
                        base64Data = base64Data.replace(base64PrefixMatch, '');
                    }
                    anthropicContent.push({
                        type: 'image',
                        source: {
                            type: 'base64',
                            media_type: part.mimeType,
                            data: base64Data,
                        },
                    });
                } else if (part.type === 'data' && typeof part.data === 'object' && part.data !== null) {
                    // If there's a structured data part, stringify it and put in a text block
                    anthropicContent.push({ type: 'text', text: JSON.stringify(part.data) });
                }
            }
        // Case 2: mcpToolResult has a direct 'image' property
        } else if (mcpToolResult && mcpToolResult.image && typeof mcpToolResult.image.base64Data === 'string' && mcpToolResult.image.mimeType) {
            let base64Data = mcpToolResult.image.base64Data;
            const base64PrefixMatch = /^data:[a-zA-Z0-9\/+]+;base64,/;
            if (base64PrefixMatch.test(base64Data)) {
                base64Data = base64Data.replace(base64PrefixMatch, '');
            }
            anthropicContent.push({
                type: 'image',
                source: {
                    type: 'base64',
                    media_type: mcpToolResult.image.mimeType,
                    data: base64Data,
                },
            });
            // Include other properties from mcpToolResult (excluding the image itself) as text
            const { image, ...otherProps } = mcpToolResult;
            if (Object.keys(otherProps).length > 0) {
                 anthropicContent.push({ type: 'text', text: JSON.stringify(otherProps) });
            }

        // Case 3: mcpToolResult is some other object - stringify it.
        } else if (typeof mcpToolResult === 'object' && mcpToolResult !== null) {
            return JSON.stringify(mcpToolResult);
        } else {
            // Fallback for unexpected types, or if processing above yielded nothing
            return String(mcpToolResult);
        }
        
        // If anthropicContent array is empty after processing, return an empty string or a placeholder.
        // Or, if it contains only one text block, Anthropic might prefer just the string content.
        if (anthropicContent.length === 0) {
            this.logger.warn(`[AnthropicProvider.convertToolResult] Tool result for ${toolName} converted to empty content. Original:`, mcpToolResult);
            return ""; // Or some placeholder like JSON.stringify({status: "empty result"})
        }
        // Anthropic's 'content' for tool_result can be a string if it's simple text.
        // If we have one text block, extract its text. Otherwise, return the array of blocks.
        if (anthropicContent.length === 1 && anthropicContent[0].type === 'text') {
            return anthropicContent[0].text;
        }

        return anthropicContent; // Array of content blocks
    }

    /**
     * Helper to parse a data URL string (e.g., from OpenAI image_url)
     * @param {string} dataUrl The data URL string.
     * @returns {{mimeType: string|null, base64Data: string|null}}
     * @private
     */
    _parseDataUrl(dataUrl) {
        const match = dataUrl.match(/^data:([\w\/\-\.]+);base64,(.*)$/);
        if (match) {
            return { mimeType: match[1], base64Data: match[2] };
        }
        this.logger.warn(`[AnthropicProvider._parseDataUrl] Failed to parse data URL: ${dataUrl.substring(0, 50)}...`);
        return { mimeType: null, base64Data: null };
    }

    _convertUserContentForAnthropic(messageContent, toolCalls) {
        const content = [];
        if (typeof messageContent === 'string') {
            content.push({ type: 'text', text: messageContent });
        } else if (Array.isArray(messageContent)) {
            // Handle mixed content, potentially from other providers like OpenAI
            messageContent.forEach(part => {
                if (part.type === 'text' && typeof part.text === 'string') {
                    content.push({ type: 'text', text: part.text });
                } else if (part.type === 'image' && part.source && part.source.type === 'base64') {
                    // Pass through existing Anthropic image format
                    content.push(part);
                } else if (part.type === 'image_url' && part.image_url && typeof part.image_url.url === 'string') {
                    // Convert OpenAI's image_url to Anthropic's image format
                    const { mimeType, base64Data } = this._parseDataUrl(part.image_url.url);
                    if (mimeType && base64Data) {
                        content.push({
                            type: 'image',
                            source: {
                                type: 'base64',
                                media_type: mimeType,
                                data: base64Data,
                            },
                        });
                        this.logger.debug(`[AnthropicProvider._convertUserContent] Converted OpenAI image_url (type: ${mimeType}) to Anthropic image part.`);
                    } else {
                        this.logger.warn(`[AnthropicProvider._convertUserContent] Could not parse or convert image_url part: ${JSON.stringify(part.image_url)}`);
                    }
                } else if (part.type === 'image_mcp' && part.source && part.source.media_type && part.source.data) {
                    // Handle the new image_mcp type
                    content.push({
                        type: 'image',
                        source: {
                            type: 'base64',
                            media_type: part.source.media_type,
                            data: part.source.data // Assuming data is already raw base64
                        }
                    });
                    this.logger.debug(`[AnthropicProvider._convertUserContent] Converted image_mcp (type: ${part.source.media_type}) to Anthropic image part.`);
                } else {
                    this.logger.warn(`[AnthropicProvider._convertUserContent] Skipping unknown part type in user message content array: ${part.type}`);
                }
            });
        }
        // ... existing tool_calls processing ...
        return content;
    }
} 