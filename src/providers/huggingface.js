import OpenAI from 'openai';
import { BaseProvider } from './base-provider.js';

// Known Hugging Face Inference Partner URL path segments
// The OpenAI SDK will append /chat/completions, so these paths should lead to the directory containing that.
const HF_PARTNER_URL_SEGMENTS = {
    "together": "/together/v1",                 // e.g., https://router.huggingface.co/together/v1
    "nscale": "/nscale/v1",                   // e.g., https://router.huggingface.co/nscale/v1
    "novita": "/novita/v3/openai",            // e.g., https://router.huggingface.co/novita/v3/openai
    "fireworks-ai": "/fireworks-ai/inference/v1",       
    "cerebras": "/cerebras/v1",
    "sambanova": "/sambanova/v1"
    // Example for HF's own, confirm actual path: "hf-inference": "/hf-inference/v1", 

    // Add more partners and their specific paths as they are verified.
    // Some might be just '/<partner_name>/v1'
    // Others might be '/<partner_name>/<version_string>/openai'
};

const HF_ROUTER_BASE = "https://router.huggingface.co";

export class HuggingFaceProvider extends BaseProvider {
    constructor(apiKey, modelName, partnerName, systemMessage, allTools, logger, initialConversationHistory = [], imageAnalysisPromptSuffix) {
        super(apiKey, modelName, systemMessage, allTools, logger, imageAnalysisPromptSuffix);
        this.partnerName = partnerName?.toLowerCase();

        if (!this.partnerName || !HF_PARTNER_URL_SEGMENTS[this.partnerName]) {
            const errorMsg = `Hugging Face partner name '${partnerName}' is not provided or not supported. Supported: ${Object.keys(HF_PARTNER_URL_SEGMENTS).join(', ')}.`;
            this.logger.error(`[HuggingFaceProvider] ${errorMsg}`);
            throw new Error(errorMsg);
        }
        
        const partnerSegment = HF_PARTNER_URL_SEGMENTS[this.partnerName];
        const baseURL = `${HF_ROUTER_BASE}${partnerSegment}`;
        
        this.logger.info(`[HuggingFaceProvider] Initializing for model: ${this.modelName}, Partner: ${this.partnerName}, BaseURL: ${baseURL}`);
        
        this.llmClient = new OpenAI({ 
            apiKey: this.apiKey, 
            baseURL: baseURL 
        });
        
        // imageAnalysisPromptSuffix is already stored in super
        this.initialize(initialConversationHistory);
    }

    async initialize(initialConversationHistory = []) {
        this.logger.info(`[HuggingFaceProvider] Initialized for model: ${this.modelName} with partner: ${this.partnerName}. Client configured.`);
        // No specific async initialization needed for OpenAI-compatible clients itself usually.
    }

    async reconfigure(newSystemMessage, newAllTools) {
        await super.reconfigure(newSystemMessage, newAllTools);
        this.logger.info(`[HuggingFaceProvider] Reconfigured. New system message and tools will be used on next API call.`);
    }

    updateImageAnalysisPromptSuffix(newSuffix) {
        super.updateImageAnalysisPromptSuffix(newSuffix);
        // No partner-specific action needed beyond BaseProvider storing it.
    }

    getToolDeclarations() {
        // Using OpenAI's tool declaration format
        this.logger.debug(`[HuggingFaceProvider.getToolDeclarations]this.allTools count: ${this.allTools ? this.allTools.length : 'null or undefined'}`);
        if (!this.allTools || this.allTools.length === 0) {
            return undefined; // OpenAI SDK expects undefined if no tools
        }
        const declarations = this.allTools.map(tool => ({
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description || `Tool from ${tool.serverName}`,
                parameters: tool.inputSchema || { type: "object", properties: {} }
            }
        }));
        this.logger.debug('[HuggingFaceProvider.getToolDeclarations] Generated declarations:', JSON.stringify(declarations, null, 2));
        return declarations;
    }

    _convertToolResultForPartner(mcpToolResult, toolName) {
        // Using OpenAI's expected string format for tool results
        this.logger.debug(`[HuggingFaceProvider._convertToolResult] Raw MCP Result for ${toolName} (Partner: ${this.partnerName}):`, JSON.stringify(mcpToolResult, (key, value) => {
            if ((key === 'base64Data' || key === 'data') && typeof value === 'string' && value.length > 100) return `[base64_len:${value.length}]`;
            return value;
        }, 2));

        let hasImage = false;
        if (mcpToolResult && typeof mcpToolResult === 'object' && mcpToolResult !== null) {
            if (mcpToolResult.image && typeof mcpToolResult.image.base64Data === 'string' && mcpToolResult.image.mimeType) {
                hasImage = true;
            } else if (mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
                if (mcpToolResult.content.some(p => p.type === 'image' && p.data && typeof p.data === 'string' && p.mimeType)) {
                    hasImage = true;
                }
            }
        }

        // TODO: Check if this specific partner/model supports vision. For now, assume it does if it's similar to OpenAI.
        // If not, this should return a summary indicating image was present but not sent.
        if (hasImage) {
            // For OpenAI-compatible VLM, the client (MCPClient) handles the two-step image flow.
            // The tool result message itself should just be a string summary.
            this.logger.debug(`[HuggingFaceProvider._convertToolResult] Tool result for '${toolName}' contains image data. Returning neutral placeholder for partner ${this.partnerName}.`);
            return "Tool execution was successful. Output includes image data, which will be processed separately if applicable by the client.";
        }

        if (typeof mcpToolResult === 'string') return mcpToolResult;
        if (mcpToolResult && mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
            // Logic from OpenAIProvider to extract/summarize text or data
            let firstTextPartContent = null;
            for (const part of mcpToolResult.content) {
                if (part.type === 'text' && typeof part.text === 'string') {
                    if (firstTextPartContent === null) firstTextPartContent = part.text;
                    try { JSON.parse(part.text); return part.text; } catch (e) { /* Not JSON */ }
                }
            }
            const dataPart = mcpToolResult.content.find(p => p.type === 'data' && typeof p.data !== 'undefined');
            if (dataPart) return JSON.stringify(dataPart.data);
            if (firstTextPartContent !== null) return firstTextPartContent;
        }
        return JSON.stringify(mcpToolResult);
    }

    extractImageFromToolResult(toolCallResult) {
        // Reusing OpenAIProvider's logic
        let base64ImageData = null;
        let mimeType = null;
        if (toolCallResult && typeof toolCallResult === 'object' && toolCallResult !== null) {
            if (toolCallResult.image && typeof toolCallResult.image.base64Data === 'string' && toolCallResult.image.mimeType) {
                base64ImageData = toolCallResult.image.base64Data;
                mimeType = toolCallResult.image.mimeType;
            } else if (toolCallResult.content && Array.isArray(toolCallResult.content)) {
                const imageContentPart = toolCallResult.content.find(p => p.type === 'image' && p.data && typeof p.data === 'string' && p.mimeType);
                if (imageContentPart) {
                    base64ImageData = imageContentPart.data;
                    mimeType = imageContentPart.mimeType;
                }
            }
        }
        if (base64ImageData && mimeType) {
            const base64PrefixMatch = /^data:[a-zA-Z0-9\/\+]+;base64,/;
            if (base64PrefixMatch.test(base64ImageData)) base64ImageData = base64ImageData.replace(base64PrefixMatch, '');
            this.logger.debug(`[HuggingFaceProvider.extractImageFromToolResult] Extracted image (Partner: ${this.partnerName}): ${mimeType}, data length: ${base64ImageData.length}`);
        } else {
            this.logger.debug(`[HuggingFaceProvider.extractImageFromToolResult] No image found in tool result (Partner: ${this.partnerName}).`);
        }
        return { base64ImageData, mimeType };
    }

    prepareImageMessageContent(base64ImageData, mimeType, textPrompt) {
        // Reusing OpenAIProvider's logic
        return [
            { type: 'text', text: textPrompt },
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64ImageData}` } },
        ];
    }

    formatConversation(conversationHistory) {
        // Reusing OpenAIProvider's conversation formatting logic
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
                let processedUserContent = message.content;
                if (Array.isArray(message.content)) {
                    processedUserContent = message.content.map(part => {
                        if (part.type === 'image_mcp' && part.source && part.source.media_type && part.source.data) {
                            return { type: 'image_url', image_url: { url: `data:${part.source.media_type};base64,${part.source.data}` } };
                        } else if (part.type === 'image' && part.source && part.source.type === 'base64' && part.source.media_type && part.source.data) {
                            return { type: 'image_url', image_url: { url: `data:${part.source.media_type};base64,${part.source.data}`}};
                        }
                        // Pass through 'text' and 'image_url' as is, assuming they are correctly formatted for OpenAI
                        return part; 
                    }).filter(part => part.type === 'text' || part.type === 'image_url'); // Ensure only valid parts remain
                }
                if (typeof processedUserContent === 'string' || (Array.isArray(processedUserContent) && processedUserContent.length > 0)) {
                    formattedMessages.push({ role: 'user', content: processedUserContent });
                }
            } else if (message.role === 'assistant') {
                const assistantMessagePayload = { role: 'assistant', content: (message.content && typeof message.content === 'string' && message.content.trim() !== '') ? message.content : null };
                if (message.tool_calls && message.tool_calls.length > 0) {
                    assistantMessagePayload.tool_calls = message.tool_calls.map(tc => ({
                        id: tc.id, type: 'function', function: { name: tc.name, arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args || {}) }
                    }));
                }
                if (assistantMessagePayload.content || assistantMessagePayload.tool_calls) {
                    formattedMessages.push(assistantMessagePayload);
                }
            } else if (message.role === 'tool') {
                formattedMessages.push({
                    role: 'tool',
                    tool_call_id: message.tool_call_id,
                    content: this._convertToolResultForPartner(message.content, message.name)
                });
                // Two-step image handling (adding synthetic user message with image) is done by MCPClient for OpenAI-like providers.
                // If this partner/model is VLM and the tool result contained an image, MCPClient's `continueConversation`
                // will call `this.extractImageFromToolResult` and `this.prepareImageMessageContent`.
            }
        }
        this.logger.debug(`[HuggingFaceProvider.formatConversation] Formatted messages for partner ${this.partnerName}:`, JSON.stringify(formattedMessages, null, 2));
        return formattedMessages;
    }

    async sendMessage(formattedMessages) {
        if (!this.llmClient) {
            const errorMsg = `[HuggingFaceProvider.sendMessage] LLM client not initialized for partner ${this.partnerName}!`;
            this.logger.error(errorMsg);
            throw new Error(errorMsg);
        }

        const toolDeclarations = this.getToolDeclarations();
        const requestToLLM = {
            model: this.modelName,
            messages: formattedMessages,
        };
        if (toolDeclarations) {
            requestToLLM.tools = toolDeclarations;
            requestToLLM.tool_choice = "auto";
        }

        this.logger.debug(`[HuggingFaceProvider.sendMessage] Sending request to partner ${this.partnerName} (model: ${this.modelName}, messages: ${requestToLLM.messages.length}, tools: ${requestToLLM.tools ? requestToLLM.tools.length : 0}):`, JSON.stringify(requestToLLM, null, 2));
        return this.llmClient.chat.completions.create(requestToLLM);
    }

    extractTextContent(llmResponse) {
        // Reusing OpenAIProvider's logic
        if (llmResponse.choices && llmResponse.choices.length > 0 && llmResponse.choices[0].message) {
            return llmResponse.choices[0].message.content || '';
        }
        return '';
    }

    extractToolCalls(llmResponse) {
        // Reusing OpenAIProvider's logic
        if (llmResponse.choices && llmResponse.choices.length > 0 && llmResponse.choices[0].message && llmResponse.choices[0].message.tool_calls) {
            return llmResponse.choices[0].message.tool_calls.map(tc => ({
                id: tc.id,
                name: tc.function.name,
                args: JSON.parse(tc.function.arguments) // Assuming arguments are always JSON string
            })).filter(tc => tc.args !== null); // Filter out calls where arg parsing failed if JSON.parse throws & is caught
        }
        return null;
    }
    
    convertToolResult(mcpToolResult, toolName) {
        // Public interface for BaseProvider
        return this._convertToolResultForPartner(mcpToolResult, toolName);
    }
} 