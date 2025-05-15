import { GoogleGenerativeAI } from '@google/generative-ai';
import { BaseProvider } from './base-provider.js';
import { logger as defaultLogger } from '../utils/logging.js'; // Keep for direct use if a method becomes static or for default in helpers

export class GoogleProvider extends BaseProvider {
    constructor(apiKey, modelName, systemMessage, allTools, logger, initialConversationHistory = []) {
        super(apiKey, modelName, systemMessage, allTools, logger);
        this.googleAI = new GoogleGenerativeAI(this.apiKey);
        this.generativeModel = null; // Will be set in initialize
        // chatSession (this.chatSession) is inherited from BaseProvider and will be set in initialize
        this.initialize(initialConversationHistory); // Auto-initialize upon construction, passing history
    }

    async initialize(initialConversationHistory = []) {
        this.logger.info(`[GoogleProvider] Initializing for model: ${this.modelName}`);
        const toolDeclarations = this.getToolDeclarations(); // Uses class method
        this.generativeModel = this.googleAI.getGenerativeModel({
            model: this.modelName,
            tools: toolDeclarations.length > 0 ? [{ functionDeclarations: toolDeclarations }] : undefined,
            systemInstruction: { role: "system", parts: [{text: this.systemMessage}]}
        });
        
        // Use the public, corrected formatConversation method
        const initialChatHistoryForSDK = this.formatConversation(initialConversationHistory); 
        this.logger.debug(`[GoogleProvider] Initializing new chat session with history for startChat:`, JSON.stringify(initialChatHistoryForSDK, null, 2));
        this.chatSession = this.generativeModel.startChat({
            history: initialChatHistoryForSDK,
        });
        this.logger.info(`[GoogleProvider] Initialized successfully.`);
    }

    async reconfigure(newSystemMessage, newAllTools) {
        this.logger.info(`[GoogleProvider] Reconfiguring...`);
        const oldSystemMessage = this.systemMessage;
        const oldModelName = this.modelName; // Keep old model name

        await super.reconfigure(newSystemMessage, newAllTools); // Updates this.systemMessage, this.allTools

        if (newSystemMessage && newSystemMessage !== oldSystemMessage) {
            this.logger.info(`[GoogleProvider] System message changed or tools updated. Re-initializing chat model and session.`);
            // Re-initialize the model and chat session with the new system message/tools
            // This effectively calls a slimmed down version of initialize focusing on model and chat session recreation
            const toolDeclarations = this.getToolDeclarations();
            this.generativeModel = this.googleAI.getGenerativeModel({
                model: this.modelName, // Use current or potentially new modelName if that logic is added
                tools: toolDeclarations.length > 0 ? [{ functionDeclarations: toolDeclarations }] : undefined,
                systemInstruction: { role: "system", parts: [{text: this.systemMessage}]}
            });
            // When re-initializing chat after /setsystem, we should use the current conversation history
            // This requires MCPClient to pass its current conversation to the reconfigure method, 
            // or for the provider to have a way to access it or be given it.
            // For now, let's assume MCPClient will handle re-passing history on /setsystem by re-creating the provider instance
            // Or, if reconfigure is called with history:
            // const currentFormattedHistory = this._formatConversationForChat(MCPClient.conversation); 
            // this.chatSession = this.generativeModel.startChat({ history: currentFormattedHistory });
            // For simplicity of this step, we assume setLLMProvider in MCPClient will create a new provider instance on /setsystem for Google.
            // Thus, the `initialize` method will be called with the fresh system message.
            // If MCPClient is modified to call `reconfigure` instead, then `reconfigure` here must handle history correctly.
            // For now, the reconfigure will just re-init the model, and expect `setLLMProvider` to handle history via new instance.
            this.logger.info("[GoogleProvider] Model re-initialized with new system instruction/tools. Chat session needs to be restarted by setLLMProvider with history.");
            // To fully re-initialize chat with history, MCPClient would typically call setLLMProvider, which creates a new Provider instance.
            // So, direct re-initialization of chatSession here might be redundant if MCPClient does that.
            // Let's simplify: reconfigure updates the model, MCPClient handles creating a new provider instance for chat history.
        } else {
            this.logger.info("[GoogleProvider] No system message change, model configuration remains.");
        }
    }

    // Internal helper for schema transformation, using instance logger
    _transformMcpSchemaToGemini(mcpSchema, toolName = 'unknown') {
        if (!mcpSchema || typeof mcpSchema !== 'object' || Object.keys(mcpSchema).length === 0) {
            this.logger.debug(`[GoogleProvider Schema Transform] Tool '${toolName}' has no inputSchema or it's empty.`);
            return { type: 'OBJECT', properties: {} };
        }
        let geminiSchema = JSON.parse(JSON.stringify(mcpSchema)); // Simplified clone
        const recursivelyProcessNode = (node, path = '') => {
            if (typeof node !== 'object' || node === null) return node;
            if (node.type && typeof node.type === 'string') node.type = node.type.toUpperCase();
            else if (node.type && Array.isArray(node.type)) {
                const firstNonNullType = node.type.find(t => t !== null && t !== 'null');
                node.type = firstNonNullType ? String(firstNonNullType).toUpperCase() : 'STRING';
                if (node.type !== 'STRING' && node.format) delete node.format;
            }
            if (node.type === 'STRING' && node.format && node.format !== 'date-time' && node.format !== 'enum') {
                this.logger.debug(`[GoogleProvider Schema Transform] Tool '${toolName}', Path '${path}': Removing unsupported format '${node.format}'.`);
                delete node.format;
            }
            if (node.properties) Object.keys(node.properties).forEach(propName => node.properties[propName] = recursivelyProcessNode(node.properties[propName], `${path}.${propName}`));
            if (node.type === 'ARRAY' && node.items) node.items = recursivelyProcessNode(node.items, `${path}.items`);
            delete node['$schema'];
            return node;
        };
        const processedSchema = recursivelyProcessNode(geminiSchema, 'root');
        delete processedSchema.additionalProperties;
        if (processedSchema.properties && processedSchema.type !== 'OBJECT') processedSchema.type = 'OBJECT';
        else if (!processedSchema.type && processedSchema.properties) processedSchema.type = 'OBJECT';
        else if (!processedSchema.properties && (!processedSchema.type || Object.keys(processedSchema).length === 0)) return { type: 'OBJECT', properties: {} };
        if (typeof processedSchema.properties === 'undefined' && processedSchema.type !== 'OBJECT' && Object.keys(processedSchema).length === 1 && processedSchema.type) return { type: 'OBJECT', properties: {} };
        return processedSchema;
    }

    getToolDeclarations() {
        if (!this.allTools || this.allTools.length === 0) {
            return [];
        }
        return this.allTools.map(tool => {
            const parameters = this._transformMcpSchemaToGemini(tool.inputSchema, tool.name);
            return {
                name: tool.name,
                description: tool.description || `Tool from ${tool.serverName}`,
                parameters: parameters,
            };
        });
    }
    
    //Formats conversation for CHAT HISTORY (used in initialize or re-init of chat)
    _formatConversationForChat(conversationHistory) {
        this.logger.debug("[GoogleProvider._formatConversationForChat] Processing history:", JSON.stringify(conversationHistory, null, 2));
        const chatHistoryForGoogle = [];
        for (const message of conversationHistory) {
            if (message.role === 'user') {
                chatHistoryForGoogle.push({ role: 'user', parts: [{ text: message.content }] });
            } else if (message.role === 'assistant') {
                const modelParts = [];
                if (message.content && typeof message.content === 'string' && message.content.trim() !== '') {
                    modelParts.push({ text: message.content });
                }
                if (message.tool_calls && message.tool_calls.length > 0) {
                    message.tool_calls.forEach(tc => modelParts.push({ functionCall: { name: tc.name, args: tc.args } }));
                }
                if (modelParts.length > 0) chatHistoryForGoogle.push({ role: 'model', parts: modelParts });
            }
            // Tool results are not included in the history for startChat per previous findings
        }
        this.logger.debug("[GoogleProvider._formatConversationForChat] Constructed history:", JSON.stringify(chatHistoryForGoogle, null, 2));
        return chatHistoryForGoogle;
    }

    // This is for formatting messages to *send*, which for Google is just the latest content or tool response parts.
    // The main `formatConversation` is used by MCPClient to get history for re-init.
    // This method might not be directly used if sendMessage takes raw content.
    formatConversation(conversationHistory) {
        const chatHistoryForGoogle = [];
        this.logger.debug("[GoogleProvider.formatConversation] Processing conversation history for Google startChat/reconfigure:", JSON.stringify(conversationHistory, null, 2));

        for (let i = 0; i < conversationHistory.length; i++) {
            const message = conversationHistory[i];

            if (message.role === 'user') {
                const userParts = [];
                if (typeof message.content === 'string') {
                    userParts.push({ text: message.content });
                } else if (Array.isArray(message.content)) {
                    message.content.forEach(part => {
                        if (part.type === 'text' && typeof part.text === 'string') {
                            userParts.push({ text: part.text });
                        } else if (part.type === 'image_url' && part.image_url?.url) { // From OpenAI user message
                            const { mimeType, base64Data } = this._parseDataUrl(part.image_url.url);
                            if (mimeType && base64Data) {
                                userParts.push({
                                    inlineData: {
                                        mimeType: mimeType,
                                        data: base64Data
                                    }
                                });
                                this.logger.debug(`[GoogleProvider.formatConversation] Converted OpenAI image_url to inlineData for user message.`);
                            }
                        } else if (part.type === 'image' && part.source?.type === 'base64' && part.source?.media_type && part.source?.data) { // From Anthropic user message
                             userParts.push({
                                inlineData: {
                                    mimeType: part.source.media_type,
                                    data: part.source.data // Assuming data is already base64 clean
                                }
                            });
                            this.logger.debug(`[GoogleProvider.formatConversation] Converted Anthropic image to inlineData for user message.`);
                        } else if (part.type === 'image_mcp' && part.source && part.source.media_type && part.source.data) {
                            // Handle the new image_mcp type
                            userParts.push({
                                inlineData: {
                                    mimeType: part.source.media_type,
                                    data: part.source.data // Assuming data is already raw base64
                                }
                            });
                            this.logger.debug(`[GoogleProvider.formatConversation] Converted image_mcp (type: ${part.source.media_type}) to inlineData for user message.`);
                        } else {
                            this.logger.warn(`[GoogleProvider.formatConversation] Skipping unknown part in user message content array: ${JSON.stringify(part).substring(0,100)}`);
                        }
                    });
                }
                if (userParts.length > 0) {
                    chatHistoryForGoogle.push({ role: 'user', parts: userParts });
                }
            } else if (message.role === 'assistant') {
                const modelParts = [];
                if (message.content && typeof message.content === 'string' && message.content.trim() !== '') {
                    modelParts.push({ text: message.content });
                }
                if (message.tool_calls && message.tool_calls.length > 0) {
                    message.tool_calls.forEach(tc => {
                        modelParts.push({ functionCall: { name: tc.name, args: tc.args || {} } });
                    });
                }
                if (modelParts.length > 0) {
                    chatHistoryForGoogle.push({ role: 'model', parts: modelParts });
                }
            } else if (message.role === 'tool' && message.name && message.tool_call_id) {
                // Process tool results from MCPClient history into Google's FunctionResponse format
                const functionResponsePart = this.convertToolResult(message.content, message.name);
                // For startChat history, function responses should be in a message with role: 'function'
                chatHistoryForGoogle.push({ role: 'function', parts: [functionResponsePart] });
                this.logger.debug(`[GoogleProvider.formatConversation] Added MCP tool result for '${message.name}' as a Google 'function' role turn to chat history.`);
            }
        }
        this.logger.debug("[GoogleProvider.formatConversation] Constructed history for startChat/reconfigure:", JSON.stringify(chatHistoryForGoogle, null, 2));
        return chatHistoryForGoogle;
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
    this.logger.warn(`[GoogleProvider._parseDataUrl] Failed to parse data URL: ${dataUrl.substring(0, 50)}...`);
    return { mimeType: null, base64Data: null };
  }

    async sendMessage(content, isContinuation = false) { // isContinuation is for conceptual alignment, not directly used by Gemini's sendMessage
        if (!this.chatSession) {
            this.logger.error("[GoogleProvider.sendMessage] Chat session not initialized!");
            throw new Error("GoogleProvider chat session not initialized.");
        }
        
        let partsToSend;
        if (typeof content === 'string') {
            partsToSend = [{ text: content }];
            this.logger.debug(`[GoogleProvider.sendMessage] Sending string content as parts: "${content}"`);
        } else if (Array.isArray(content)) {
            partsToSend = content;
            this.logger.debug(`[GoogleProvider.sendMessage] Sending array of parts:`, JSON.stringify(partsToSend, null, 2));
        } else {
            this.logger.error("[GoogleProvider.sendMessage] Invalid content type. Must be a string or an array of parts.", content);
            throw new Error("Invalid content type for GoogleProvider.sendMessage. Must be a string or an array of parts.");
        }
        
        // The `tools` argument is ignored here as tools are configured at model initialization for Google
        return this.chatSession.sendMessage(partsToSend);
    }

    extractTextContent(llmResponse) {
        let text = '';
        if (llmResponse?.response?.candidates?.[0]?.content?.parts) {
            for (const part of llmResponse.response.candidates[0].content.parts) {
                if (part.text) text += part.text;
            }
        }
        return text;
    }

    extractToolCalls(llmResponse) {
        const toolCalls = [];
        if (llmResponse?.response?.candidates?.[0]?.content?.parts) {
            for (const part of llmResponse.response.candidates[0].content.parts) {
                if (part.functionCall) {
                    toolCalls.push({
                        id: part.functionCall.name + '_' + Date.now(), 
                        name: part.functionCall.name,
                        args: part.functionCall.args || {}
                    });
                }
            }
        }
        return toolCalls.length > 0 ? toolCalls : null;
    }

    // This is for converting MCP tool result to the FunctionResponse PART for Google
    convertToolResult(mcpToolResult, toolName) {
        this.logger.debug(`[GoogleProvider Convert Tool Result] Raw MCP Result for ${toolName}:`, JSON.stringify(mcpToolResult, (key, value) => {
            if ((key === 'base64Data' || key === 'data') && typeof value === 'string' && value.length > 100) return `[base64_len:${value.length}]`;
            return value;
          }, 2));
          let responseData = null;
          if (mcpToolResult && mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
            for (const part of mcpToolResult.content) {
              if (part.type === 'text' && typeof part.text === 'string') {
                try {
                  const parsedJson = JSON.parse(part.text);
                  this.logger.debug(`[GoogleProvider Convert Tool Result] Parsed JSON from text part. Type: ${typeof parsedJson}, IsArray: ${Array.isArray(parsedJson)}, Snippet: ${JSON.stringify(parsedJson).substring(0,100)}`);
                  if (Array.isArray(parsedJson)) responseData = { result: parsedJson };
                  else if (typeof parsedJson === 'object' && parsedJson !== null) responseData = parsedJson;
                  else responseData = { value: parsedJson };
                  break;
                } catch (e) { /* Not JSON, continue */ }
              }
            }
            if (responseData === null) {
              const dataPart = mcpToolResult.content.find(p => p.type === 'data' && typeof p.data === 'object' && p.data !== null);
              if (dataPart) responseData = dataPart.data;
            }
            if (responseData === null) {
              const imageContentPart = mcpToolResult.content.find(p => p.type === 'image' && p.data && p.mimeType);
              const textContentPart = mcpToolResult.content.find(p => p.type === 'text' && p.text);
              if (imageContentPart) responseData = {
                image_captured_summary: {
                    mime_type: imageContentPart.mimeType,
                    status: "Image data provided separately to the model.",
                    description: textContentPart ? textContentPart.text.substring(0,100)+"..." : "Image content."
                }
              };
            }
            if (responseData === null && mcpToolResult.content.length > 0) responseData = { mcpToolCallResponseContentSummary: `Processed ${mcpToolResult.content.length} parts.` };
          } else if (typeof mcpToolResult === 'object' && mcpToolResult !== null) {
            if (Array.isArray(mcpToolResult)) responseData = { tool_result_array: mcpToolResult };
            else if (mcpToolResult.image && typeof mcpToolResult.image.base64Data === 'string') {
                const { base64Data, ...imageMetadata } = mcpToolResult.image; const otherProps = {...mcpToolResult}; delete otherProps.image;
                responseData = {...otherProps, image_summary: {...imageMetadata, status: "Image data provided separately to the model."}};
            } else responseData = mcpToolResult;
          } else if (typeof mcpToolResult === 'string') responseData = { result: mcpToolResult };
          else responseData = { result: String(mcpToolResult) };
          if (responseData === null || typeof responseData !== 'object') responseData = { error: "Failed to process tool result into valid object."};
          return { functionResponse: { name: toolName, response: responseData } }; // This IS the FunctionResponse PART
    }

    extractImageFromToolResult(mcpToolResult) {
        let base64ImageData = null;
        let mimeType = null;

        if (mcpToolResult && typeof mcpToolResult === 'object' && mcpToolResult !== null) {
            // Case 1: Image directly under `image` property
            if (mcpToolResult.image && typeof mcpToolResult.image.base64Data === 'string' && mcpToolResult.image.mimeType) {
                base64ImageData = mcpToolResult.image.base64Data;
                mimeType = mcpToolResult.image.mimeType;
                this.logger.debug("[GoogleProvider.extractImageFromToolResult] Found image in toolResult.image");
            // Case 2: Image within `content` array
            } else if (mcpToolResult.content && Array.isArray(mcpToolResult.content)) {
                const imageContentPart = mcpToolResult.content.find(
                    p => p.type === 'image' && p.data && typeof p.data === 'string' && p.mimeType
                );
                if (imageContentPart) {
                    base64ImageData = imageContentPart.data;
                    mimeType = imageContentPart.mimeType;
                    this.logger.debug("[GoogleProvider.extractImageFromToolResult] Found image in toolResult.content array");
                }
            }
        }

        if (base64ImageData && mimeType) {
            // Remove data URI prefix if present
            const base64PrefixMatch = /^data:[a-zA-Z0-9\\/+]+;base64,/;
            if (base64PrefixMatch.test(base64ImageData)) {
                base64ImageData = base64ImageData.replace(base64PrefixMatch, '');
            }
            return { base64ImageData, mimeType };
        }
        this.logger.debug("[GoogleProvider.extractImageFromToolResult] No image found in tool result.");
        return { base64ImageData: null, mimeType: null };
    }

    prepareImageMessageParts(base64ImageData, mimeType, toolName, fullConversationHistory) {
        if (!base64ImageData || !mimeType) {
            this.logger.error("[GoogleProvider.prepareImageMessageParts] Missing base64ImageData or mimeType.");
            throw new Error("Missing image data or mimeType for prepareImageMessageParts.");
        }

        const imagePartForGemini = {
            inlineData: { mimeType: mimeType, data: base64ImageData }
        };
        
        let textForImagePrompt = `The tool '${toolName}' returned the accompanying image. Please analyze it and respond to the ongoing request.`; 
        
        // New robust logic: Iterate backwards to find the relevant user message.
        // The `fullConversationHistory` when this is called by MCPClient looks like:
        // [..., some_user_message (U_orig), assistant_called_tool (A_tc), tool_result (T_res), assistant_response_to_tool_data (A_sum)]
        // We want to find U_orig.
        // We expect at least these last three messages before U_orig could be found.
        if (fullConversationHistory && fullConversationHistory.length >= 2) { // Need at least T_res and A_sum to look before them.
            let foundUserMessage = null;
            // Start searching from before the last two known messages (T_res, A_sum from MCPClient's perspective, which are history.length-1 and history.length-2 by the time this is called)
            // However, MCPClient's `this.conversation` has [..., U_orig, A_tc, T_res], then MCPClient calls handleLLMResponse for A_sum which adds it.
            // So when `prepareImageMessageParts` is called, `fullConversationHistory` is [..., U_orig, A_tc, T_res, A_sum]. Length is L.
            // T_res is at L-2. A_tc is at L-3. U_orig is at L-4 or earlier if there were other messages.
            // We are looking for the last 'user' message that comes *before* the 'assistant' message that contained the tool_call for `toolName`.
            
            let assistantToolCallMessageIndex = -1;
            for (let i = fullConversationHistory.length - 1; i >= 0; i--) {
                const msg = fullConversationHistory[i];
                if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.some(tc => tc.name === toolName)) {
                    // This is likely the assistant message that invoked the current tool.
                    // However, if the same tool was called multiple times, this might not be the *specific* invocation for *this* image result.
                    // The current structure of MCPClient doesn't give us the ID of the tool_call that led to this image.
                    // For now, we assume the last assistant message calling *any* tool (or specifically this toolName) is the one.
                    // A more robust solution would involve MCPClient passing the specific tool_call_id that this image result corresponds to.
                    assistantToolCallMessageIndex = i;
                    break;
                }
            }

            if (assistantToolCallMessageIndex > 0) { // If we found such an assistant message, and it's not the first message
                for (let i = assistantToolCallMessageIndex - 1; i >= 0; i--) {
                    const msg = fullConversationHistory[i];
                    if (msg.role === 'user' && typeof msg.content === 'string' && msg.content.trim() !== '') {
                        foundUserMessage = msg.content;
                        this.logger.debug(`[GoogleProvider.prepareImageMessageParts] Found user message for prompt context at index ${i} (before assistant tool call at ${assistantToolCallMessageIndex}).`);
                        break;
                    }
                }
            }

            if (foundUserMessage) {
                textForImagePrompt = `The tool '${toolName}' returned an image. Based on your request: "${foundUserMessage}", please use this image to formulate your response.`;
            } else {
                this.logger.warn(`[GoogleProvider.prepareImageMessageParts] Could not robustly find a preceding user message for context. Using generic image prompt.`);
            }
        } else {
            this.logger.warn(`[GoogleProvider.prepareImageMessageParts] Conversation history too short (${fullConversationHistory?.length}) for robust user message search. Using generic prompt.`);
        }

        const textPromptPart = { text: textForImagePrompt };
        
        this.logger.debug(`[GoogleProvider.prepareImageMessageParts] Prepared parts: image (mime: ${mimeType}), text prompt: "${textForImagePrompt}"`);
        return [imagePartForGemini, textPromptPart];
    }
}

// Remove the old standalone exported functions as they are now methods or part of the class logic
// export function _transformMcpSchemaToGemini(...) {}
// export function getGoogleFunctionDeclarations(...) {}
// export function formatConversationForGoogle(...) {}
// export function extractTextContentFromGoogleResponse(...) {}
// export function extractToolCallsFromGoogleResponse(...) {}
// export function convertToolResultForGoogle(...) {} 