export class BaseProvider {
    constructor(apiKey, modelName, systemMessage, allTools, logger, imageAnalysisPromptSuffix) {
        if (this.constructor === BaseProvider) {
            throw new Error("Abstract classes can't be instantiated.");
        }
        this.apiKey = apiKey;
        this.modelName = modelName;
        this.systemMessage = systemMessage;
        this.allTools = allTools; 
        this.logger = logger;
        this.imageAnalysisPromptSuffix = imageAnalysisPromptSuffix;
        this.llmClient = null; // The actual SDK client (e.g., Anthropic SDK, OpenAI SDK)
        this.chatSession = null; // For stateful providers like Google Gemini
    }

    async initialize() {
        throw new Error("Method 'initialize()' must be implemented.");
    }

    getToolDeclarations() {
        throw new Error("Method 'getToolDeclarations()' must be implemented.");
    }

    formatConversation(conversationHistory) {
        throw new Error("Method 'formatConversation()' must be implemented.");
    }

    async sendMessage(formattedConversation, tools) {
        throw new Error("Method 'sendMessage()' must be implemented.");
    }
    
    async sendToolResponse(toolResult, toolName, conversationHistory) {
        // This method will be specific to providers that have a distinct step for sending tool responses (like Google's image flow)
        // For others, the logic might be embedded in how formatConversation handles role: 'tool'
        // or this method might just format the tool result to be included in the next sendMessage.
        throw new Error("Method 'sendToolResponse()' must be implemented.");
    }

    extractTextContent(llmResponse) {
        throw new Error("Method 'extractTextContent()' must be implemented.");
    }

    extractToolCalls(llmResponse) {
        throw new Error("Method 'extractToolCalls()' must be implemented.");
    }

    convertToolResult(mcpToolResult, toolName) {
        throw new Error("Method 'convertToolResult()' must be implemented.");
    }

    // Helper to re-initialize chat if needed (e.g., for Google if system message changes)
    // Or to update tool configurations if they change dynamically.
    async reconfigure(newSystemMessage, newAllTools) {
        if (newSystemMessage) this.systemMessage = newSystemMessage;
        if (newAllTools) this.allTools = newAllTools;
        // Default implementation might just update properties. 
        // Providers like Google will need to override this to re-initialize their chat model with new system instructions/tools.
        this.logger.info(`[BaseProvider] Reconfigured with new system message/tools for ${this.constructor.name}.`);
        // Child classes should call super.reconfigure and then do their specific re-initialization.
    }

    updateImageAnalysisPromptSuffix(newSuffix) {
        this.imageAnalysisPromptSuffix = newSuffix;
        this.logger.info(`[${this.constructor.name}] Image analysis prompt suffix updated to: "${newSuffix}"`);
        // Individual providers can override if they need to do more than just store it.
    }
} 