#!/usr/bin/env node
import readline from 'readline';
import { spawn } from 'child_process';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import 'dotenv/config';
import { logger } from './utils/logging.js';
import fs from 'fs/promises';
import path from 'path';
import { createRequire } from 'module';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

// Import the new provider function
// import { getGoogleFunctionDeclarations as getGoogleFunctionDeclarationsFromProvider } from './providers/google.js'; // No longer needed directly here

// Import from Google provider module
// These are not needed here anymore as they will be accessed via the GoogleProvider instance.
/*
import {
    formatConversationForGoogle as formatConversationForGoogleProvider,
    extractTextContentFromGoogleResponse as extractTextContentFromGoogleProviderResponse,
    extractToolCallsFromGoogleResponse as extractToolCallsFromGoogleProviderResponse,
    convertToolResultForGoogle as convertToolResultForGoogleProvider
} from './providers/google.js';
*/

// Import from Anthropic provider module
// These are no longer needed here as they will be accessed via the AnthropicProvider instance.
/*
import {
    extractTextContentFromAnthropicResponse as extractTextContentFromAnthropicProviderResponse,
    extractToolCallsFromAnthropicResponse as extractToolCallsFromAnthropicProviderResponse,
    formatConversationForAnthropicProvider as formatConversationForAnthropicProviderFunc,
    convertToolResultForAnthropicProvider as convertToolResultForAnthropicProviderFunc
} from './providers/anthropic.js';
*/

// Import from OpenAI provider module
// These are no longer needed here as they will be accessed via the OpenAiProvider instance.
/*
import {
    extractTextContentFromOpenAIResponse as extractTextContentFromOpenAIProviderResponse,
    extractToolCallsFromOpenAIResponse as extractToolCallsFromOpenAIProviderResponse,
    formatConversationForOpenAI as formatConversationForOpenAIProviderFunc,
    // convertToolResultForOpenAI is used internally by formatConversationForOpenAIProviderFunc
} from './providers/openai.js';
*/

// Import Provider Classes
import { GoogleProvider } from './providers/google.js';
import { AnthropicProvider } from './providers/anthropic.js'; 
import { OpenAIProvider } from './providers/openai.js';    
import { DeepSeekProvider } from './providers/deepseek.js';

const require = createRequire(import.meta.url);

// Configure logging
logger.setLevel((process.env.LOG_LEVEL || 'debug').toLowerCase());

// System message for LLMs
const DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant that has access to various tools through MCP servers. Use these tools when appropriate to help the user.";

/**
 * MCP Client that supports both local and remote servers using the official MCP SDK
 * with proper image handling for Anthropic/Claude and initial support for OpenAI and Google Gemini.
 */
export class MCPClient {
  /**
   * Create a new MCPClient
   * 
   * @param {Object} config Configuration object
   * @param {string} config.llmProvider LLM provider to use ('anthropic', 'openai', or 'google')
   * @param {Array} config.mcpServers Array of MCP server configurations
   * @param {string} [config.systemMessage] Optional system message to override the default.
   * @param {number} [config.maxConversationHistoryLength] Optional limit for conversation history length (number of messages).
   */
  constructor(config) {
    this.config = config;
    this.mcpClients = new Map(); // Map of server name to MCP SDK Client
    this.transports = new Map(); // Map of server name to transport
    this.tools = new Map(); // Map of server name to tools array
    this.conversation = [];
    this.rl = null;
    
    // Initialize systemMessage: config.json > environment variable > default
    this.systemMessage = config.systemMessage || process.env.SYSTEM_MESSAGE || DEFAULT_SYSTEM_MESSAGE;
    logger.info(`Using system message: "${this.systemMessage.substring(0, 100)}${this.systemMessage.length > 100 ? '...' : ''}"`);

    // Initialize maxConversationHistoryLength
    this.maxConversationHistoryLength = null;
    const configHistoryLength = parseInt(config.maxConversationHistoryLength, 10);
    const envHistoryLength = parseInt(process.env.MAX_CONVERSATION_HISTORY_LENGTH, 10);

    if (!isNaN(configHistoryLength) && configHistoryLength > 0) {
      this.maxConversationHistoryLength = configHistoryLength;
    } else if (!isNaN(envHistoryLength) && envHistoryLength > 0) {
      this.maxConversationHistoryLength = envHistoryLength;
    }

    if (this.maxConversationHistoryLength) {
      logger.info(`Conversation history will be limited to ${this.maxConversationHistoryLength} messages.`);
    } else {
      logger.info('No limit set for conversation history length.');
    }

    // Configure console logging for tool results
    this.logToolResultsBase64Full = (config.logToolResultsBase64Full === true || process.env.LOG_TOOL_RESULTS_BASE64_FULL === 'true');
    logger.info(`Log full base64 in tool results: ${this.logToolResultsBase64Full}`);

    this.logToolCallVerbosity = config.logToolCallVerbosity || process.env.LOG_TOOL_CALL_VERBOSITY || 'default';
    if (!['minimal', 'default', 'debug'].includes(this.logToolCallVerbosity)) {
        logger.warn(`Invalid logToolCallVerbosity value: "${this.logToolCallVerbosity}". Defaulting to 'default'.`);
        this.logToolCallVerbosity = 'default';
    }
    logger.info(`Tool call console log verbosity: ${this.logToolCallVerbosity}`);

    // LLM Client SDK instances (will be managed by provider instances or cleared)
    this.anthropic = null; 
    this.openai = null;
    // No direct googleAI, geminiModel, googleChat on MCPClient anymore

    // Current active LLM provider instance
    this.currentLlmProviderInstance = null;
    // Model names for display/dynamic prompt still needed if not directly on provider instance
    this.model = null; // For Anthropic display
    this.openaiModel = null; // For OpenAI display
    this.geminiModel = null; // For Google display
    this.deepseekModel = null; // For DeepSeek display

    // Determine initial provider and model from config and environment variables
    const configuredProviderName = config.llmProvider?.toLowerCase();
    let modelForProvider = null;

    if (configuredProviderName === 'anthropic') {
      modelForProvider = process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-latest';
    } else if (configuredProviderName === 'openai') {
      modelForProvider = process.env.OPENAI_MODEL || 'gpt-4o';
    } else if (configuredProviderName === 'google') {
      modelForProvider = process.env.GOOGLE_GEMINI_MODEL || 'gemini-2.5-flash-preview-04-17';
    } else if (configuredProviderName === 'deepseek') {
      modelForProvider = process.env.DEEPSEEK_MODEL || 'deepseek-chat';
    }
    // If configuredProviderName is undefined or not one of the above, modelForProvider remains null.
    // setLLMProvider will use its own defaults if modelForProvider is null.

    // Attempt to set the initial LLM provider.
    // The 'true' flag for initialSetup suppresses console.error from setLLMProvider, but it still logs to logger.error.
    const providerSuccessfullySet = this.setLLMProvider(configuredProviderName, modelForProvider, true);

    if (!providerSuccessfullySet) {
      // setLLMProvider would have logged the specific error (e.g. API key missing or unsupported provider name)
      // Now, make the client fail hard, preventing it from starting in a broken state.
      throw new Error(
        `Failed to initialize the configured LLM provider "${config.llmProvider}". ` +
        `Please check your 'config.json', ensure the provider name is valid ('anthropic', 'openai', 'google', 'deepseek'), ` +
        `and that the required API key (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_GEMINI_API_KEY, DEEPSEEK_API_KEY) is set in your environment. ` +
        `Client cannot start.`
      );
    }
  }

  /**
   * Trims the conversation history if it exceeds the configured maximum length.
   * @private
   */
  _trimConversationHistory() {
    if (this.maxConversationHistoryLength && this.conversation.length > this.maxConversationHistoryLength) {
      const numToRemove = this.conversation.length - this.maxConversationHistoryLength;
      this.conversation.splice(0, numToRemove);
      logger.info(`Conversation history trimmed. Removed ${numToRemove} oldest messages to maintain a length of ${this.maxConversationHistoryLength}.`);
    }
  }

  /**
   * Get the dynamic prompt string for the CLI.
   * @returns {string} The prompt string.
   * @private
   */
  _getDynamicPrompt() {
    let providerName = this.config.llmProvider || 'none';
    let modelName = '';
    if (this.config.llmProvider === 'anthropic' && this.model) {
      modelName = this.model;
    } else if (this.config.llmProvider === 'openai' && this.openaiModel) {
      modelName = this.openaiModel;
    } else if (this.config.llmProvider === 'google' && this.geminiModel) {
      modelName = this.geminiModel;
    } else if (this.config.llmProvider === 'deepseek' && this.deepseekModel) {
      modelName = this.deepseekModel;
    }
    // Sanitize modelName to remove potentially problematic characters for prompt display if any
    // For now, direct usage is fine, but could add regex replace for non-alphanumeric if needed.
    return `${providerName}${modelName ? ` (${modelName.split('/').pop()})` : ''}> `; // Show only last part of model name if it's a path
  }

  /**
   * Helper to deeply truncate base64 strings within an object for logging.
   * @param {any} obj The object to process.
   * @returns {any} A new object with base64 strings truncated.
   * @private
   */
  _truncateBase64InObject(obj) {
    const B64_TRUNCATE_LENGTH = 50; // Length of preview for base64
    const B64_MIN_LENGTH_TO_TRUNCATE = 200; // Only truncate if longer than this

    // --- BEGIN ADDED DEBUG LOGGING ---
    if (typeof obj === 'string' && obj.startsWith('data:') && obj.includes(';base64,')) {
        logger.debug(`[_truncateBase64InObject] Processing data URL string. Length: ${obj.length}`);
    } else if (typeof obj === 'object' && obj !== null) {
        if (obj.base64Data && typeof obj.base64Data === 'string') {
            logger.debug(`[_truncateBase64InObject] Object has base64Data key with string length: ${obj.base64Data.length}`);
        }
        if (obj.data && typeof obj.data === 'string' && (obj.type === 'image' || (obj.source && obj.source.type === 'base64'))) {
             logger.debug(`[_truncateBase64InObject] Object has data key (image context) with string length: ${obj.data.length}`);
        }
    }
    // --- END ADDED DEBUG LOGGING ---

    if (typeof obj === 'string') {
      if (obj.startsWith('data:') && obj.includes(';base64,')) {
        const parts = obj.split(';base64,');
        if (parts.length === 2 && parts[1].length > B64_MIN_LENGTH_TO_TRUNCATE) {
          return `${parts[0]};base64,${parts[1].substring(0, B64_TRUNCATE_LENGTH)}... (truncated len:${parts[1].length})`;
        }
      }
      return obj;
    } else if (Array.isArray(obj)) {
      return obj.map(item => this._truncateBase64InObject(item));
    } else if (typeof obj === 'object' && obj !== null) {
      const newObj = {};
      for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          if (key === 'base64Data' || key === 'data') { // Specific keys often holding base64
            const value = obj[key];
            if (typeof value === 'string' && value.length > B64_MIN_LENGTH_TO_TRUNCATE) {
                // Simpler check for non-data-URL raw base64 strings
                const isLikelyBase64 = /^[A-Za-z0-9+/=]+$/.test(value.substring(0,100)); // Check prefix
                if (isLikelyBase64 && !value.startsWith('data:')) { // Ensure it's not a data URL already handled by string path
                     newObj[key] = `${value.substring(0, B64_TRUNCATE_LENGTH)}... (raw_b64 truncated len:${value.length})`;
                     // Safe substring for logging preview
                     const preview = newObj[key].substring(0, Math.min(70, newObj[key].length));
                     logger.debug(`[_truncateBase64InObject] Truncated raw base64 string for key '${key}'. Preview: ${preview}`);
                } else {
                    newObj[key] = this._truncateBase64InObject(value); // recurse if not matching common keys for raw b64 or if it is a data URL
                }
            } else {
                 newObj[key] = this._truncateBase64InObject(value);
            }
          } else {
            newObj[key] = this._truncateBase64InObject(obj[key]);
          }
        }
      }
      return newObj;
    }
    return obj;
  }

  /**
   * Sets or switches the LLM provider and model.
   * @param {string} newProvider The new provider ('anthropic', 'openai', or 'google').
   * @param {string} [newModelName] The specific model name for the provider.
   * @param {boolean} [initialSetup=false] Flag to suppress console output during initial constructor setup.
   * @returns {boolean} True if successful, false otherwise.
   */
  setLLMProvider(newProvider, newModelName, initialSetup = false) {
    newProvider = newProvider?.toLowerCase();
    logger.info(`Attempting to set LLM provider to: ${newProvider}` + (newModelName ? ` with model: ${newModelName}` : ' (using default model)'));

    // Clear old provider's SDK specific instances if any were on MCPClient directly
    this.anthropic = null; 
    this.openai = null;
    this.currentLlmProviderInstance = null; // Clear current provider instance
    
    // Clear model names for display, will be reset by successful provider setup
    // this.model is still used for display for Anthropic, this.openaiModel for OpenAI, etc.
    // The provider instance will hold the actual model name it's configured with.
    this.model = null;        
    this.openaiModel = null; 
    this.geminiModel = null;   
    this.deepseekModel = null;

    if (newProvider === 'anthropic') {
      const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
      if (!anthropicApiKey) {
        const errorMsg = 'ANTHROPIC_API_KEY environment variable is required for Anthropic provider.';
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
      try {
        const determinedModelName = newModelName || process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-latest';
        this.currentLlmProviderInstance = new AnthropicProvider(anthropicApiKey, determinedModelName, this.systemMessage, this.getAllTools(), logger, this.conversation);
        this.model = determinedModelName; // For display
        this.config.llmProvider = 'anthropic';
        if (!initialSetup) console.log(`Switched to Anthropic provider with model: ${this.model}`);
        logger.info(`Successfully set LLM provider to Anthropic with model ${this.model}`);
      } catch (error) {
        const errorMsg = `Failed to initialize Anthropic provider: ${error.message}`;
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
    } else if (newProvider === 'openai') {
      const openaiApiKey = process.env.OPENAI_API_KEY;
      if (!openaiApiKey) {
        const errorMsg = 'OPENAI_API_KEY environment variable is required for OpenAI provider.';
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
      try {
        const determinedModelName = newModelName || process.env.OPENAI_MODEL || 'gpt-4o';
        this.currentLlmProviderInstance = new OpenAIProvider(openaiApiKey, determinedModelName, this.systemMessage, this.getAllTools(), logger, this.conversation);
        this.openaiModel = determinedModelName; // For display
        this.config.llmProvider = 'openai';
        if (!initialSetup) console.log(`Switched to OpenAI provider with model: ${this.openaiModel}`);
        logger.info(`Successfully set LLM provider to OpenAI with model ${this.openaiModel}`);
      } catch (error) {
        const errorMsg = `Failed to initialize OpenAI provider: ${error.message}`;
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
    } else if (newProvider === 'google') {
      const geminiApiKey = process.env.GOOGLE_GEMINI_API_KEY;
      if (!geminiApiKey) {
        const errorMsg = 'GOOGLE_GEMINI_API_KEY environment variable is required for Google provider.';
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
      try {
        const determinedModelName = newModelName || process.env.GOOGLE_GEMINI_MODEL || 'gemini-2.5-flash-preview-04-17';
        // Instantiate GoogleProvider. It handles its own SDK client and chat session init.
        // It also needs the current conversation history to initialize the chat session correctly.
        // The GoogleProvider's initialize method currently starts with an empty history.
        // We need to adjust GoogleProvider or how it's called here.
        // For now, this will create a GoogleProvider, and its internal `initialize` will run.
        // If MCPClient.conversation has items, GoogleProvider will need to be enhanced to accept this.
        // Let's assume for this step that GoogleProvider's constructor/initialize will be updated to take `this.conversation`
        // OR MCPClient will call a method like `provider.startChatWithHistory(this.conversation)` after instantiation.
        // For now: just instantiate. The `initialize` in GoogleProvider uses an empty history for startChat.
        // THIS WILL BE A PROBLEM FOR CONTEXT RETENTION WHEN SWITCHING TO GOOGLE or /setsystem for google.
        // We will address this by modifying GoogleProvider to accept initial history, or MCPClient passes it.
        this.currentLlmProviderInstance = new GoogleProvider(geminiApiKey, determinedModelName, this.systemMessage, this.getAllTools(), logger, this.conversation);
        // The GoogleProvider constructor now calls its own `initialize`, which sets up chatSession.

        this.geminiModel = determinedModelName; // For display
        this.config.llmProvider = 'google';
        if (!initialSetup) console.log(`Switched to Google provider with model: ${this.geminiModel}`);
        logger.info(`Successfully set LLM provider to Google with model ${this.geminiModel}`);
      } catch (error) {
        const errorMsg = `Failed to initialize Google provider: ${error.message}`;
        logger.error(errorMsg, error.stack);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        this.currentLlmProviderInstance = null; // Ensure it's null on failure
        return false;
      }
    } else if (newProvider === 'deepseek') {
      const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
      if (!deepseekApiKey) {
        const errorMsg = 'DEEPSEEK_API_KEY environment variable is required for DeepSeek provider.';
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
      try {
        const determinedModelName = newModelName || process.env.DEEPSEEK_MODEL || 'deepseek-chat';
        this.currentLlmProviderInstance = new DeepSeekProvider(deepseekApiKey, determinedModelName, this.systemMessage, this.getAllTools(), logger, this.conversation);
        this.deepseekModel = determinedModelName; // For display
        this.config.llmProvider = 'deepseek';
        if (!initialSetup) console.log(`Switched to DeepSeek provider with model: ${this.deepseekModel}`);
        logger.info(`Successfully set LLM provider to DeepSeek with model ${this.deepseekModel}`);
      } catch (error) {
        const errorMsg = `Failed to initialize DeepSeek provider: ${error.message}`;
        logger.error(errorMsg);
        if (!initialSetup) console.error(`Error: ${errorMsg}`);
        return false;
      }
    } else {
      const errorMsg = `Unsupported LLM provider: ${newProvider}. Supported: 'anthropic', 'openai', 'google', 'deepseek'.`;
      logger.error(errorMsg);
      if (!initialSetup && newProvider) console.error(`Error: ${errorMsg}`);
      else if (!initialSetup && !newProvider) console.info ('No LLM provider specified in command.');
      return false;
    }

    if (this.rl && !initialSetup) this.rl.setPrompt(this._getDynamicPrompt());
    return true;
  }

  /**
   * Create appropriate MCP SDK client for a server configuration
   * 
   * @param {Object} serverConfig Server configuration
   * @returns {Object} Object containing client and transport instances
   */
  createClient(serverConfig) {
    // Create a new MCP SDK Client
    const client = new Client({
      name: `mcp-sdk-${serverConfig.name}`,
      version: "1.0.0"
    });
    
    let transport;
    const transportType = serverConfig.transport?.type?.toLowerCase();

    if (transportType === 'streamable-http') {
      if (!serverConfig.baseUrl) {
        throw new Error(`Configuration for server '${serverConfig.name}' is type 'streamable-http' but is missing 'baseUrl'.`);
      }
      logger.info(`Creating StreamableHTTPClientTransport for server ${serverConfig.name} with baseUrl: ${serverConfig.baseUrl}`);
      try {
        const baseUrlObject = new URL(serverConfig.baseUrl);
        transport = new StreamableHTTPClientTransport({
          baseUrl: baseUrlObject,
          headers: serverConfig.transport?.headers
        });
      } catch (e) {
        throw new Error(`Invalid baseUrl '${serverConfig.baseUrl}' for server '${serverConfig.name}': ${e.message}`);
      }
    } else if (!transportType || transportType === 'stdio') {
      // Default to stdio or if explicitly stdio
      if (!serverConfig.command) {
        throw new Error(`Configuration for server '${serverConfig.name}' is type 'stdio' (or default) but is missing 'command'.`);
      }
      logger.info(`Creating StdioClientTransport for server ${serverConfig.name}`);
      
      // Process environment variables for stdio transport
      const processedEnv = {};
      for (const [key, value] of Object.entries(serverConfig.env || {})) {
        if (typeof value === 'string' && value.startsWith('${') && value.endsWith('}')) {
          const envVar = value.slice(2, -1);
          if (!process.env[envVar]) {
            if (envVar !== 'LOG_LEVEL') { // LOG_LEVEL is handled by defaulting to main logger.level
              throw new Error(`Environment variable ${envVar} (referenced in config for server ${serverConfig.name}) is not set in the main environment.`);
            }
            // If LOG_LEVEL is not set in parent, it will be implicitly handled by finalChildEnv defaulting to logger.level
          } else {
            processedEnv[key] = process.env[envVar];
          }
        } else {
          processedEnv[key] = value;
        }
      }
      
      const finalChildEnv = {
        ...process.env,
        ...processedEnv,
        // Conditionally set child's LOG_LEVEL
        // Pass 'debug' to child only if main logger is 'debug' AND tool call verbosity is also 'debug'
        // Otherwise, default child to 'info' to suppress verbose server debug logs like full base64.
        LOG_LEVEL: (this.logToolCallVerbosity === 'debug' && logger.level.toLowerCase() === 'debug') ? 'debug' : 'info'
      };

      transport = new StdioClientTransport({
        command: serverConfig.command,
        args: serverConfig.args || [],
        env: finalChildEnv,
      });
    } else {
      throw new Error(`Unsupported transport type '${serverConfig.transport?.type}' for server ${serverConfig.name}. Supported types: 'stdio', 'streamable-http'.`);
    }
    
    return { client, transport };
  }

  /**
   * Start the client
   */
  async start() {
    try {
      // Start all MCP servers and connect clients
      for (const [serverName, serverDetails] of Object.entries(this.config.mcpServers)) {
        const serverConfig = {
          name: serverName,
          ...serverDetails
        };
        logger.info(`Starting server ${serverConfig.name}...`);
        
        const { client, transport } = this.createClient(serverConfig);
        
        logger.info(`Connecting to server ${serverConfig.name}` + 
                    (serverConfig.transport?.type?.toLowerCase() === 'streamable-http' ? 
                     ` via Streamable HTTP at ${serverConfig.baseUrl}` : 
                     ` via stdio (command: ${serverConfig.command} ${(serverConfig.args || []).join(' ')})`)
                   + '...');
        try {
          await client.connect(transport);
          logger.info(`Connected to server ${serverConfig.name} ✓`);
          
          // Store client and transport using serverConfig.name (which is the key from the mcpServers object)
          this.mcpClients.set(serverConfig.name, client);
          this.transports.set(serverConfig.name, transport);
        } catch (connectionError) {
          logger.error(`Failed to connect to server ${serverConfig.name}: ${connectionError.message}`);
          console.error(`ERROR: Failed to connect to server ${serverConfig.name}: ${connectionError.message}`);
        }
      }
      
      if (this.mcpClients.size === 0) {
        throw new Error('No MCP servers could be connected. Check the configuration and try again.');
      }
      
      // Load available tools from all servers
      await this.loadTools();
      
      const allTools = this.getAllTools();
      if (allTools.length === 0) {
        logger.warn('No tools were found on any connected servers.');
        console.warn('WARNING: No tools were found on any connected servers.');
      }
      
      // Start the CLI interface
      this.startCLI();
      
      return true;
    } catch (error) {
      logger.error(`Failed to start client: ${error.message}`);
      console.error(`\nERROR: ${error.message}`);
      return false;
    }
  }

  /**
   * Load available tools from all MCP servers
   */
  async loadTools() {
    logger.info('Loading tools from MCP servers...');
    
    try {
      for (const [serverName, client] of this.mcpClients) {
        try {
          // List tools using MCP SDK
          const toolsList = await client.listTools();
          
          if (toolsList && toolsList.tools) {
            // Add server name to each tool for identification
            const toolsWithServer = toolsList.tools.map(tool => ({
              ...tool,
              serverName
            }));
            
            this.tools.set(serverName, toolsWithServer);
            logger.info(`Loaded ${toolsWithServer.length} tools from server ${serverName}`);
            
            // Log the tools
            toolsWithServer.forEach((tool, index) => {
              logger.info(`Tool ${index + 1} from ${serverName}: ${tool.name} - ${tool.description || 'No description'}`);
            });
          } else {
            logger.warn(`No tools found on server ${serverName}`);
            this.tools.set(serverName, []);
          }
        } catch (error) {
          logger.error(`Error loading tools from ${serverName}: ${error.message}`);
          this.tools.set(serverName, []);
        }
      }

      // After all tools are loaded and this.tools (Map) is populated:
      if (this.currentLlmProviderInstance && typeof this.currentLlmProviderInstance.reconfigure === 'function') {
        logger.info(`[MCPClient.loadTools] Tools loaded/updated. Reconfiguring current LLM provider (${this.config.llmProvider || 'unknown'}) with fresh toolset.`);
        // Pass the current system message and the newly available tools from getAllTools()
        await this.currentLlmProviderInstance.reconfigure(this.systemMessage, this.getAllTools());
      } else if (this.currentLlmProviderInstance) {
        logger.warn(`[MCPClient.loadTools] Current LLM provider instance does not have a reconfigure method. Tools may be stale.`);
      } else {
        logger.warn('[MCPClient.loadTools] No current LLM provider instance to reconfigure with tools. This might be an issue if a provider was expected.');
      }

    } catch (error) {
      logger.error(`Error loading tools: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all available tools across all servers
   * 
   * @returns {Array} Combined array of all tools
   */
  getAllTools() {
    const allTools = [];
    for (const tools of this.tools.values()) {
      allTools.push(...tools);
    }
    return allTools;
  }

  /**
   * Start the CLI interface
   */
  startCLI() {
    const allTools = this.getAllTools();
    
    // Display available tools
    console.log(`\nAvailable tools (${allTools.length}):`);
    allTools.forEach((tool, index) => {
      console.log(`${index + 1}. [${tool.serverName}] ${tool.name} - ${tool.description || 'No description'}`);
    });
    
    // Create readline interface
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
      // Prompt will be set dynamically
    });
    
    this.rl.setPrompt(this._getDynamicPrompt()); // Set initial prompt

    console.log('\nMCP Client (SDK Version)');
    // The dynamic prompt now shows the provider and model, so this specific line might be redundant or rephrased.
    // For now, let's keep it for clarity on startup.
    console.log(`Currently using LLM provider: ${this.config.llmProvider}` + 
                (this.config.llmProvider === 'anthropic' ? ` (${this.model})` : 
                 this.config.llmProvider === 'openai' ? ` (${this.openaiModel})` : 
                 this.config.llmProvider === 'google' ? ` (${this.geminiModel})` : 
                 this.config.llmProvider === 'deepseek' ? ` (${this.deepseekModel})` : 
                 ' (None)'));
    console.log('Type your questions or instructions. Type /help for available commands.');
    
    this.rl.prompt();
    
    // Handle user input
    this.rl.on('line', async (line) => {
      const input = line.trim();
      
      if (!input) {
        this.rl.prompt();
        return;
      }

      if (input.toLowerCase() === '/help') {
        console.log('\nAvailable commands:');
        console.log('  /exit, /quit         - Exit the client.');
        console.log('  /clear               - Clear the conversation history.');
        console.log('  /setsystem <message> - Set a new system message for the LLM.');
        console.log('  /setprovider <name> [model] - Switch LLM provider (e.g., /setprovider google gemini-2.5-flash-preview-04-17).');
        console.log('                         Supported providers: anthropic, openai, google, deepseek.');
        console.log('  /help                - Show this help message.');
        this.rl.prompt();
        return;
      }
      
      if (input === '/exit' || input === '/quit') {
        console.log('Exiting...');
        await this.stop();
        process.exit(0);
      }
      
      if (input === '/clear') {
        console.log('Clearing conversation history...');
        this.conversation = [];
        this.rl.prompt();
        return;
      }

      if (input.startsWith('/setsystem ')) {
        const newSystemMessage = input.substring('/setsystem '.length).trim();
        if (newSystemMessage) {
          this.systemMessage = newSystemMessage;
          logger.info(`System message updated to: "${this.systemMessage.substring(0, 100)}${this.systemMessage.length > 100 ? '...' : ''}"`);
          console.log(`System message updated. It will be applied on the next interaction.`);
          // For Google, the system message is part of the model initialization.
          // Re-initialize to apply the new system message immediately.
          if (this.config.llmProvider === 'google') {
            console.log("Re-initializing Google provider to apply new system message...");
            // Use current model for re-initialization, suppress console output with initialSetup=true
            const currentGoogleModel = this.geminiModel || process.env.GOOGLE_GEMINI_MODEL || 'gemini-2.5-flash-preview-04-17';
            if (this.setLLMProvider('google', currentGoogleModel, true)) {
                console.log("Google provider re-initialized with new system message.");
            } else {
                console.error("Failed to re-initialize Google provider with new system message. Please try /setprovider manually.");
            }
          }
        } else {
          console.log('Current system message:');
          console.log(this.systemMessage);
          console.log('Usage: /setsystem <your new system message>');
        }
        this.rl.prompt();
        return;
      }
      
      if (input.startsWith('/setprovider')) {
        const parts = input.split(' ').slice(1);
        const provider = parts[0]?.toLowerCase();
        const modelName = parts[1];
        if (provider) {
          this.setLLMProvider(provider, modelName);
        } else {
          console.error('Usage: /setprovider <provider_name> [model_name]');
          console.error('Supported providers: anthropic, openai, google, deepseek');
        }
        this.rl.prompt();
        return;
      }
      
      // Process user query with the LLM
      try {
        console.log('Processing your request...');
        await this.processQuery(input);
      } catch (error) {
        logger.error(`Error processing query: ${error.message}`);
        console.error(`Error: ${error.message}`);
      }
      
      this.rl.prompt();
    });
    
    // Handle Ctrl+C
    this.rl.on('SIGINT', async () => {
      console.log('\nExiting...');
      await this.stop();
      process.exit(0);
    });
  }

  /**
   * Process a user query with the LLM
   * 
   * @param {string} query User query
   */
  async processQuery(query) {
    this.conversation.push({ role: 'user', content: query });
    const allTools = this.getAllTools();
    logger.info(`Sending query to LLM provider: ${this.config.llmProvider} with ${allTools.length} tools`);

    try {
      let response;
      if (this.config.llmProvider === 'anthropic') {
        if (!this.currentLlmProviderInstance) {
            logger.warn("Anthropic provider instance not available in processQuery. Attempting re-initialization.");
            const currentAnthropicModel = this.model || process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-latest';
            if (!this.setLLMProvider('anthropic', currentAnthropicModel, true)) {
                throw new Error("Anthropic provider instance failed to re-initialize. Please use /setprovider anthropic [model_name].");
            }
            logger.info("Successfully re-initialized Anthropic provider instance in processQuery.");
        }
        // MCPClient is responsible for trimming history before passing to provider's formatConversation
        this._trimConversationHistory(); 
        const messagesForLLM = this.currentLlmProviderInstance.formatConversation(this.conversation);
        // sendMessage on the provider will handle constructing the full request with tools, system message etc.
        response = await this.currentLlmProviderInstance.sendMessage(messagesForLLM);
      } else if (this.config.llmProvider === 'openai') {
        if (!this.currentLlmProviderInstance) {
            logger.warn("OpenAI provider instance not available in processQuery. Attempting re-initialization.");
            const currentOpenAIModel = this.openaiModel || process.env.OPENAI_MODEL || 'gpt-4o';
            if (!this.setLLMProvider('openai', currentOpenAIModel, true)) {
                throw new Error("OpenAI provider instance failed to re-initialize. Please use /setprovider openai [model_name].");
            }
            logger.info("Successfully re-initialized OpenAI provider instance in processQuery.");
        }
        this._trimConversationHistory();
        const messagesForLLM = this.currentLlmProviderInstance.formatConversation(this.conversation);
        response = await this.currentLlmProviderInstance.sendMessage(messagesForLLM);
      } else if (this.config.llmProvider === 'google') {
        if (!this.currentLlmProviderInstance) {
             logger.warn("Google provider instance not available in processQuery. This should have been handled by setLLMProvider or a previous error.");
             // Attempt to re-initialize. Use current model, suppress console output.
             const currentGoogleModelForRetry = this.geminiModel || process.env.GOOGLE_GEMINI_MODEL || 'gemini-2.5-flash-preview-04-17';
             if(!this.setLLMProvider('google', currentGoogleModelForRetry, true)) { 
                 throw new Error("Google provider instance is not initialized and auto-reinitialization failed. Please use /setprovider google [model_name] again.");
            }
            logger.info("Successfully re-initialized Google provider instance in processQuery.");
        }
        
        // The GoogleProvider instance now manages its own conversation history and formatting.
        // We just need to send the latest user query.
        // The provider's sendMessage method will take care of adding it to its internal history.
        const latestUserQuery = this.conversation.findLast(m => m.role === 'user');
        if (!latestUserQuery || (typeof latestUserQuery.content !== 'string' && !Array.isArray(latestUserQuery.content)) ) {
            throw new Error("Could not find the latest user query in conversation history or content is not string/array for Google/DeepSeek.");
        }
        logger.debug(`[${this.config.llmProvider} Process Query] Sending latest user query to Provider: "${typeof latestUserQuery.content === 'string' ? latestUserQuery.content : JSON.stringify(latestUserQuery.content)}"`);
        response = await this.currentLlmProviderInstance.sendMessage(latestUserQuery.content);
      } else if (this.config.llmProvider === 'deepseek') {
        if (!this.currentLlmProviderInstance) {
            logger.warn("DeepSeek provider instance not available in processQuery. Attempting re-initialization.");
            const currentDeepSeekModel = this.deepseekModel || process.env.DEEPSEEK_MODEL || 'deepseek-chat';
            if (!this.setLLMProvider('deepseek', currentDeepSeekModel, true)) {
                throw new Error("DeepSeek provider instance failed to re-initialize. Please use /setprovider deepseek [model_name].");
            }
            logger.info("Successfully re-initialized DeepSeek provider instance in processQuery.");
        }
        this._trimConversationHistory();
        const messagesForLLM = this.currentLlmProviderInstance.formatConversation(this.conversation);
        response = await this.currentLlmProviderInstance.sendMessage(messagesForLLM);
      }
      await this.handleLLMResponse(response);
    } catch (error) {
      logger.error(`LLM API error: ${error.message}`); console.error(`API Error: ${error.message}`);
      this.conversation.push({ role: 'assistant', content: `An API error occurred: ${error.message}` });
    }
  }

  /**
   * Extract base64 image data and mime type from an MCP tool result's content.
   * @param {Object|string} toolResultContent The raw content from client.callTool().
   * @returns {{base64Data: string, mimeType: string}|null} Extracted image info or null.
   * @private
   */
  _extractImageFromMcpToolResult(toolResultContent) {
    if (!toolResultContent || typeof toolResultContent !== 'object') {
      return null;
    }

    // Scenario 1: Direct image object in toolResultContent itself (e.g., if toolResultContent IS the image object)
    // Example: { base64Data: "...", mimeType: "..." } (less common for complex tools)
    // Or more likely: toolResultContent.image = { base64Data: "...", mimeType: "..." }
    const imageObj = toolResultContent.image || toolResultContent; // Check direct or nested under .image

    if (imageObj && typeof imageObj.base64Data === 'string' && typeof imageObj.mimeType === 'string') {
      let base64Data = imageObj.base64Data;
      // Ensure raw base64 (remove data URL prefix if present)
      if (base64Data.startsWith('data:')) {
        base64Data = base64Data.substring(base64Data.indexOf(',') + 1);
      }
      return { base64Data, mimeType: imageObj.mimeType };
    }

    // Scenario 2: MCP content array within toolResultContent
    // Example: { content: [ { type: 'text', ... }, { type: 'image', data: '...', mimeType: '...' } ] }
    if (Array.isArray(toolResultContent.content)) {
      const imagePart = toolResultContent.content.find(
        part => part.type === 'image' && typeof part.data === 'string' && typeof part.mimeType === 'string'
      );
      if (imagePart) {
        let base64Data = imagePart.data;
        if (base64Data.startsWith('data:')) {
          base64Data = base64Data.substring(base64Data.indexOf(',') + 1);
        }
        return { base64Data, mimeType: imagePart.mimeType };
      }
    }
    return null;
  }

  /**
   * Handle an LLM response, including any tool calls
   * 
   * @param {Object} response LLM API response
   */
  async handleLLMResponse(response) {
    logger.debug(`Raw LLM response: ${JSON.stringify(response)}`);
    let textContent = '';
    let toolCalls = null;

    if (this.config.llmProvider === 'anthropic') {
      if (!this.currentLlmProviderInstance) {
        throw new Error("Anthropic provider instance not found in handleLLMResponse.");
      }
      textContent = this.currentLlmProviderInstance.extractTextContent(response);
      toolCalls = this.currentLlmProviderInstance.extractToolCalls(response);
    } else if (this.config.llmProvider === 'openai') {
      if (!this.currentLlmProviderInstance) {
        throw new Error("OpenAI provider instance not found in handleLLMResponse.");
      }
      textContent = this.currentLlmProviderInstance.extractTextContent(response);
      toolCalls = this.currentLlmProviderInstance.extractToolCalls(response);
    } else if (this.config.llmProvider === 'google') {
      if (!this.currentLlmProviderInstance) {
        throw new Error("Google provider instance not found in handleLLMResponse. This indicates a critical setup error.");
      }
      textContent = this.currentLlmProviderInstance.extractTextContent(response);
      toolCalls = this.currentLlmProviderInstance.extractToolCalls(response);
    } else if (this.config.llmProvider === 'deepseek') {
      if (!this.currentLlmProviderInstance) {
        throw new Error("DeepSeek provider instance not found in handleLLMResponse.");
      }
      textContent = this.currentLlmProviderInstance.extractTextContent(response);
      toolCalls = this.currentLlmProviderInstance.extractToolCalls(response);
    }
    
    this.conversation.push({ role: 'assistant', content: textContent, tool_calls: toolCalls });
    if (textContent) console.log(textContent);

    if (toolCalls && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        let toolResultForAugmentation = null; // Variable to store the result for later augmentation
        try {
          if (this.logToolCallVerbosity === 'minimal' || this.logToolCallVerbosity === 'default' || this.logToolCallVerbosity === 'debug') {
            console.log(`\nExecuting tool: ${toolCall.name}...`);
          }
          
          let serverName = null;
          for (const [name, tools] of this.tools) { if (tools.some(t => t.name === toolCall.name)) { serverName = name; break; } }
          if (!serverName) throw new Error(`No server found for tool: ${toolCall.name}`);
          
          const result = await this.callTool(serverName, toolCall.name, toolCall.args);
          toolResultForAugmentation = result; // Store the result
          
          logger.info('Raw tool result from this.callTool in MCPClient:', JSON.stringify(result, null, 2));
          
          this.conversation.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            name: toolCall.name,      
            content: result 
          });

          // Display the result based on verbosity settings
          if (this.logToolCallVerbosity === 'default' || this.logToolCallVerbosity === 'debug') {
            console.log('Tool result:');

            // --- BEGIN ADDED DEBUG LOGGING ---
            if (typeof result === 'object' && result !== null && !this.logToolResultsBase64Full) {
              logger.debug('[MCPClient.handleLLMResponse] BEFORE truncation: Checking result object for base64 content.');
              if (result.image && typeof result.image.base64Data === 'string') {
                logger.debug(`[MCPClient.handleLLMResponse] BEFORE truncation: result.image.base64Data length: ${result.image.base64Data.length}`);
              }
              if (result.content && Array.isArray(result.content)) {
                result.content.forEach((part, index) => {
                  if (part.type === 'image' && typeof part.data === 'string') {
                    logger.debug(`[MCPClient.handleLLMResponse] BEFORE truncation: result.content[${index}] (image part) data length: ${part.data.length}`);
                  }
                });
              }
            }
            // --- END ADDED DEBUG LOGGING ---

            const resultToLog = this.logToolResultsBase64Full ? result : this._truncateBase64InObject(result);

            // --- BEGIN ADDED DEBUG LOGGING ---
            if (typeof resultToLog === 'object' && resultToLog !== null && !this.logToolResultsBase64Full) {
              logger.debug('[MCPClient.handleLLMResponse] AFTER truncation: Checking resultToLog object for base64 content.');
              if (resultToLog.image && typeof resultToLog.image.base64Data === 'string') {
                logger.debug(`[MCPClient.handleLLMResponse] AFTER truncation: resultToLog.image.base64Data (string check): ${resultToLog.image.base64Data.substring(0, Math.min(70, resultToLog.image.base64Data.length))}...`);
              }
               if (resultToLog.base64Data && typeof resultToLog.base64Data === 'string') { // Check if root object has base64Data
                logger.debug(`[MCPClient.handleLLMResponse] AFTER truncation: resultToLog.base64Data (string check): ${resultToLog.base64Data.substring(0, Math.min(70, resultToLog.base64Data.length))}...`);
              }
              if (resultToLog.content && Array.isArray(resultToLog.content)) {
                resultToLog.content.forEach((part, index) => {
                  if (part.type === 'image' && typeof part.data === 'string') {
                    logger.debug(`[MCPClient.handleLLMResponse] AFTER truncation: resultToLog.content[${index}] (image part) data: ${part.data.substring(0, Math.min(70, part.data.length))}...`);
                  } else if (part.type === 'image_url' && part.image_url && typeof part.image_url.url === 'string') {
                     logger.debug(`[MCPClient.handleLLMResponse] AFTER truncation: resultToLog.content[${index}] (image_url part) url: ${part.image_url.url.substring(0, Math.min(100, part.image_url.url.length))}...`);
                  }
                });
              }
            }
             logger.debug(`[MCPClient.handleLLMResponse] Final resultToLog type: ${typeof resultToLog}`);
            // --- END ADDED DEBUG LOGGING ---
            
            if (this.logToolCallVerbosity === 'debug') {
                // For debug mode, always log the potentially truncated full object
                console.log(JSON.stringify(resultToLog, null, 2));
            } else { // 'default' verbosity
                if (typeof resultToLog === 'string') console.log(resultToLog);
                else if (resultToLog && resultToLog.content && Array.isArray(resultToLog.content)) {
                  const textPart = resultToLog.content.find(p => p.type === 'text');
                  if (textPart) console.log(textPart.text); // Prints only text for multi-part
                  else console.log(JSON.stringify(resultToLog.content, null, 2)); // Prints content array if no text part
                } 
                else console.log(JSON.stringify(resultToLog, null, 2)); // Prints whole object if not string/multi-part
            }
          } else if (this.logToolCallVerbosity === 'minimal') {
            // Optionally, log a very brief confirmation like "Tool execution completed." or nothing more.
            // For now, minimal means only the "Executing tool..." message.
          }
          
          await this.continueConversation(); // Let the current provider continue with the tool result

        } catch (error) {
          logger.error(`Error executing tool ${toolCall.name}: ${error.message}`);
          console.error(`Tool error: ${error.message}`);
          this.conversation.push({ role: 'tool', tool_call_id: toolCall.id, name: toolCall.name, content: JSON.stringify({ error: error.message }) });
          await this.continueConversation(); // Still attempt to continue so LLM can respond to the error
        }

        // AFTER continueConversation, add the synthetic user message if an image was in the tool result
        if (toolResultForAugmentation) {
          const extractedImage = this._extractImageFromMcpToolResult(toolResultForAugmentation);
          // Only add synthetic image_mcp message for providers that DON'T have robust internal handling
          // for images within tool results or dedicated multi-step flows (like OpenAI/Google).
          // Anthropic can handle images directly in its tool_result content.
          const providersToExcludeImageMcpAugmentation = ['openai', 'anthropic', 'google'];
          if (!providersToExcludeImageMcpAugmentation.includes(this.config.llmProvider) &&
              extractedImage && extractedImage.base64Data && extractedImage.mimeType) {
            const syntheticUserMessage = {
              role: 'user',
              content: [
                {
                  type: 'image_mcp', // Standardized internal type for cross-provider compatibility
                  source: {
                    media_type: extractedImage.mimeType,
                    data: extractedImage.base64Data // Raw base64
                  }
                },
                {
                  type: 'text',
                  text: `[Image from tool '${toolCall.name}'. This is the image content. You can now analyze this image.]`
                }
              ]
            };
            this.conversation.push(syntheticUserMessage);
            logger.info(`[MCPClient] (Post-Continuation) Augmented conversation with synthetic user image message (type: image_mcp) from tool '${toolCall.name}'. Current provider: ${this.config.llmProvider}.`);
          }
        }
      }
    }
  }

  /**
   * Continue the conversation with the LLM after a tool call
   */
  async continueConversation() {
    try {
      let response;
      const allTools = this.getAllTools();
      if (this.config.llmProvider === 'anthropic') {
        if (!this.currentLlmProviderInstance) {
            logger.warn("Anthropic provider instance not available in continueConversation. Attempting re-initialization.");
            const currentAnthropicModel = this.model || process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-latest';
            if (!this.setLLMProvider('anthropic', currentAnthropicModel, true)) {
                throw new Error("Anthropic provider instance failed to re-initialize. Please use /setprovider anthropic [model_name].");
            }
            logger.info("Successfully re-initialized Anthropic provider instance in continueConversation.");
        }
        this._trimConversationHistory();
        const messagesForLLM = this.currentLlmProviderInstance.formatConversation(this.conversation);
        response = await this.currentLlmProviderInstance.sendMessage(messagesForLLM);
        
        // Process and add LLM's response (the acknowledgement to the tool summary) to history.
        // This ensures the acknowledgement is ALWAYS added, removing the 'skipInitialHandleLLMResponse' logic.
        // await this.handleLLMResponse(response); // REMOVED - this was causing double processing for Anthropic

      } else if (this.config.llmProvider === 'openai') {
        // Check the last message in the conversation. If it's a tool result and contains an image,
        // then this `continueConversation` call is the one that should trigger the image analysis.
        const lastMessage = this.conversation.length > 0 ? this.conversation[this.conversation.length - 1] : null;

        if (lastMessage && lastMessage.role === 'tool') {
          const { base64ImageData, mimeType } = this.currentLlmProviderInstance.extractImageFromToolResult(lastMessage.content);

          if (base64ImageData && mimeType) {
            logger.info('[MCPClient.continueConversation] OpenAI: Image detected in last tool result. Proceeding with image analysis.');
            
            let originalUserQuery = "Describe the image."; // Default prompt
            let assistantMessageThatCalledTheTool = null;
            let indexOfAssistantMessage = -1;

            // Find the assistant message that made the tool call for this image
            for (let i = this.conversation.length - 2; i >= 0; i--) { // Start before the tool result message
                const msg = this.conversation[i];
                if (msg.role === 'assistant' && msg.tool_calls) {
                    if (msg.tool_calls.some(tc => tc.id === lastMessage.tool_call_id)) {
                        assistantMessageThatCalledTheTool = msg;
                        indexOfAssistantMessage = i;
                        logger.debug(`[MCPClient.continueConversation] OpenAI: Found assistant message (index ${i}) that called tool ${lastMessage.tool_call_id}`);
                        break;
                    }
                }
            }

            if (assistantMessageThatCalledTheTool && indexOfAssistantMessage > 0) {
                for (let i = indexOfAssistantMessage - 1; i >= 0; i--) {
                    if (this.conversation[i].role === 'user') {
                        const rawContent = this.conversation[i].content;
                        if (typeof rawContent === 'string') {
                            originalUserQuery = rawContent;
                        } else if (Array.isArray(rawContent) && rawContent.length > 0 && typeof rawContent[0].text === 'string') {
                            originalUserQuery = rawContent[0].text;
                        }
                        logger.debug(`[MCPClient.continueConversation] OpenAI: Found original user query (index ${i}): "${originalUserQuery.substring(0,50)}..."`);
                        break; 
                    }
                }
            }

            const toolName = lastMessage.name || 'the tool';
            const imagePrompt = `The tool '${toolName}' returned an image. Based on your previous request: "${originalUserQuery.substring(0, 200)}${originalUserQuery.length > 200 ? '...':''}". Please analyze this image. You are capable of identifying elements and their pixel coordinates (x,y from top-left).`;
            
            const imageContentParts = this.currentLlmProviderInstance.prepareImageMessageContent(base64ImageData, mimeType, imagePrompt);

            this.conversation.push({
              role: 'user',
              content: imageContentParts,
              timestamp: new Date().toISOString()
            });
            this._trimConversationHistory();
            logger.debug('[MCPClient.continueConversation] OpenAI: Sending image and prompt to LLM for analysis.');

            const messagesForLLM_image_step = this.currentLlmProviderInstance.formatConversation(this.conversation);
            logger.debug('[MCPClient.continueConversation] OpenAI: Messages being sent for image analysis:', JSON.stringify(messagesForLLM_image_step, null, 2));
            const response_image_step = await this.currentLlmProviderInstance.sendMessage(messagesForLLM_image_step);
            await this.handleLLMResponse(response_image_step);
            return; // Image analysis path complete, exit continueConversation
          } else {
            logger.debug('[MCPClient.continueConversation] OpenAI: Last tool result did not contain an image. Proceeding with standard continuation.');
            // Fall through to the general OpenAI message sending if no image in the tool result
          }
        } else {
           logger.debug('[MCPClient.continueConversation] OpenAI: Last message not a tool result, or conversation empty. Proceeding with standard continuation.');
           // Fall through to the general OpenAI message sending
        }
        // Standard continuation for OpenAI if no image was detected in the *last tool message*
        // This will be reached if the last message wasn't a tool message, or if the tool message had no image.
        this._trimConversationHistory();
        const messagesForLLM = this.currentLlmProviderInstance.formatConversation(this.conversation);
        response = await this.currentLlmProviderInstance.sendMessage(messagesForLLM);
        await this.handleLLMResponse(response); // Call handleLLMResponse for the general continuation
        return; // OpenAI path is now complete.
      } else if (this.config.llmProvider === 'google') {
        if (!this.currentLlmProviderInstance) {
             logger.warn("Google provider instance not available in continueConversation. Attempting to re-initialize.");
             const currentGoogleModelForRetry = this.geminiModel || process.env.GOOGLE_GEMINI_MODEL || 'gemini-2.5-flash-preview-04-17';
             if(!this.setLLMProvider('google', currentGoogleModelForRetry, true)) { 
                 throw new Error("Google provider instance failed to initialize. Please use /setprovider google [model_name] again.");
            }
            logger.info("Successfully re-initialized Google provider instance in continueConversation.");
        }

        const lastMessage = this.conversation[this.conversation.length - 1];
        if (lastMessage?.role !== 'tool') {
            logger.error("Last message is not a tool response for Google continuation. This should not happen.", lastMessage);
            throw new Error("Programming error: Expected last message to be a tool response for Google continuation.");
        }
        
        const toolCallResult = lastMessage.content; // Raw result from client.callTool()
        const toolName = lastMessage.name;
        const toolCallId = lastMessage.tool_call_id; // Required by some provider interfaces if they need it.

        // The GoogleProvider instance will handle the two-step image process internally if needed.
        // Its continueConversation method should accept the tool result and manage sending
        // the FunctionResponse and potentially a follow-up image message.
        // It should then return the final LLM response after all steps.
        // The MCPClient's `this.conversation` is the source of truth for history passed to the provider.
        // The provider's `continueConversation` method will get the latest state.

        // The provider method needs:
        // 1. The full conversation history (or rely on its internal synchronized history).
        // 2. The specific tool call result that needs to be processed.
        
        // Let's assume `this.currentLlmProviderInstance.continueConversation` takes the tool call details
        // and internally uses its synchronized history.
        // It will return the Gemini SDK's response(s), and we'll call handleLLMResponse once with the final one.
        
        // The `GoogleProvider.continueConversation` is expected to:
        // 1. Convert `toolCallResult` to `FunctionResponsePart` (using its `convertToolResult` method).
        // 2. Send this `FunctionResponsePart` to Gemini.
        // 3. Handle the LLM response to this part (internally, or return it).
        // 4. If an image was part of `toolCallResult`, prepare and send the image part with a prompt.
        // 5. Handle the LLM response to the image part (internally, or return it).
        // 6. Return the *final* relevant LLM response object that MCPClient's `handleLLMResponse` can process.

        // The old Google `continueConversation` already calls `handleLLMResponse` for each step.
        // This is the key: the provider's methods should *not* call `this.handleLLMResponse`.
        // They should return the raw SDK response, and `MCPClient` orchestrates `handleLLMResponse`.

        // Revised flow for Google in MCPClient.continueConversation:
        const googleProvider = this.currentLlmProviderInstance;

        // Step 1: Send FunctionResponse
        const functionResponsePart = googleProvider.convertToolResult(toolCallResult, toolName); // Assumes method exists on provider
        logger.debug(`[Google ContinueConversation] Step 1: Sending FunctionResponse part for ${toolName} via provider:`, JSON.stringify(functionResponsePart, null, 2));
        
        // Provider's sendMessage needs to handle history correctly.
        // It should append the functionResponsePart to its *current* chat session history before sending.
        let llmResponseStep1 = await googleProvider.sendMessage([functionResponsePart], true); // `true` to indicate it's a continuation
        
        // Process the LLM's reaction to the tool's structured data output
        // This will add the assistant's response to `this.conversation`
        await this.handleLLMResponse(llmResponseStep1); 

        // Step 2: If the toolResult also contained an image, send it in a subsequent, separate message
        const imageDetails = googleProvider.extractImageFromToolResult(toolCallResult); // New helper on provider

        if (imageDetails.base64ImageData && imageDetails.mimeType) {
            logger.info(`[Google ContinueConversation] Step 2: Tool ${toolName} returned image (mimeType: ${imageDetails.mimeType}). Sending image via provider.`);
            
            const imageMessageParts = googleProvider.prepareImageMessageParts(
                imageDetails.base64ImageData, 
                imageDetails.mimeType, 
                toolName, 
                this.conversation // Pass conversation for context
            );
            
            logger.debug(`[Google ContinueConversation] Step 2: Sending image message parts via provider:`, JSON.stringify(imageMessageParts, null, 2));
            let llmResponseStep2 = await googleProvider.sendMessage(imageMessageParts, true); // continuation
            await this.handleLLMResponse(llmResponseStep2); 
        }
        return; // Google path handles its own calls to handleLLMResponse and returns early.
      }
      // The following call to handleLLMResponse is for Anthropic and DeepSeek (if it uses a similar pattern),
      // and potentially others if not handled by specific blocks above.
      // OpenAI and Google paths now have explicit returns or their own handleLLMResponse calls within their blocks.
      if (this.config.llmProvider !== 'google' && this.config.llmProvider !== 'openai') { 
        await this.handleLLMResponse(response);
      }

    } catch (error) {
      logger.error(`Error continuing conversation: ${error.message}`);
      console.error(`Error: ${error.message}`);
      this.conversation.push({
        role: 'assistant',
        content: `An API error occurred while continuing the conversation: ${error.message}`
      });
    }
  }

  /**
   * Call a tool on an MCP server
   * 
   * @param {string} serverName Server name
   * @param {string} toolName Tool name
   * @param {Object} args Tool arguments
   * @returns {Promise<Object>} Tool result
   */
  async callTool(serverName, toolName, args) {
    logger.info(`Calling tool ${toolName} on server ${serverName} with args: ${JSON.stringify(args)}`);
    
    const client = this.mcpClients.get(serverName);
    if (!client) {
      throw new Error(`Client for server ${serverName} not found`);
    }
    
    // Call the tool using the MCP SDK
    try {
      const result = await client.callTool({
        name: toolName,
        arguments: args
      });
      
      logger.debug(`[MCPClient.callTool] Raw result from SDK client.callTool for ${toolName}:`, JSON.stringify(result, null, 2));
      return result;
    } catch (error) {
      logger.error(`Error calling tool ${toolName}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Stop the client
   */
  async stop() {
    logger.info('Stopping client...');
    
    if (this.rl) {
      this.rl.close();
    }
    
    // Stop all server transports
    for (const [serverName, transport] of this.transports) {
      try {
        logger.info(`Stopping server ${serverName}...`);
        await transport.close();
      } catch (error) {
        logger.error(`Error stopping server ${serverName}: ${error.message}`);
      }
    }
    
    logger.info('Client stopped');
  }
}

/**
 * Main entry point
 */
async function main() {
  try {
    // Load configuration
    const configPath = path.join(process.cwd(), 'config.json');
    const configData = await fs.readFile(configPath, 'utf8');
    const config = JSON.parse(configData);
    
    // Create and start client
    const client = new MCPClient(config);
    await client.start();
  } catch (error) {
    console.error(`Initialization error: ${error.message}`);
    process.exit(1);
  }
}

// Run the main function
main().catch(error => {
  console.error(`Fatal error: ${error.message}`);
  process.exit(1);
}); 