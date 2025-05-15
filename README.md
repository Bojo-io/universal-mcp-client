# Universal MCP Client (SDK Version)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, command-line interface (CLI) client for interacting with Language Model (LLM) providers (Anthropic, OpenAI, Google Gemini, DeepSeek) and Model Context Protocol (MCP) servers. This client utilizes the official `@modelcontextprotocol/sdk` for robust communication with MCP-compliant tool servers.

It's designed to be easily configurable and extensible, serving as a testing tool for MCP server implementations or as a simple, multi-provider LLM interaction hub.

## Features

*   **Multi-LLM Provider Support:** Seamlessly switch between Anthropic (Claude), OpenAI (GPT models), Google (Gemini models), and DeepSeek (DeepSeek models).
*   **MCP SDK Integration:** Uses the official MCP SDK for standardized tool discovery and execution with MCP servers.
*   **Multi-modal LLM Interaction:** Supports image input for vision-capable models from Anthropic, OpenAI, and Google. Tools returning images (e.g., screenshots) can be analyzed by the LLM.
*   **Command-Line Interface:** Interactive CLI for sending queries and managing the client.
*   **Dynamic Provider Switching & Context Preservation:** Change LLM providers and models on-the-fly. Conversation history, including text and images (where supported by provider transitions), is preserved.
*   **Configurable System Message:** Customize the system message sent to the LLM via `config.json` or environment variables.
*   **Conversation History Limit:** Control the length of the conversation history to manage context and token usage.
*   **Stdio Transport for MCP Servers:** Primarily uses stdio for connecting to local or npx-run MCP servers.
*   **Streamable HTTP Transport Support:** Connect to remote MCP servers that expose a Streamable HTTP interface (replaces the older HTTP+SSE).
*   **Environment Variable Support:** Configure API keys, default models, and client behavior through a `.env` file.
*   **Example Configurations:** Includes `.env.example` and `config.json.example` to get started quickly.

## Prerequisites

*   [Node.js](https://nodejs.org/) (v18.x or later recommended)
*   [npm](https://www.npmjs.com/) (usually comes with Node.js) or [npx](https://www.npmjs.com/package/npx)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd universal-mcp-client
    ```

2.  **Install Dependencies:**
    While this client is designed to be lightweight and mainly uses packages via `npx` for servers, ensure you have the necessary SDKs listed in `package.json` if you modify the core client. For basic usage, `npm install` is good practice to ensure all declared dependencies are available locally.
    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Copy the example environment file and fill in your API keys and any other desired settings:
    ```bash
    cp .env.example .env
    ```
    Now, edit `.env` with your actual API keys for Anthropic, OpenAI, Google Gemini, and DeepSeek. You can also set default models and other options here.

    ```env
    # .env (Example - fill with your actual values)
    ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
    OPENAI_API_KEY="sk-your-openai-key"
    GOOGLE_GEMINI_API_KEY="your-google-gemini-key"
    DEEPSEEK_API_KEY="your-deepseek-api-key"
    # ... other variables as per .env.example
    ```

4.  **Configure Client and MCP Servers:**
    Copy the example configuration file:
    ```bash
    cp config.json.example config.json
    ```
    Edit `config.json` to:
    *   Set your initial `llmProvider`.
    *   Optionally set a default `systemMessage` and `maxConversationHistoryLength`.
    *   Define the MCP servers you want to connect to. For each server, specify its `name`, `command` to start it (e.g., using `npx` for published MCP servers or `node` for local scripts), and any necessary `env` variables (which can reference those in your `.env` file).

    **Example `config.json` snippet for an MCP server:**
    ```json
    {
      // ... other global settings ...
      "mcpServers": {
        "my_tool_server": {
          "command": "npx",
          "args": ["-y", "@someorg/some-mcp-server"],
          "transport": { "type": "stdio" },
          "env": {
            "SERVER_SPECIFIC_API_KEY": "MY_SERVER_API_KEY_FROM_DOTENV"
          }
        }
      }
    }
    ```

## Running the Client

Once configured, the primary way to run the client from your local project directory is:

```bash
npm run start
```

This will execute the `start` script defined in `package.json` (which runs `node ./src/mcp-client.js`). The client will connect to the configured MCP servers, load their tools, and present you with a prompt.

**Alternative ways to run locally:**

*   **Directly with Node.js:**
    ```bash
    node src/mcp-client.js
    ```
*   **Using the `bin` command with `npx` (after `npm install`):
    ```bash
    npx universal-mcp-client
    ```

**For more convenient development (optional):**

If you want to run `universal-mcp-client` as a command from any directory without `npx`, you can use `npm link`:

1.  Navigate to your project root: `cd /path/to/universal-mcp-client`
2.  Run: `npm link`

Now you can type `universal-mcp-client` in your terminal anywhere to start the client from your local project.

*(If this package were published to npm, users could install it globally via `npm install -g universal-mcp-client` and then run `universal-mcp-client` directly.)*

## Installation from npm

Once published, users can install the client globally using npm:

```bash
npm install -g universal-mcp-client
```
*(Replace `universal-mcp-client` with your actual published package name if you had to change it, e.g., `@yourusername/universal-mcp-client`)*

Then, they can run the client from anywhere in their terminal:

```bash
universal-mcp-client
```

## CLI Commands

*   `/help`: Displays available commands.
*   `/setprovider <provider_name> [model_name]`: Switches the LLM provider. 
    *   Examples:
        *   `/setprovider openai gpt-4o`
        *   `/setprovider google gemini-1.5-pro-latest`
        *   `/setprovider anthropic` (uses default model for Anthropic)
        *   `/setprovider deepseek deepseek-chat` (or just `/setprovider deepseek` for default)
*   `/setsystem <your new system message>`: Sets a new system message for the LLM. If no message is provided, it displays the current system message. Note: For the Google provider, changing the system message will re-initialize its chat session *with the current conversation history preserved* to apply the new instructions.
*   `/clear`: Clears the current conversation history.
*   `/exit` or `/quit`: Exits the client.

Any other input is treated as a query to the current LLM provider.

## Multi-modal Image Handling

The client supports interactions with vision-capable LLMs from all configured providers (Anthropic, OpenAI, Google). 
When a tool (e.g., a screenshot tool) returns an image:

*   The client facilitates a multi-step interaction where the LLM first acknowledges the tool's output and then, in a subsequent step, receives the image data for analysis.
*   This allows you to ask the LLM to describe the image, answer questions about its content, or use it for further context.
*   When switching providers (`/setprovider`), the client attempts to preserve image context:
    *   Images processed by any provider's tools (e.g., OpenAI, Anthropic, Google) are added to the shared conversation history using a standardized internal format (`image_mcp`).
    *   When switching to a new vision-capable provider, its message formatting logic converts this stored image data into its native format (e.g., `image_url` for OpenAI, `image` block for Anthropic, `inlineData` for Google), making the image from the previous session accessible.
    *   This allows for sequences like: Google tool returns an image -> switch to OpenAI -> OpenAI can analyze that image.
*   **DeepSeek Image Handling:** Currently, the `deepseek-chat` model (and other DeepSeek models available via their API) do not support direct image input in messages. If the conversation history contains image messages (e.g., from a tool used with another provider), the `DeepSeekProvider` will convert these messages into a text summary indicating that image data was present but not sent to the DeepSeek LLM. Tool calls made *by* DeepSeek that return images will still have their images added to the shared history by the MCP Client for potential use by other providers.

## Configuration Details

### `config.json`

*   `llmProvider` (string): The initial LLM provider to use. Options: `"anthropic"`, `"openai"`, `"google"`, `"deepseek"`.
*   `systemMessage` (string, optional): A custom system message to guide the LLM's behavior. Overrides the internal default. Can also be set by the `SYSTEM_MESSAGE` environment variable.
*   `maxConversationHistoryLength` (number, optional): Limits the number of messages (user, assistant, tool interactions) kept in the conversation history. If 0 or omitted, no limit is applied.
*   `mcpServers` (object): An object where each key is a custom name for an MCP server, and the value is an object containing:
    *   `command` (string): The command to execute to start the MCP server (e.g., `"npx"`, `"node"`). **Required for `stdio` transport.**
    *   `args` (array of strings, optional): Arguments to pass to the command.
    *   `baseUrl` (string): The base URL for the MCP server. **Required for `streamable-http` transport.** (e.g., `"http://localhost:7000/mcp"`)
    *   `transport` (object): Defines the transport type.
        *   `type` (string): Can be `"stdio"` (default if omitted) or `"streamable-http"`.
        *   `headers` (object, optional): For `streamable-http` transport, key-value pairs for custom HTTP headers (e.g., for authentication).
    *   `env` (object, optional): Environment variables to set. For `stdio`, these are passed to the server's process. For `streamable-http`, they can be used to substitute placeholders in `baseUrl` or `headers` if such logic is implemented in the client (currently, direct values or `.env` substitution in `config.json` is standard).

### `.env` File

(Create by copying `.env.example`)

*   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GEMINI_API_KEY`, `DEEPSEEK_API_KEY`: Your respective API keys.
*   `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GOOGLE_GEMINI_MODEL`, `DEEPSEEK_MODEL` (optional): Default models to use for each provider if not specified by `/setprovider`. Examples: `claude-3-5-sonnet-latest`, `gpt-4o`, `gemini-1.5-pro-latest`, `deepseek-chat`.
*   `LOG_LEVEL` (optional): Sets the logging level for the client. Options: `"fatal"`, `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`, `"silent"`. Default is `"info"`.
*   `SYSTEM_MESSAGE` (optional): Overrides any system message set in `config.json` or the internal default. Note: For the Google provider, changing this environment variable (which requires a client restart) means the new chat session will start with this system message and a fresh conversation history.
*   `MAX_CONVERSATION_HISTORY_LENGTH` (optional): Overrides `maxConversationHistoryLength` from `config.json`.
*   **Server-Specific Variables:** Any other variables your configured MCP servers might need (e.g., `SLACK_BOT_TOKEN`, `GOFAST_CC_API_KEY`).

## Troubleshooting

*   **API Key Errors:** Ensure your API keys in `.env` are correct and have the necessary permissions for the models you are trying to use.
*   **MCP Server Connection Issues:** 
    *   Verify the `command` and `args` in `config.json` correctly point to your MCP server executables or scripts.
    *   Check that any `env` variables required by the MCP server are correctly set in `config.json` or your main `.env` file.
    *   Ensure the MCP server itself is functional and not outputting errors on its own startup.
*   **"Unsupported LLM provider" Error:** Make sure the provider name in `config.json` or `/setprovider` is one of `anthropic`, `openai`, `google`, or `deepseek`.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
