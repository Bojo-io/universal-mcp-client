/**
 * Simple logging utility with different log levels
 */
class Logger {
    constructor() {
      this.logLevel = 'trace'; // default log level
      this.levels = {
        error: 0,
        warn: 1,
        info: 2,
        debug: 3,
        trace: 4
      };
    }
  
    /**
     * Set the logging level
     * @param {string} level - Logging level (error, warn, info, debug, trace)
     */
    setLevel(level) {
      if (this.levels[level] !== undefined) {
        this.logLevel = level;
      } else {
        console.warn(`Invalid log level: ${level}. Using 'info' instead.`);
        this.logLevel = 'info';
      }
    }
  
    /**
     * Check if a log level is enabled
     * @param {string} level - Log level to check
     * @returns {boolean} Whether the level is enabled
     */
    isLevelEnabled(level) {
      return this.levels[level] <= this.levels[this.logLevel];
    }
  
    /**
     * Format a message with timestamp
     * @param {string} level - Log level
     * @param {string} message - Log message
     * @returns {string} Formatted message
     */
    formatMessage(level, message) {
      const timestamp = new Date().toISOString();
      return `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    }
  
    /**
     * Log an error message
     * @param {string} message - Error message
     */
    error(message) {
      if (this.isLevelEnabled('error')) {
        console.error(this.formatMessage('error', message));
      }
    }
  
    /**
     * Log a warning message
     * @param {string} message - Warning message
     */
    warn(message) {
      if (this.isLevelEnabled('warn')) {
        console.warn(this.formatMessage('warn', message));
      }
    }
  
    /**
     * Log an info message
     * @param {string} message - Info message
     */
    info(message) {
      if (this.isLevelEnabled('info')) {
        console.info(this.formatMessage('info', message));
      }
    }
  
    /**
     * Log a debug message
     * @param {string} message - Debug message
     */
    debug(message) {
      if (this.isLevelEnabled('debug')) {
        console.debug(this.formatMessage('debug', message));
      }
    }
  
    /**
     * Log a trace message
     * @param {string} message - Trace message
     */
    trace(message) {
      if (this.isLevelEnabled('trace')) {
        console.log(this.formatMessage('trace', message));
      }
    }
  
    /**
     * Log a separator line for better visual organization
     */
    separator() {
      if (this.isLevelEnabled('info')) {
        const separatorLine = '-'.repeat(80);
        console.info(this.formatMessage('info', separatorLine));
      }
    }
  }
  
  // Export a singleton instance
  export const logger = new Logger();
  
  // Also export the class for testing or if multiple instances are needed
  export default Logger; 