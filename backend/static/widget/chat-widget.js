/**
 * RAG Chatbot Widget
 * Embeddable chat interface for digital books
 */
class RagChatWidget {
    constructor(options = {}) {
        this.apiUrl = options.apiUrl || 'http://localhost:8000';
        this.bookId = options.bookId || null;
        this.containerId = options.containerId || 'rag-chatbot-widget';
        this.widget = null;
        this.isOpen = false;
        this.messageHistory = [];

        // Bind methods
        this.init = this.init.bind(this);
        this.toggleWidget = this.toggleWidget.bind(this);
        this.sendMessage = this.sendMessage.bind(this);
        this.handleSendMessage = this.handleSendMessage.bind(this);
        this.displayMessage = this.displayMessage.bind(this);
        this.displayBotResponse = this.displayBotResponse.bind(this);
        this.showTypingIndicator = this.showTypingIndicator.bind(this);
        this.hideTypingIndicator = this.hideTypingIndicator.bind(this);
    }

    init() {
        // Create the widget HTML
        this.createWidget();

        // Add event listeners
        this.addEventListeners();

        console.log('RAG Chatbot Widget initialized');
    }

    createWidget() {
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'rag-chatbot-toggle';
        toggleButton.id = 'rag-chatbot-toggle';
        toggleButton.innerHTML = '💬';
        toggleButton.title = 'Open Chat with Book';
        document.body.appendChild(toggleButton);

        // Create main widget container
        this.widget = document.createElement('div');
        this.widget.className = 'rag-chatbot-widget';
        this.widget.id = this.containerId;
        this.widget.style.display = 'none'; // Initially hidden

        this.widget.innerHTML = `
            <div class="rag-chatbot-header">
                <h3 class="rag-chatbot-title">Book Assistant</h3>
                <button class="rag-chatbot-close" title="Close">×</button>
            </div>
            <div class="rag-chatbot-body">
                <div class="rag-chatbot-messages" id="rag-chatbot-messages">
                    <div class="rag-chatbot-message bot">
                        <div class="rag-chatbot-message-content">
                            Hello! I'm your book assistant. Ask me anything about this book.
                        </div>
                    </div>
                </div>
                <div class="rag-chatbot-typing-indicator" id="rag-chatbot-typing">
                    <div class="rag-chatbot-typing-dots">
                        <div class="rag-chatbot-typing-dot"></div>
                        <div class="rag-chatbot-typing-dot"></div>
                        <div class="rag-chatbot-typing-dot"></div>
                    </div>
                </div>
                <div class="rag-chatbot-input-area">
                    <input
                        type="text"
                        class="rag-chatbot-input"
                        id="rag-chatbot-input"
                        placeholder="Ask about the book..."
                        autocomplete="off"
                    >
                    <button class="rag-chatbot-send-button" id="rag-chatbot-send" title="Send">
                        <span>➤</span>
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(this.widget);
    }

    addEventListeners() {
        // Toggle button
        document.getElementById('rag-chatbot-toggle').addEventListener('click', this.toggleWidget);

        // Close button
        this.widget.querySelector('.rag-chatbot-close').addEventListener('click', this.toggleWidget);

        // Send button
        document.getElementById('rag-chatbot-send').addEventListener('click', this.handleSendMessage);

        // Input key events
        const input = document.getElementById('rag-chatbot-input');
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSendMessage();
            }
        });

        // Prevent widget from closing when clicking inside
        this.widget.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // Close widget when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.widget.contains(e.target) &&
                e.target.id !== 'rag-chatbot-toggle' &&
                this.isOpen) {
                this.toggleWidget();
            }
        });
    }

    toggleWidget() {
        if (this.isOpen) {
            this.widget.style.display = 'none';
            document.getElementById('rag-chatbot-toggle').style.display = 'block';
        } else {
            document.getElementById('rag-chatbot-toggle').style.display = 'none';
            this.widget.style.display = 'flex';
            document.getElementById('rag-chatbot-input').focus();
        }
        this.isOpen = !this.isOpen;
    }

    async handleSendMessage() {
        const input = document.getElementById('rag-chatbot-input');
        const message = input.value.trim();

        if (!message) return;

        // Clear input
        input.value = '';

        // Display user message
        this.displayMessage(message, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Determine if we should use selected text mode or full book mode
            const selectedText = this.getSelectedText();

            let response;
            if (selectedText && selectedText.length > 10) { // If user has selected text
                response = await this.sendSelectedTextQuery(message, selectedText);
            } else {
                response = await this.sendFullBookQuery(message);
            }

            // Hide typing indicator
            this.hideTypingIndicator();

            // Display bot response
            this.displayBotResponse(response);
        } catch (error) {
            // Hide typing indicator
            this.hideTypingIndicator();

            // Display error message
            this.displayMessage('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Error sending message:', error);
        }
    }

    getSelectedText() {
        const selection = window.getSelection();
        return selection.toString().trim();
    }

    async sendFullBookQuery(query) {
        if (!this.bookId) {
            throw new Error('Book ID is required for full book queries');
        }

        const response = await fetch(`${this.apiUrl}/api/v1/query/full-book`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                book_id: this.bookId,
                query: query,
                top_k: 7,
                enable_rerank: true
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        return await response.json();
    }

    async sendSelectedTextQuery(query, selectedText) {
        const response = await fetch(`${this.apiUrl}/api/v1/query/selected-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                selected_text: selectedText,
                query: query,
                temp_collection_ttl_seconds: 300
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        return await response.json();
    }

    displayMessage(text, sender) {
        const messagesContainer = document.getElementById('rag-chatbot-messages');

        const messageDiv = document.createElement('div');
        messageDiv.className = `rag-chatbot-message ${sender}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'rag-chatbot-message-content';
        contentDiv.textContent = text;

        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Add to history
        this.messageHistory.push({ sender, text, timestamp: new Date() });
    }

    displayBotResponse(response) {
        this.displayMessage(response.response_text, 'bot');

        // Display citations if available
        if (response.citations && response.citations.length > 0) {
            const messagesContainer = document.getElementById('rag-chatbot-messages');
            const lastMessage = messagesContainer.lastChild;

            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'rag-chatbot-citations visible';
            citationsDiv.innerHTML = '<strong>Citations:</strong><br>' +
                response.citations.map(citation =>
                    `Pages: ${citation.page_numbers.join(', ')} | Snippet: ${citation.snippet.substring(0, 100)}...`
                ).join('<br>');

            lastMessage.appendChild(citationsDiv);
        }

        // Scroll to bottom
        const messagesContainer = document.getElementById('rag-chatbot-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTypingIndicator() {
        const typingIndicator = document.getElementById('rag-chatbot-typing');
        typingIndicator.classList.add('visible');

        // Scroll to bottom
        const messagesContainer = document.getElementById('rag-chatbot-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('rag-chatbot-typing');
        typingIndicator.classList.remove('visible');
    }
}

// Auto-initialize if used as a module
if (typeof window !== 'undefined') {
    window.RagChatWidget = RagChatWidget;
}