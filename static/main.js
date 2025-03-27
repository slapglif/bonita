document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    const thoughtProcess = document.getElementById('thought-process');

    // Function to append a message to the chat history
    function appendMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process markdown-like formatting
        let processedContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre class="code-block">$1</pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        messageContent.innerHTML = processedContent;
        messageDiv.appendChild(messageContent);
        chatHistory.appendChild(messageDiv);
        
        // Scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Function to display thought process
    function displayThoughtProcess(steps) {
        if (!steps || steps.length === 0) {
            thoughtProcess.innerHTML = '<p class="text-muted">No thought process available for this response.</p>';
            return;
        }

        let formattedSteps = '';
        steps.forEach((step, index) => {
            if (typeof step === 'string') {
                formattedSteps += `Step ${index + 1}: ${step}\n\n`;
            } else if (typeof step === 'object') {
                // Handle different possible formats
                if (step.action && step.action_input) {
                    formattedSteps += `Step ${index + 1}:\nAction: ${step.action}\nInput: ${step.action_input}\n`;
                    if (step.observation) {
                        formattedSteps += `Observation: ${step.observation}\n`;
                    }
                    formattedSteps += '\n';
                } else {
                    formattedSteps += `Step ${index + 1}: ${JSON.stringify(step, null, 2)}\n\n`;
                }
            }
        });

        thoughtProcess.innerText = formattedSteps;
    }

    // Function to send a message to the API
    async function sendMessage(message) {
        thinkingIndicator.classList.remove('d-none');
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                appendMessage(data.response);
                displayThoughtProcess(data.thought_process);
            } else {
                appendMessage(`Error: ${data.error || 'Unknown error occurred'}`, false);
                console.error('Error:', data.error);
            }
        } catch (error) {
            appendMessage(`Error: ${error.message}`, false);
            console.error('Error:', error);
        } finally {
            thinkingIndicator.classList.add('d-none');
        }
    }

    // Event listener for send button
    sendButton.addEventListener('click', function() {
        const message = userInput.value.trim();
        if (message) {
            appendMessage(message, true);
            userInput.value = '';
            sendMessage(message);
        }
    });

    // Event listener for Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const message = userInput.value.trim();
            if (message) {
                appendMessage(message, true);
                userInput.value = '';
                sendMessage(message);
            }
        }
    });

    // Event listener for clear chat button
    clearChatButton.addEventListener('click', function() {
        // Keep only the welcome message
        chatHistory.innerHTML = `
            <div class="welcome-message">
                <div class="message assistant-message">
                    <div class="message-content">
                        <p>Hello! I'm your AI assistant with memory and web search capabilities. How can I help you today?</p>
                    </div>
                </div>
            </div>
        `;
        thoughtProcess.innerHTML = '<p class="text-muted">The agent\'s thought process will appear here...</p>';
    });

    // Focus on input field when page loads
    userInput.focus();
});
