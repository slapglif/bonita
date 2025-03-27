document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    const thoughtProcess = document.getElementById('thought-process');
    
    // Business extraction elements
    const businessExtractionForm = document.getElementById('business-extraction-form');
    const excelFilePath = document.getElementById('excel-file-path');
    const maxConcurrency = document.getElementById('max-concurrency');
    const extractionStatus = document.getElementById('extraction-status');
    const statusValue = document.getElementById('status-value');
    const progressBar = document.getElementById('progress-bar');
    const extractionResults = document.getElementById('extraction-results');
    const resultsContainer = document.getElementById('results-container');
    const downloadLink = document.getElementById('download-link');

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
    
    // Business Extraction Functions
    
    // Start extraction process
    async function startExtraction(excelPath, maxConcurrencyValue) {
        try {
            const response = await fetch('/api/business/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    excel_path: excelPath,
                    max_concurrency: maxConcurrencyValue 
                }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                updateExtractionStatus(data.details);
                return true;
            } else {
                statusValue.textContent = 'Error: ' + (data.error || 'Unknown error');
                console.error('Error:', data.error);
                return false;
            }
        } catch (error) {
            statusValue.textContent = 'Error: ' + error.message;
            console.error('Error:', error);
            return false;
        }
    }
    
    // Check extraction status
    async function checkExtractionStatus() {
        try {
            const response = await fetch('/api/business/status');
            const data = await response.json();
            
            if (response.ok) {
                updateExtractionStatus(data);
                return data;
            } else {
                console.error('Error checking status:', data.error);
                return null;
            }
        } catch (error) {
            console.error('Error checking status:', error);
            return null;
        }
    }
    
    // Update the UI with extraction status
    function updateExtractionStatus(statusData) {
        if (!statusData) return;
        
        // Show the status section
        extractionStatus.classList.remove('d-none');
        
        // Update status text
        statusValue.textContent = statusData.status.charAt(0).toUpperCase() + statusData.status.slice(1);
        
        // Update progress bar
        let progressPercent = 0;
        if (statusData.total_count > 0) {
            progressPercent = Math.round((statusData.processed_count / statusData.total_count) * 100);
        }
        progressBar.style.width = progressPercent + '%';
        progressBar.setAttribute('aria-valuenow', progressPercent);
        progressBar.textContent = progressPercent + '%';
        
        // Show results if available
        if (statusData.status === 'completed' && statusData.sample_results.length > 0) {
            extractionResults.classList.remove('d-none');
            
            // Display sample results
            resultsContainer.innerHTML = '';
            
            // Create a table for the results
            const table = document.createElement('table');
            table.className = 'table table-sm table-bordered';
            
            // Create header
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            const headers = ['Business Name', 'Owner Name', 'Address', 'Confidence'];
            headers.forEach(headerText => {
                const th = document.createElement('th');
                th.textContent = headerText;
                headerRow.appendChild(th);
            });
            
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            // Create body with sample results
            const tbody = document.createElement('tbody');
            statusData.sample_results.forEach(result => {
                const row = document.createElement('tr');
                
                // Add cells
                const businessCell = document.createElement('td');
                businessCell.textContent = result.business_name;
                row.appendChild(businessCell);
                
                const ownerCell = document.createElement('td');
                ownerCell.textContent = result.owner_name;
                row.appendChild(ownerCell);
                
                const addressCell = document.createElement('td');
                addressCell.textContent = result.primary_address;
                row.appendChild(addressCell);
                
                const confidenceCell = document.createElement('td');
                confidenceCell.textContent = (result.confidence_score * 100).toFixed(0) + '%';
                row.appendChild(confidenceCell);
                
                tbody.appendChild(row);
            });
            
            table.appendChild(tbody);
            resultsContainer.appendChild(table);
            
            // Show download link if output path is available
            if (statusData.output_path) {
                downloadLink.href = '/api/business/download/' + statusData.output_path;
                downloadLink.classList.remove('d-none');
            } else {
                downloadLink.classList.add('d-none');
            }
        }
    }
    
    // Poll for status updates when extraction is running
    let statusInterval = null;
    function startStatusPolling() {
        // Clear any existing interval
        if (statusInterval) {
            clearInterval(statusInterval);
        }
        
        // Set up new polling interval
        statusInterval = setInterval(async () => {
            const status = await checkExtractionStatus();
            
            // Stop polling if process is completed or errored
            if (status && (status.status === 'completed' || status.status === 'error')) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        }, 2000); // Check every 2 seconds
    }
    
    // Business extraction form submission
    if (businessExtractionForm) {
        businessExtractionForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const excelPath = excelFilePath.value.trim();
            const maxConcurrencyValue = parseInt(maxConcurrency.value) || 5;
            
            if (!excelPath) {
                alert('Please enter a valid Excel file path');
                return;
            }
            
            // Reset status UI
            statusValue.textContent = 'Starting...';
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            progressBar.textContent = '0%';
            extractionResults.classList.add('d-none');
            extractionStatus.classList.remove('d-none');
            
            // Start extraction
            const success = await startExtraction(excelPath, maxConcurrencyValue);
            
            if (success) {
                // Start polling for status updates
                startStatusPolling();
            }
        });
    }
});
