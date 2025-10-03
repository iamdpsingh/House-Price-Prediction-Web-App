/**
 * House Price Predictor - JavaScript
 * Main client-side functionality for the prediction app
 */

class HousePricePredictor {
    constructor() {
        this.form = document.getElementById('prediction-form');
        this.predictBtn = document.getElementById('predict-btn');
        this.resultsContainer = document.getElementById('results-container');
        this.loadingOverlay = document.getElementById('loading-overlay');
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.initializeValidation();
        this.loadLocationData();
    }
    
    bindEvents() {
        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        }
        
        // Real-time validation
        const inputs = this.form?.querySelectorAll('input, select');
        inputs?.forEach(input => {
            input.addEventListener('input', () => this.validateField(input));
            input.addEventListener('blur', () => this.validateField(input));
        });
        
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        // Auto-hide toasts
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => {
                this.hideAllToasts();
            }, 5000);
        });
    }
    
    async loadLocationData() {
        try {
            const response = await fetch('/locations');
            const data = await response.json();
            
            if (data.success && data.locations) {
                this.updateLocationOptions(data.locations);
            }
        } catch (error) {
            console.warn('Failed to load location data:', error);
        }
    }
    
    updateLocationOptions(locations) {
        const locationSelect = document.getElementById('location');
        if (!locationSelect) return;
        
        // Clear existing options except the first one
        const firstOption = locationSelect.querySelector('option[value=""]');
        locationSelect.innerHTML = '';
        
        if (firstOption) {
            locationSelect.appendChild(firstOption);
        }
        
        // Add new options
        locations.forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        });
    }
    
    initializeValidation() {
        // Add validation rules
        const totalSqftInput = document.getElementById('total_sqft');
        if (totalSqftInput) {
            totalSqftInput.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (value < 300 || value > 8000) {
                    this.setFieldError(e.target, 'Area must be between 300-8000 sq ft');
                } else {
                    this.clearFieldError(e.target);
                }
            });
        }
    }
    
    validateField(field) {
        const value = field.value.trim();
        const fieldName = field.getAttribute('name');
        
        // Clear previous errors
        this.clearFieldError(field);
        
        // Required field validation
        if (field.hasAttribute('required') && !value) {
            this.setFieldError(field, `${this.getFieldLabel(fieldName)} is required`);
            return false;
        }
        
        // Specific field validation
        switch (fieldName) {
            case 'total_sqft':
                return this.validateTotalSqft(field, value);
            case 'bhk':
                return this.validateBHK(field, value);
            case 'bath':
                return this.validateBath(field, value);
            case 'location':
                return this.validateLocation(field, value);
            default:
                return true;
        }
    }
    
    validateTotalSqft(field, value) {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
            this.setFieldError(field, 'Please enter a valid number');
            return false;
        }
        if (numValue < 300) {
            this.setFieldError(field, 'Minimum area is 300 sq ft');
            return false;
        }
        if (numValue > 8000) {
            this.setFieldError(field, 'Maximum area is 8000 sq ft');
            return false;
        }
        return true;
    }
    
    validateBHK(field, value) {
        const numValue = parseInt(value);
        if (isNaN(numValue) || numValue < 1 || numValue > 6) {
            this.setFieldError(field, 'Please select a valid BHK');
            return false;
        }
        return true;
    }
    
    validateBath(field, value) {
        const numValue = parseInt(value);
        if (isNaN(numValue) || numValue < 1 || numValue > 5) {
            this.setFieldError(field, 'Please select valid number of bathrooms');
            return false;
        }
        return true;
    }
    
    validateLocation(field, value) {
        if (!value) {
            this.setFieldError(field, 'Please select a location');
            return false;
        }
        return true;
    }
    
    setFieldError(field, message) {
        field.classList.add('error');
        
        // Remove existing error message
        const existingError = field.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Add new error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
        
        // Add error styles
        const style = document.createElement('style');
        style.textContent = `
            .form-control.error {
                border-color: var(--danger) !important;
                box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
            }
            .field-error {
                color: var(--danger);
                font-size: var(--font-size-xs);
                margin-top: var(--space-1);
                display: flex;
                align-items: center;
                gap: var(--space-1);
            }
            .field-error::before {
                content: "⚠";
                font-size: var(--font-size-sm);
            }
        `;
        if (!document.querySelector('#validation-styles')) {
            style.id = 'validation-styles';
            document.head.appendChild(style);
        }
    }
    
    clearFieldError(field) {
        field.classList.remove('error');
        const errorDiv = field.parentNode.querySelector('.field-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    getFieldLabel(fieldName) {
        const labels = {
            'location': 'Location',
            'total_sqft': 'Total Area',
            'bhk': 'BHK',
            'bath': 'Bathrooms',
            'balcony': 'Balconies'
        };
        return labels[fieldName] || fieldName;
    }
    
    validateForm() {
        let isValid = true;
        const formElements = this.form.querySelectorAll('input[required], select[required]');
        
        formElements.forEach(element => {
            if (!this.validateField(element)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    async handleFormSubmit(e) {
        e.preventDefault();
        
        if (!this.validateForm()) {
            this.showToast('Please fix the errors in the form', 'error');
            return;
        }
        
        const formData = new FormData(this.form);
        const data = Object.fromEntries(formData);
        
        // Set default balcony value if not provided
        if (!data.balcony) {
            data.balcony = '0';
        }
        
        await this.makePrediction(data);
    }
    
    async makePrediction(data) {
        try {
            this.setLoadingState(true);
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.showToast('Prediction completed successfully!', 'success');
                
                // Smooth scroll to results
                setTimeout(() => {
                    this.resultsContainer.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                }, 100);
            } else {
                this.displayError(result.errors || [result.error || 'Prediction failed']);
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.displayError(['Network error. Please check your connection and try again.']);
            this.showToast('Network error occurred', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    setLoadingState(loading) {
        if (loading) {
            this.predictBtn.classList.add('loading');
            this.predictBtn.disabled = true;
            this.showLoadingOverlay();
        } else {
            this.predictBtn.classList.remove('loading');
            this.predictBtn.disabled = false;
            this.hideLoadingOverlay();
        }
    }
    
    showLoadingOverlay() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('show');
        }
    }
    
    hideLoadingOverlay() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('show');
        }
    }
    
    displayResults(result) {
        const prediction = result.prediction || result;
        const property = result.property_details || result.input_data;
        const metrics = result.metrics || {};
        const category = result.category || {};
        const insights = result.market_insights || {};
        
        const resultsHTML = `
            <div class="prediction-success">
                <div class="result-price">
                    <div class="price-main">
                        ${prediction.formatted_price || `₹${prediction.predicted_price} lakhs`}
                    </div>
                    <div class="price-range">
                        Range: ${this.formatPriceRange(prediction.predicted_price)}
                    </div>
                    <div class="price-category">
                        <span class="category-badge ${category.name ? category.name.toLowerCase().replace(' ', '-') : 'premium'}">
                            <i class="fas fa-tag"></i>
                            ${category.name || 'Premium'} Property
                        </span>
                    </div>
                </div>
                
                <div class="result-details">
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-map-marker-alt"></i>
                            Location
                        </span>
                        <span class="detail-value">${property.location}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-ruler-combined"></i>
                            Total Area
                        </span>
                        <span class="detail-value">${this.formatNumber(property.total_sqft)} sq ft</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-home"></i>
                            Configuration
                        </span>
                        <span class="detail-value">${property.bhk} BHK, ${property.bath} Bath</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-tree"></i>
                            Balconies
                        </span>
                        <span class="detail-value">${property.balcony}</span>
                    </div>
                    
                    ${metrics.price_per_sqft ? `
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-calculator"></i>
                            Price per Sq Ft
                        </span>
                        <span class="detail-value">₹${this.formatNumber(Math.round(metrics.price_per_sqft))}</span>
                    </div>
                    ` : ''}
                    
                    ${metrics.sqft_per_room ? `
                    <div class="detail-row">
                        <span class="detail-label">
                            <i class="fas fa-expand-arrows-alt"></i>
                            Sq Ft per Room
                        </span>
                        <span class="detail-value">${Math.round(metrics.sqft_per_room)} sq ft</span>
                    </div>
                    ` : ''}
                </div>
                
                ${insights.investment_potential ? `
                <div class="market-insights">
                    <h4>
                        <i class="fas fa-chart-line"></i>
                        Market Insights
                    </h4>
                    <div class="insights-grid">
                        <div class="insight-item">
                            <span class="insight-label">Location Tier</span>
                            <span class="insight-value">${insights.location_tier}</span>
                        </div>
                        <div class="insight-item">
                            <span class="insight-label">Investment Potential</span>
                            <span class="insight-value">${insights.investment_potential}</span>
                        </div>
                        <div class="insight-item">
                            <span class="insight-label">Market Position</span>
                            <span class="insight-value">${insights.market_position}</span>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="result-actions-mini">
                    <button class="action-btn secondary" onclick="window.housePredictor.shareResults('${prediction.predicted_price}', '${property.location}')">
                        <i class="fas fa-share-alt"></i>
                        Share Result
                    </button>
                    <button class="action-btn outline" onclick="window.housePredictor.downloadResults()">
                        <i class="fas fa-download"></i>
                        Download Report
                    </button>
                </div>
                
                <div class="prediction-disclaimer">
                    <p>
                        <i class="fas fa-info-circle"></i>
                        <strong>Note:</strong> This prediction is based on machine learning analysis and should be used for informational purposes only. Actual prices may vary based on market conditions.
                    </p>
                </div>
            </div>
        `;
        
        this.resultsContainer.innerHTML = resultsHTML;
        
        // Add CSS for new elements
        this.addResultStyles();
    }
    
    addResultStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .market-insights {
                margin-top: var(--space-6);
                padding: var(--space-4);
                background: var(--gray-50);
                border-radius: var(--radius-lg);
            }
            
            .market-insights h4 {
                display: flex;
                align-items: center;
                gap: var(--space-2);
                margin-bottom: var(--space-4);
                color: var(--gray-800);
            }
            
            .insights-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: var(--space-3);
            }
            
            .insight-item {
                text-align: center;
                padding: var(--space-3);
                background: white;
                border-radius: var(--radius-md);
            }
            
            .insight-label {
                display: block;
                font-size: var(--font-size-sm);
                color: var(--gray-600);
                margin-bottom: var(--space-1);
            }
            
            .insight-value {
                font-weight: 600;
                color: var(--primary);
                font-size: var(--font-size-base);
            }
            
            .result-actions-mini {
                display: flex;
                gap: var(--space-3);
                margin-top: var(--space-6);
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .prediction-disclaimer {
                margin-top: var(--space-6);
                padding: var(--space-4);
                background: var(--gray-100);
                border-radius: var(--radius-lg);
                border-left: 4px solid var(--primary);
            }
            
            .prediction-disclaimer p {
                font-size: var(--font-size-sm);
                color: var(--gray-700);
                line-height: 1.5;
                display: flex;
                align-items: flex-start;
                gap: var(--space-2);
            }
            
            .prediction-disclaimer i {
                color: var(--primary);
                margin-top: 2px;
                flex-shrink: 0;
            }
        `;
        
        if (!document.querySelector('#result-styles')) {
            style.id = 'result-styles';
            document.head.appendChild(style);
        }
    }
    
    displayError(errors) {
        const errorList = Array.isArray(errors) ? errors : [errors];
        
        const errorHTML = `
            <div class="prediction-error">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <h4>Unable to Make Prediction</h4>
                <div class="error-messages">
                    ${errorList.map(error => `
                        <div class="error-message">
                            <i class="fas fa-times-circle"></i>
                            <span>${error}</span>
                        </div>
                    `).join('')}
                </div>
                <button class="retry-btn" onclick="window.housePredictor.clearResults()">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
            </div>
        `;
        
        this.resultsContainer.innerHTML = errorHTML;
        this.addErrorStyles();
    }
    
    addErrorStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .prediction-error {
                text-align: center;
                padding: var(--space-8);
                color: var(--danger);
            }
            
            .error-icon {
                font-size: var(--font-size-4xl);
                margin-bottom: var(--space-4);
                opacity: 0.7;
            }
            
            .prediction-error h4 {
                color: var(--gray-800);
                margin-bottom: var(--space-4);
                font-size: var(--font-size-xl);
            }
            
            .error-messages {
                text-align: left;
                max-width: 400px;
                margin: 0 auto var(--space-6);
            }
            
            .error-message {
                display: flex;
                align-items: center;
                gap: var(--space-2);
                padding: var(--space-2);
                margin-bottom: var(--space-2);
                background: rgba(239, 68, 68, 0.1);
                border-radius: var(--radius-md);
                font-size: var(--font-size-sm);
                color: var(--danger);
            }
            
            .retry-btn {
                background: var(--danger);
                color: white;
                border: none;
                padding: var(--space-3) var(--space-6);
                border-radius: var(--radius-lg);
                font-weight: 500;
                cursor: pointer;
                transition: var(--transition);
                display: flex;
                align-items: center;
                gap: var(--space-2);
                margin: 0 auto;
            }
            
            .retry-btn:hover {
                background: color-mix(in srgb, var(--danger) 85%, black);
                transform: translateY(-1px);
            }
        `;
        
        if (!document.querySelector('#error-styles')) {
            style.id = 'error-styles';
            document.head.appendChild(style);
        }
    }
    
    clearResults() {
        this.resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <i class="fas fa-calculator"></i>
                </div>
                <h4>Ready to Predict</h4>
                <p>Fill in the property details and click "Predict Price" to get your instant valuation.</p>
            </div>
        `;
    }
    
    formatPriceRange(price) {
        const min = (price * 0.9).toFixed(2);
        const max = (price * 1.1).toFixed(2);
        return `₹${min} - ₹${max} lakhs`;
    }
    
    formatNumber(num) {
        return new Intl.NumberFormat('en-IN').format(num);
    }
    
    showToast(message, type = 'info') {
        const toastId = `${type}-toast`;
        const toast = document.getElementById(toastId);
        
        if (toast) {
            const messageElement = toast.querySelector('.toast-message');
            messageElement.textContent = message;
            toast.classList.add('show');
            
            // Auto hide after 5 seconds
            setTimeout(() => {
                toast.classList.remove('show');
            }, 5000);
        } else {
            // Fallback to alert if toast element not found
            alert(message);
        }
    }
    
    hideAllToasts() {
        document.querySelectorAll('.toast').forEach(toast => {
            toast.classList.remove('show');
        });
    }
    
    shareResults(price, location) {
        const shareData = {
            title: 'House Price Prediction Result',
            text: `Predicted price for property in ${location}: ₹${price} lakhs`,
            url: window.location.href
        };
        
        if (navigator.share) {
            navigator.share(shareData)
                .then(() => console.log('Share successful'))
                .catch((error) => console.log('Share failed:', error));
        } else if (navigator.clipboard) {
            const shareText = `${shareData.text} - ${shareData.url}`;
            navigator.clipboard.writeText(shareText)
                .then(() => this.showToast('Results link copied to clipboard!', 'success'))
                .catch(() => this.showToast('Could not copy to clipboard', 'error'));
        } else {
            // Fallback
            const shareText = `${shareData.text} - ${shareData.url}`;
            prompt('Copy this link:', shareText);
        }
    }
    
    downloadResults() {
        // Get current results data
        const priceElement = document.querySelector('.price-main');
        const locationElement = document.querySelector('.detail-value');
        
        if (!priceElement || !locationElement) {
            this.showToast('No results to download', 'error');
            return;
        }
        
        const reportData = {
            prediction: {
                price: priceElement.textContent.trim(),
                location: locationElement.textContent.trim(),
                timestamp: new Date().toISOString()
            },
            disclaimer: 'This prediction is for informational purposes only. Actual prices may vary.',
            generated_by: 'House Price Predictor AI'
        };
        
        const dataStr = JSON.stringify(reportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `house_price_prediction_${Date.now()}.json`;
        link.click();
        
        URL.revokeObjectURL(link.href);
        this.showToast('Report downloaded successfully!', 'success');
    }
}

// Utility functions
window.hideToast = function(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.classList.remove('show');
    }
};

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.housePredictor = new HousePricePredictor();
    
    // Add smooth scroll behavior for all internal links
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add loading states for external links
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        link.addEventListener('click', function() {
            this.style.opacity = '0.7';
            this.style.pointerEvents = 'none';
            setTimeout(() => {
                this.style.opacity = '1';
                this.style.pointerEvents = 'auto';
            }, 1000);
        });
    });
});

// Handle page visibility for performance
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden - pause any animations or polling
        console.log('Page hidden - pausing activities');
    } else {
        // Page is visible - resume activities
        console.log('Page visible - resuming activities');
    }
});

// Add error handling for uncaught errors
window.addEventListener('error', function(e) {
    console.error('Uncaught error:', e.error);
    // Could send to error tracking service
});

// Add handling for promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    e.preventDefault();
});
