// Global Variables
let uploadedFile = null;
let extractedText = '';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const browseBtn = document.getElementById('browseBtn');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const languageSelect = document.getElementById('languageSelect');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const demoForm = document.getElementById('demoForm');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    initializeFAQ();
    smoothScrollSetup();
});

// Event Listeners
function initializeEventListeners() {
    // File upload handlers
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    removeBtn.addEventListener('click', removeFile);
    
    // Drag and drop handlers
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeReport);
    
    // New analysis button
    newAnalysisBtn.addEventListener('click', startNewAnalysis);
    
    // Demo form
    demoForm.addEventListener('submit', handleDemoRequest);
    
    // Mobile menu
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', toggleMobileMenu);
    }
}

// File Upload Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showNotification('Please upload an image file', 'error');
    }
}

function processFile(file) {
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('File size should be less than 10MB', 'error');
        return;
    }
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please upload an image file', 'error');
        return;
    }
    
    uploadedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.upload-content').style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeFile() {
    uploadedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    document.querySelector('.upload-content').style.display = 'block';
    previewArea.style.display = 'none';
    
    // Check if manual text is entered
    if (!manualTextInput.value.trim()) {
        analyzeBtn.disabled = true;
    }
}

// Analyze Report
async function analyzeReport() {
    // Check if we have either a file or manual text
    if (!uploadedFile) {
        showNotification('Please upload a medical report image', 'error');
        return;
    }
    
    // Show loading state
    analyzeBtn.style.display = 'none';
    loadingState.style.display = 'block';
    
    try {
        let extractedText = '';
        let ocrResult = null;
        
        // Extract text using OCR from uploaded image
        if (uploadedFile) {
            ocrResult = await extractTextFromImage(uploadedFile);
            
            if (!ocrResult.success) {
                throw new Error(ocrResult.message || 'Failed to extract text from image');
            }
            
            extractedText = ocrResult.extracted_text;
        }
        
        if (!extractedText || extractedText.length < 10) {
            throw new Error('Extracted text is too short. Please provide a clear medical report image.');
        }
        
        // Get BOTH analysis AND recommendations in ONE API call (reduces rate limits)
        const completeResult = await getCompleteAnalysis(extractedText, languageSelect.value);
        
        console.log('Complete result structure:', completeResult);
        
        // Display results - completeResult has {analysis: {...}, recommendations: {...}}
        displayResults(ocrResult, completeResult.analysis, completeResult);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(error.message || 'An error occurred during analysis', 'error');
    } finally {
        analyzeBtn.style.display = 'flex';
        loadingState.style.display = 'none';
    }
}

// API Calls
async function extractTextFromImage(file) {
    try {
        console.log('Starting client-side OCR with Tesseract.js...');
        
        // Use Tesseract.js for client-side OCR (no server memory needed)
        const result = await Tesseract.recognize(
            file,
            'eng',
            {
                logger: m => {
                    if (m.status === 'recognizing text') {
                        console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
                    }
                }
            }
        );
        
        const extractedText = result.data.text;
        console.log(`OCR completed. Extracted ${extractedText.length} characters`);
        
        return {
            success: true,
            extracted_text: extractedText,
            message: 'Text extracted successfully'
        };
    } catch (error) {
        console.error('OCR error:', error);
        return {
            success: false,
            message: 'Failed to extract text from image. Please try again.'
        };
    }
}

// Combined API call - reduces requests by 50%
async function getCompleteAnalysis(reportText, language) {
    try {
        console.log('Requesting complete analysis in language:', language);
        const response = await fetch('/api/analyze-complete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                report_text: reportText,
                language: language
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            
            if (response.status === 429) {
                throw new Error('⏳ Too many requests. Please wait 30 seconds and try again.');
            }
            
            throw new Error(errorData.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        console.log('=== COMPLETE ANALYSIS RECEIVED ===');
        console.log('Full result:', JSON.stringify(result, null, 2));
        console.log('Analysis:', result.analysis);
        console.log('Recommendations:', result.recommendations);
        if (result.recommendations) {
            console.log('diet_advice:', result.recommendations.diet_advice);
            console.log('lifestyle_advice:', result.recommendations.lifestyle_advice);
        }
        console.log('=================================');
        return result;
    } catch (error) {
        console.error('Complete analysis error:', error);
        throw error;
    }
}

// Keep old function for backward compatibility
async function getReportAnalysis(reportText, language) {
    try {
        console.log('Requesting analysis in language:', language);
        const response = await fetch('/api/analyze-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                report_text: reportText,
                language: language
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            
            // Handle rate limit specifically
            if (response.status === 429) {
                throw new Error('⏳ Please wait a moment and try again. The AI service is temporarily busy.');
            }
            
            throw new Error(errorData.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        console.log('Analysis result received:', result);
        return result;
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
}

async function getRecommendations(reportText, language) {
    try {
        console.log('Requesting recommendations in language:', language);
        const response = await fetch('/api/recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                report_text: reportText,
                language: language
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            
            // Handle rate limit specifically
            if (response.status === 429) {
                throw new Error('⏳ Please wait a moment. The AI service is temporarily busy.');
            }
            
            throw new Error(errorData.detail || 'Recommendations failed');
        }
        
        const result = await response.json();
        console.log('Recommendations received:', result);
        return result;
    } catch (error) {
        console.error('Recommendations error:', error);
        throw error;
    }
}

// Display Results
function displayResults(ocrResult, analysisResult, completeResult) {
    console.log('=== DISPLAY RESULTS CALLED ===');
    console.log('completeResult:', completeResult);
    console.log('analysisResult:', analysisResult);
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display simplified explanation with formatting
    const explanationEl = document.getElementById('simplifiedExplanation');
    const explanation = analysisResult.simplified_explanation || 'Analysis in progress...';
    explanationEl.innerHTML = formatText(explanation);
    
    console.log('Checking recommendations...');
    console.log('Has recommendations?', !!completeResult.recommendations);
    
    // Display recommendations - check if recommendations object exists
    if (completeResult.recommendations) {
        const recs = completeResult.recommendations;
        console.log('Recommendations object:', recs);
        console.log('Type of diet_advice:', typeof recs.diet_advice);
        console.log('diet_advice value:', recs.diet_advice);
        console.log('lifestyle_advice value:', recs.lifestyle_advice);
        
        // Try both formats: {recommendations: [...]} and direct array
        const dietItems = Array.isArray(recs.diet_advice) ? recs.diet_advice : (recs.diet_advice?.recommendations || []);
        const lifestyleItems = Array.isArray(recs.lifestyle_advice) ? recs.lifestyle_advice : (recs.lifestyle_advice?.recommendations || []);
        const exerciseItems = Array.isArray(recs.exercise_tips) ? recs.exercise_tips : (recs.exercise_tips?.recommendations || []);
        const yogaItems = Array.isArray(recs.yoga_tips) ? recs.yoga_tips : (recs.yoga_tips?.recommendations || []);
        const avoidItems = Array.isArray(recs.things_to_avoid) ? recs.things_to_avoid : (recs.things_to_avoid?.recommendations || []);
        const benefitsItems = Array.isArray(recs.benefits) ? recs.benefits : (recs.benefits?.recommendations || []);
        
        console.log('Extracted items:');
        console.log('- Diet:', dietItems);
        console.log('- Lifestyle:', lifestyleItems);
        
        displayList('dietAdvice', dietItems);
        displayList('lifestyleAdvice', lifestyleItems);
        displayList('exerciseTips', exerciseItems);
        displayList('yogaTips', yogaItems);
        displayList('thingsToAvoid', avoidItems);
        displayList('benefits', benefitsItems);
    } else {
        console.error('❌ No recommendations found in response!');
    }
    console.log('=== DISPLAY COMPLETE ===');
}

// Format text with HTML elements
function formatText(text) {
    // Split by paragraphs and add formatting
    const paragraphs = text.split('\n\n').filter(p => p.trim());
    return paragraphs.map(p => {
        // Check if it's a heading (contains colon or all caps)
        if (p.includes(':') && p.split(':')[0].length < 50) {
            const [heading, ...content] = p.split(':');
            return `<div class="formatted-section"><strong class="section-heading">${heading}:</strong> ${content.join(':')}</div>`;
        }
        return `<p class="formatted-paragraph">${p}</p>`;
    }).join('');
}

function displayList(elementId, items) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element with ID '${elementId}' not found`);
        return;
    }
    
    console.log(`Displaying ${items?.length || 0} items for ${elementId}`);
    
    element.innerHTML = '';
    
    if (!items || items.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty-state';
        li.textContent = 'No recommendations available';
        element.appendChild(li);
        return;
    }
    
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        element.appendChild(li);
    });
}

// Start New Analysis
function startNewAnalysis() {
    // Reset everything
    removeFile();
    extractedText = '';
    resultsSection.style.display = 'none';
    
    // Scroll to top
    document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
}

// FAQ Functionality
function initializeFAQ() {
    const faqQuestions = document.querySelectorAll('.faq-question');
    
    faqQuestions.forEach(question => {
        question.addEventListener('click', () => {
            const faqItem = question.parentElement;
            const isActive = faqItem.classList.contains('active');
            
            // Close all FAQ items
            document.querySelectorAll('.faq-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Open clicked item if it wasn't active
            if (!isActive) {
                faqItem.classList.add('active');
            }
        });
    });
}

// Smooth Scrolling
function smoothScrollSetup() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// Mobile Menu Toggle
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks.style.display === 'flex') {
        navLinks.style.display = 'none';
    } else {
        navLinks.style.display = 'flex';
        navLinks.style.flexDirection = 'column';
        navLinks.style.position = 'absolute';
        navLinks.style.top = '100%';
        navLinks.style.left = '0';
        navLinks.style.right = '0';
        navLinks.style.backgroundColor = 'white';
        navLinks.style.padding = '1rem';
        navLinks.style.boxShadow = '0 5px 15px rgba(0,0,0,0.1)';
    }
}

// Demo Request Handler
function handleDemoRequest(e) {
    e.preventDefault();
    const email = e.target.querySelector('input[type="email"]').value;
    
    // Here you would typically send this to your backend
    console.log('Demo request for:', email);
    
    showNotification('Thank you! We\'ll notify you when the demo is ready.', 'success');
    e.target.reset();
}

// Notification System
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">&times;</button>
    `;
    
    // Add styles
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        backgroundColor: getNotificationColor(type),
        color: 'white',
        borderRadius: '10px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.2)',
        zIndex: '10000',
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        minWidth: '300px',
        animation: 'slideIn 0.3s ease'
    });
    
    // Add close handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    });
    
    // Add to document
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function getNotificationColor(type) {
    switch (type) {
        case 'success': return '#4caf50';
        case 'error': return '#f44336';
        case 'warning': return '#ff9800';
        default: return '#2196f3';
    }
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        transition: background-color 0.3s;
    }
    
    .notification-close:hover {
        background-color: rgba(0, 0, 0, 0.2);
    }
`;
document.head.appendChild(style);

// Language Loading
async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        const data = await response.json();
        
        if (data.success) {
            const select = document.getElementById('languageSelect');
            select.innerHTML = '';
            
            data.languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang.code;
                option.textContent = lang.name;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Failed to load languages:', error);
    }
}

// Load languages on page load
// loadLanguages();

// Legal Modals
const privacyModal = document.getElementById('privacyModal');
const termsModal = document.getElementById('termsModal');
const disclaimerModal = document.getElementById('disclaimerModal');

const privacyLink = document.getElementById('privacyLink');
const termsLink = document.getElementById('termsLink');
const disclaimerLink = document.getElementById('disclaimerLink');

// Open modals
if (privacyLink) {
    privacyLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(privacyModal);
    });
}

if (termsLink) {
    termsLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(termsModal);
    });
}

if (disclaimerLink) {
    disclaimerLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(disclaimerModal);
    });
}

// Close buttons
document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const modal = e.target.closest('.legal-modal');
        closeModal(modal);
    });
});

// Close on outside click
window.addEventListener('click', (e) => {
    if (e.target.classList.contains('legal-modal')) {
        closeModal(e.target);
    }
});

// Close on ESC key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.legal-modal.active').forEach(modal => {
            closeModal(modal);
        });
    }
});

function openModal(modal) {
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(modal) {
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
}
