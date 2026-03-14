// ===== AUTHENTICATION =====
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/me');
        if (response.ok) {
            const data = await response.json();
            updateUIForLoggedIn(data.name);
            return data;
        } else {
            updateUIForLoggedOut();
            return null;
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        updateUIForLoggedOut();
        return null;
    }
}

function updateUIForLoggedIn(username) {
    const authDiv = document.querySelector('.auth-buttons');
    if (authDiv) {
        authDiv.innerHTML = `
            <span class="user-greeting">Hi, ${username}</span>
            <a href="/history" class="btn btn-outline"><i class="fas fa-history"></i> History</a>
            <button class="btn" onclick="logout()"><i class="fas fa-sign-out-alt"></i> Logout</button>
        `;
    }
}

function updateUIForLoggedOut() {
    const authDiv = document.querySelector('.auth-buttons');
    if (authDiv) {
        authDiv.innerHTML = `
            <a href="/login" class="btn btn-outline"><i class="fas fa-sign-in-alt"></i> Login</a>
            <a href="/signup" class="btn"><i class="fas fa-user-plus"></i> Sign Up</a>
        `;
    }
}

async function login(username, password) {
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: username, password })
        });
        const data = await response.json();
        if (response.ok) {
            showNotification('Login successful!', 'success');
            checkAuthStatus();
            const params = new URLSearchParams(window.location.search);
            const redirect = params.get('redirect') || '/';
            window.location.href = redirect;
        } else {
            showNotification(data.error || 'Login failed', 'error');
        }
    } catch (error) {
        showNotification('Network error', 'error');
    }
}

async function signup(username, email, password) {
    try {
        const response = await fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: username, email, password })
        });
        const data = await response.json();
        if (response.ok) {
            showNotification('Registration successful!', 'success');
            checkAuthStatus();
            window.location.href = '/';
        } else {
            showNotification(data.error || 'Registration failed', 'error');
        }
    } catch (error) {
        showNotification('Network error', 'error');
    }
}

async function logout() {
    try {
        const response = await fetch('/api/logout', { method: 'POST' });
        if (response.ok) {
            showNotification('Logged out', 'success');
            checkAuthStatus();
            if (window.location.pathname.includes('history')) {
                window.location.href = '/';
            }
        }
    } catch (error) {
        showNotification('Error logging out', 'error');
    }
}

// Save disease result if logged in
async function saveDiseaseResult(resultData) {
    const user = await checkAuthStatus();
    if (!user) return;

    try {
        await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: user.id,
                type: 'disease',
                crop: resultData.crop_name,
                result: `${resultData.disease_name} (${resultData.confidence}%)`,
                date: new Date().toISOString().split('T')[0]
            })
        });
    } catch (error) {
        console.error('Failed to save disease result:', error);
    }
}

// Save crop result if logged in
async function saveCropResult(resultData) {
    const user = await checkAuthStatus();
    if (!user) return;

    try {
        await fetch('/api/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: user.id,
                type: 'crop',
                crop: resultData.recommended_crops[0],
                result: JSON.stringify(resultData.recommended_crops),
                date: new Date().toISOString().split('T')[0]
            })
        });
    } catch (error) {
        console.error('Failed to save crop result:', error);
    }
}

// ===== HISTORY PAGE (Handled by React in react_app.js) =====

// ===== TAB FUNCTIONALITY =====
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    if (!tabButtons.length) return;

    tabButtons[0]?.classList.add('active');
    tabContents[0]?.classList.add('active');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');

            tabButtons.forEach(btn => {
                btn.classList.remove('active');
                btn.setAttribute('aria-selected', 'false');
            });

            tabContents.forEach(content => {
                content.classList.remove('active');
                content.setAttribute('hidden', 'true');
            });

            button.classList.add('active');
            button.setAttribute('aria-selected', 'true');

            const activeContent = document.getElementById(tabId);
            activeContent.classList.add('active');
            activeContent.removeAttribute('hidden');
        });
    });
}

// ===== DISEASE DETECTION =====
function initDiseaseDetection() {
    const plantSelect = document.getElementById('plant-select');
    const imageUpload = document.getElementById('leaf-image');
    const imagePreview = document.getElementById('image-preview');
    const uploadArea = document.querySelector('.upload-area');
    const detectButton = document.getElementById('detect-disease');
    const resultContainer = document.getElementById('disease-result');

    if (!imageUpload) return;

    // Backend handles disease detection via AI models

    imageUpload.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            if (!validateImageFile(file)) return;

            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadArea.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(46, 125, 50, 0.15)';
    });

    uploadArea.addEventListener('dragleave', function (e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(46, 125, 50, 0.05)';
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        this.style.backgroundColor = 'rgba(46, 125, 50, 0.05)';

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            if (!validateImageFile(file)) return;

            imageUpload.files = e.dataTransfer.files;

            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadArea.style.display = 'none';
            };
            reader.readAsDataURL(file);
        } else {
            showNotification('Please upload an image file', 'error');
        }
    });

    uploadArea.addEventListener('click', function () {
        imageUpload.click();
    });

    if (detectButton) {
        detectButton.addEventListener('click', async function () {
            const selectedPlant = plantSelect.value;

            if (!selectedPlant) {
                showNotification('Please select a plant first', 'warning');
                return;
            }

            if (!imageUpload.files.length) {
                showNotification('Please upload an image first', 'warning');
                return;
            }

            detectButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
            detectButton.disabled = true;

            try {
                const formData = new FormData();
                formData.append('crop', selectedPlant.charAt(0).toUpperCase() + selectedPlant.slice(1));
                formData.append('image', imageUpload.files[0]);

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Detection failed');
                }

                const result = await response.json();

                resultContainer.innerHTML = `
                    <div class="result-header">
                        <h3 class="disease-name">${result.disease}</h3>
                        <span class="confidence-badge">${(result.confidence * 100).toFixed(1)}% Confidence</span>
                    </div>
                    ${result.isHealthy ?
                        '<p>Your plant appears to be healthy. Continue with regular care and monitoring.</p>' :
                        `<div class="treatment-info">
                            <h4><i class="fas fa-prescription-bottle-medical"></i> Status:</h4>
                            <p>Potential disease detected. Please consult an expert for treatment.</p>
                        </div>`
                    }
                    <div class="prevention-tips">
                        <h4><i class="fas fa-shield-alt"></i> General Care Tips:</h4>
                        <ul>
                            <li>Remove and destroy infected leaves</li>
                            <li>Avoid overhead watering</li>
                            <li>Ensure proper plant spacing for air circulation</li>
                            <li>Rotate crops annually</li>
                        </ul>
                    </div>
                `;

                resultContainer.classList.add('show');

                // Save to history if logged in
                saveDiseaseResult({
                    crop_name: selectedPlant,
                    disease_name: result.disease,
                    confidence: (result.confidence * 100).toFixed(1)
                });

            } catch (error) {
                showNotification(error.message, 'error');
            } finally {
                detectButton.innerHTML = '<i class="fas fa-search"></i> Detect Disease';
                detectButton.disabled = false;
            }
        });
    }
}

// ===== CROP RECOMMENDATION =====
function initCropRecommendation() {
    const cropForm = document.getElementById('crop-form');
    const cropResult = document.getElementById('crop-result');

    if (!cropForm) return;

    // Backend handles crop recommendation via AI model

    cropForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = {
            N: parseFloat(document.getElementById('nitrogen').value),
            P: parseFloat(document.getElementById('phosphorus').value),
            K: parseFloat(document.getElementById('potassium').value),
            ph: parseFloat(document.getElementById('ph-level').value),
            rainfall: parseFloat(document.getElementById('rainfall').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value)
        };

        const validation = validateFormData(formData);
        if (!validation.isValid) {
            showNotification(validation.errors.join(', '), 'error');
            return;
        }

        const submitBtn = cropForm.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/crop-recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Recommendation failed');
            }

            const data = await response.json();
            const topThree = data.results.slice(0, 3);

            cropResult.innerHTML = `
                <h3><i class="fas fa-chart-line"></i> Top Crop Recommendations</h3>
                <div class="recommendation-grid">
                    ${topThree.map((crop, index) => `
                        <div class="recommendation-card" style="border-left: 4px solid ${index === 0 ? '#4CAF50' : index === 1 ? '#2196F3' : '#FF9800'};">
                            <div class="rank-badge" style="background: ${index === 0 ? '#4CAF50' : index === 1 ? '#2196F3' : '#FF9800'};">${index + 1}</div>
                            <h4>${crop.crop}</h4>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${crop.confidence * 100}%;"></div>
                            </div>
                            <span class="score-text">${(crop.confidence * 100).toFixed(1)}% Match</span>
                            <div class="suitability-badge ${crop.confidence > 0.8 ? 'excellent' : crop.confidence > 0.6 ? 'good' : 'moderate'}">
                                ${crop.confidence > 0.8 ? 'Excellent' : crop.confidence > 0.6 ? 'Good' : 'Moderate'}
                            </div>
                            <ul class="crop-details">
                                <li><span>Best season: ${getSeason(crop.crop)}</span></li>
                                <li><span>Sowing time: ${getSowingTime(crop.crop)}</span></li>
                                <li><span>Market demand: ${getMarketDemand(crop.crop)}</span></li>
                            </ul>
                        </div>
                    `).join('')}
                </div>
                <div class="analysis-summary">
                    <h4>Soil Analysis Summary</h4>
                    <p>Based on your soil parameters from the AI model, these crops are recommended for optimal yield.</p>
                </div>
            `;
            cropResult.classList.add('show');

            saveCropResult({
                recommended_crops: topThree.map(c => c.crop)
            });

        } catch (error) {
            showNotification(error.message, 'error');
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });
}

// ===== UTILITY FUNCTIONS =====
function resetImageUpload() {
    const imageUpload = document.getElementById('leaf-image');
    const imagePreview = document.getElementById('image-preview');
    const uploadArea = document.querySelector('.upload-area');

    if (imageUpload) imageUpload.value = '';
    if (imagePreview) {
        imagePreview.src = '';
        imagePreview.style.display = 'none';
    }
    if (uploadArea) uploadArea.style.display = 'block';
}

function validateImageFile(file) {
    if (file.size > 5 * 1024 * 1024) {
        showNotification('File size must be less than 5MB', 'error');
        return false;
    }

    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload JPG, PNG, or WEBP images only', 'error');
        return false;
    }

    return true;
}

function validateFormData(formData) {
    const errors = [];

    if (isNaN(formData.N) || formData.N < 0 || formData.N > 200) errors.push('Nitrogen must be 0-200');
    if (isNaN(formData.P) || formData.P < 0 || formData.P > 100) errors.push('Phosphorus must be 0-100');
    if (isNaN(formData.K) || formData.K < 0 || formData.K > 150) errors.push('Potassium must be 0-150');
    if (isNaN(formData.ph) || formData.ph < 4.0 || formData.ph > 9.0) errors.push('pH must be 4.0-9.0');
    if (isNaN(formData.temperature) || formData.temperature < -10 || formData.temperature > 50) errors.push('Temperature must be -10 to 50°C');
    if (isNaN(formData.humidity) || formData.humidity < 0 || formData.humidity > 100) errors.push('Humidity must be 0-100%');
    if (isNaN(formData.rainfall) || formData.rainfall < 0 || formData.rainfall > 5000) errors.push('Rainfall must be 0-5000mm');

    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

function showNotification(message, type = 'info') {
    alert(message);
}

function getSeason(crop) {
    const seasons = {
        'Rice': 'Kharif (June-September)',
        'Wheat': 'Rabi (October-March)',
        'Maize': 'Kharif (June-September)',
        'Cotton': 'Kharif (June-September)',
        'Sugarcane': 'Year-round',
        'Potato': 'Rabi (October-March)',
        'Tomato': 'Year-round (protected)',
        'Soybean': 'Kharif (June-September)',
        'Millet': 'Kharif (June-September)'
    };
    return seasons[crop] || 'Varies by region';
}

function getSowingTime(crop) {
    const times = {
        'Rice': 'June-July',
        'Wheat': 'November-December',
        'Maize': 'June-July',
        'Cotton': 'May-June',
        'Sugarcane': 'February-March',
        'Potato': 'October-November',
        'Tomato': 'March-April',
        'Soybean': 'June-July',
        'Millet': 'June-July'
    };
    return times[crop] || 'Consult local agriculture office';
}

function getMarketDemand(crop) {
    const demands = {
        'Rice': 'High',
        'Wheat': 'High',
        'Maize': 'Moderate-High',
        'Cotton': 'High',
        'Sugarcane': 'Moderate',
        'Potato': 'High',
        'Tomato': 'High',
        'Soybean': 'Moderate',
        'Millet': 'Moderate'
    };
    return demands[crop] || 'Moderate';
}

// ===== INITIALIZE ON DOM LOAD =====
document.addEventListener('DOMContentLoaded', function () {
    checkAuthStatus();

    // Login form handler
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            await login(username, password);
        });
    }

    // Signup form handler
    const signupForm = document.getElementById('signup-form');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirm = document.getElementById('confirm-password').value;
            if (password !== confirm) {
                showNotification('Passwords do not match', 'error');
                return;
            }
            await signup(username, email, password);
        });
    }

    initTabs();
    initDiseaseDetection();
    initCropRecommendation();
    // History is loaded by React
});