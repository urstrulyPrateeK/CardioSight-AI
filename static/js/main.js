// Import Firebase (handled via CDN in base.html)

document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    const cpMapping = {
        'Typical Angina': '0',
        'Atypical Angina': '1',
        'Non-anginal pain': '2',
        'Non-anginal Pain': '2',
        'Asymptomatic': '3'
    };
    const slopeMapping = {
        'Upsloping': '0',
        'Flat': '1',
        'Downsloping': '2'
    };
    const thalMapping = {
        'Normal': '1',
        'Fixed defect': '2',
        'Fixed Defect': '2',
        'Reversible defect': '3',
        'Reversable Defect': '3'
    };

    function formatDate(isoValue) {
        if (!isoValue) return 'N/A';
        const dateObj = new Date(isoValue);
        if (Number.isNaN(dateObj.getTime())) return 'N/A';
        return dateObj.toLocaleString();
    }

    function renderHistoryRows(records) {
        const tbody = document.getElementById('history-table-body');
        if (!tbody) return;

        if (!records.length) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:#666;">No previous predictions found.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map((item) => {
            const patient = item.patient_input || {};
            const level = item.prediction_result || 'Unknown';
            const color = level === 'High' ? '#ff4b2b' : (level === 'Moderate' ? '#ff9800' : '#00c853');
            return `
                <tr>
                    <td>${patient.name || 'N/A'}</td>
                    <td>${patient.age ?? 'N/A'}</td>
                    <td>${item.probability_score ?? 'N/A'}%</td>
                    <td><span style="font-weight:600; color:${color};">${level}</span></td>
                    <td>${formatDate(item.timestamp)}</td>
                </tr>
            `;
        }).join('');
    }

    function renderDashboardStats(records) {
        const total = records.length;
        const highRisk = records.filter((item) => item.prediction_result === 'High').length;
        const lowRisk = records.filter((item) => item.prediction_result === 'Low').length;

        const totalEl = document.getElementById('total-screenings');
        const highEl = document.getElementById('high-risk-cases');
        const lowEl = document.getElementById('low-risk-cases');
        if (totalEl) totalEl.textContent = String(total);
        if (highEl) highEl.textContent = total ? `${Math.round((highRisk / total) * 100)}%` : '0%';
        if (lowEl) lowEl.textContent = total ? `${Math.round((lowRisk / total) * 100)}%` : '0%';

        const genderMap = { Male: 0, Female: 0 };
        const ageBuckets = { '20-30': 0, '31-40': 0, '41-50': 0, '51-60': 0, '60+': 0 };

        records.forEach((item) => {
            const patient = item.patient_input || {};
            const sex = Number(patient.sex);
            if (sex === 1) genderMap.Male += 1;
            if (sex === 0) genderMap.Female += 1;

            const age = Number(patient.age);
            if (!Number.isNaN(age)) {
                if (age <= 30) ageBuckets['20-30'] += 1;
                else if (age <= 40) ageBuckets['31-40'] += 1;
                else if (age <= 50) ageBuckets['41-50'] += 1;
                else if (age <= 60) ageBuckets['51-60'] += 1;
                else ageBuckets['60+'] += 1;
            }
        });

        const genderCanvas = document.getElementById('genderChart');
        if (genderCanvas) {
            if (window.genderChartInstance) window.genderChartInstance.destroy();
            window.genderChartInstance = new Chart(genderCanvas, {
                type: 'pie',
                data: {
                    labels: ['Male', 'Female'],
                    datasets: [{ data: [genderMap.Male, genderMap.Female], backgroundColor: ['#2575fc', '#ff4b2b'] }]
                }
            });
        }

        const ageCanvas = document.getElementById('ageChart');
        if (ageCanvas) {
            if (window.ageChartInstance) window.ageChartInstance.destroy();
            window.ageChartInstance = new Chart(ageCanvas, {
                type: 'bar',
                data: {
                    labels: Object.keys(ageBuckets),
                    datasets: [{ label: 'Patients', data: Object.values(ageBuckets), backgroundColor: '#2575fc' }]
                }
            });
        }
    }

    async function loadPredictionHistory(token) {
        const stateEl = document.getElementById('history-status');
        if (stateEl) stateEl.textContent = 'Loading your prediction history...';

        try {
            const res = await fetch('/api/predictions/history', {
                headers: { Authorization: `Bearer ${token}` }
            });
            const payload = await res.json();

            if (!res.ok || !payload.success) {
                const message = payload.error || 'Could not load history.';
                if (stateEl) stateEl.textContent = message;
                renderHistoryRows([]);
                return;
            }

            const records = payload.history || [];
            renderDashboardStats(records);
            renderHistoryRows(records);
            if (stateEl) stateEl.textContent = records.length ? '' : 'No records yet. Run a prediction to populate history.';
        } catch (error) {
            if (stateEl) stateEl.textContent = 'Failed to load history due to a network error.';
            renderHistoryRows([]);
        }
    }

    // --- FIREBASE INIT ---
    const showAuthMessage = (msg, isError = true) => {
        const el = document.getElementById('auth-message');
        if (!el) return;
        el.textContent = msg || '';
        el.style.color = isError ? 'red' : 'green';
    };

    const handleAuthError = (error) => {
        const code = error?.code || '';
        if (code === 'auth/operation-not-allowed') {
            return 'Email/Password sign-in is disabled in Firebase. Enable the Email/Password provider in Firebase Console > Authentication > Sign-in method.';
        }
        if (code === 'auth/unauthorized-domain') {
            return 'Google Sign-In failed: add your local host (for example 127.0.0.1 or localhost) to Firebase Authorized Domains.';
        }
        if (code === 'auth/popup-closed-by-user') {
            return 'Google login popup was closed before sign-in completed.';
        }
        if (code === 'auth/popup-blocked') {
            return 'Popup blocked by browser. Please allow popups and try again.';
        }
        return error?.message || 'Authentication failed.';
    };

    const hasFirebaseClientConfig = () => {
        if (typeof firebaseConfig === 'undefined' || !firebaseConfig) return false;
        const required = ['apiKey', 'authDomain', 'projectId', 'appId'];
        return required.every((key) => typeof firebaseConfig[key] === 'string' && firebaseConfig[key].trim() !== '');
    };

    if (hasFirebaseClientConfig()) {
        if (!firebase.apps.length) {
            firebase.initializeApp(firebaseConfig);
        }

        const auth = firebase.auth();
        const googleProvider = new firebase.auth.GoogleAuthProvider();

        const syncBackendSession = async (user, forceRefresh = false) => {
            const idToken = await user.getIdToken(forceRefresh);
            const res = await fetch('/auth/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ idToken })
            });
            const payload = await res.json();
            if (!res.ok || !payload.success) {
                throw new Error(payload.error || 'Backend session verification failed.');
            }
            return idToken;
        };

        // --- AUTH STATE LISTENER ---
        auth.onAuthStateChanged(async (user) => {
            const navLinks = document.querySelector('.nav-links');
            if (user) {
                const loginLink = document.querySelector('a[href="/login"]');
                if (loginLink) loginLink.style.display = 'none';

                try {
                    await syncBackendSession(user, false);
                } catch (error) {
                    showAuthMessage(error.message || 'Could not verify backend session.');
                }

                if (!document.getElementById('logout-btn') && navLinks) {
                    const logoutBtn = document.createElement('a');
                    logoutBtn.id = 'logout-btn';
                    logoutBtn.href = "#";
                    logoutBtn.textContent = "Logout (" + (user.displayName?.split(' ')[0] || 'User') + ")";
                    logoutBtn.addEventListener('click', async (e) => {
                        e.preventDefault();
                        try {
                            await fetch('/auth/logout', { method: 'POST' });
                        } catch (_) { }
                        auth.signOut().then(() => window.location.href = '/');
                    });
                    navLinks.appendChild(logoutBtn);
                }

                const predictForm = document.getElementById('predict-form');
                if (predictForm) {
                    const token = await user.getIdToken();
                    let tokenInput = document.getElementById('id_token');
                    if (!tokenInput) {
                        tokenInput = document.createElement('input');
                        tokenInput.type = 'hidden';
                        tokenInput.name = 'id_token';
                        tokenInput.id = 'id_token';
                        predictForm.appendChild(tokenInput);
                    }
                    tokenInput.value = token;

                    if (!predictForm.dataset.tokenHooked) {
                        predictForm.addEventListener('submit', async () => {
                            try {
                                tokenInput.value = await user.getIdToken(true);
                            } catch (_) { }
                        });
                        predictForm.dataset.tokenHooked = '1';
                    }

                    const nameInput = document.querySelector('input[name="name"]');
                    const emailInput = document.querySelector('input[name="email"]');
                    if (nameInput && !nameInput.value) nameInput.value = user.displayName || '';
                    if (emailInput && !emailInput.value) emailInput.value = user.email || '';
                }

                if (path === '/dashboard') {
                    const token = await user.getIdToken();
                    loadPredictionHistory(token);
                }

                if (path === '/' || path === '/login' || path === '/register') {
                    window.location.href = '/home';
                }
            } else {
                const logoutBtn = document.getElementById('logout-btn');
                if (logoutBtn) logoutBtn.remove();
                const loginLink = document.querySelector('a[href="/login"]');
                if (loginLink) loginLink.style.display = 'inline-block';

                if (path === '/dashboard') {
                    const stateEl = document.getElementById('history-status');
                    if (stateEl) stateEl.textContent = 'Please login to view prediction history.';
                    renderHistoryRows([]);
                    renderDashboardStats([]);
                }
            }
        });

        // --- GOOGLE LOGIN ---
        const googleBtn = document.getElementById('google-login-btn');
        if (googleBtn) {
            googleBtn.addEventListener('click', async () => {
                showAuthMessage('', false);
                try {
                    const result = await auth.signInWithPopup(googleProvider);
                    await syncBackendSession(result.user, true);
                    window.location.href = '/home';
                } catch (error) {
                    showAuthMessage(handleAuthError(error));
                }
            });
        }

        // --- EMAIL LOGIN ---
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;

                try {
                    await auth.signInWithEmailAndPassword(email, password);
                    if (auth.currentUser) {
                        await syncBackendSession(auth.currentUser, true);
                    }
                    window.location.href = '/home';
                } catch (error) {
                    showAuthMessage(handleAuthError(error));
                }
            });
        }

        // --- REGISTRATION ---
        const regForm = document.getElementById('register-form');
        if (regForm) {
            regForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const email = document.getElementById('reg-email').value;
                const password = document.getElementById('reg-password').value;
                const name = document.getElementById('reg-name').value;
                const phone = document.getElementById('reg-phone').value;
                const dob = document.getElementById('reg-dob').value;

                try {
                    const userCredential = await auth.createUserWithEmailAndPassword(email, password);
                    const user = userCredential.user;
                    await user.updateProfile({ displayName: name });
                    const idToken = await user.getIdToken(true);
                    await fetch('/save_profile', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            idToken,
                            uid: user.uid,
                            email,
                            fullName: name,
                            phone,
                            dob
                        })
                    });
                    await syncBackendSession(user, false);
                    alert("Account created successfully!");
                    window.location.href = '/home';
                } catch (error) {
                    showAuthMessage(handleAuthError(error));
                }
            });
        }
    } else {
        const googleBtn = document.getElementById('google-login-btn');
        if (googleBtn) googleBtn.disabled = true;
        showAuthMessage('Firebase web config is missing. Add FIREBASE_* values in your .env and restart the app.');
    }

    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        // ... (Previous OCR code logic) ...
        const fileInput = document.getElementById('file-input');
        const extractBtn = document.getElementById('extract-btn');
        // Re-implement listeners for seamlessness
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFile(); }
        });
        fileInput.addEventListener('change', handleFile);

        function handleFile() {
            if (fileInput.files.length > 0) {
                document.getElementById('file-name').textContent = `Selected: ${fileInput.files[0].name}`;
                extractBtn.style.display = 'inline-block';
            }
        }

        extractBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const ocrStatus = document.getElementById('ocr-status');
            ocrStatus.textContent = "Scanning...";
            const fd = new FormData();
            fd.append('file', fileInput.files[0]);
            try {
                const res = await fetch('/extract_from_report', { method: 'POST', body: fd });
                const data = await res.json();
                if (data.success) {
                    const missing = Array.isArray(data.missing_fields) ? data.missing_fields.length : 0;
                    ocrStatus.textContent = missing
                        ? `Success! Extracted available fields (${missing} still missing).`
                        : "Success!";
                    populateForm(data.data);
                } else {
                    ocrStatus.textContent = data.message || "Failed.";
                }
            } catch (e) { console.error(e); }
        });
    }

    function populateForm(data) {
        const setField = (name, val) => {
            const el = document.querySelector(`[name="${name}"]`);
            if (el && val !== undefined && val !== null) {
                el.value = val;
            }
        };
        const mapSelectValue = (value, mapping) => {
            const strVal = String(value ?? '').trim();
            if (strVal === '') return strVal;
            return mapping[strVal] ?? strVal;
        };

        if (data.name !== undefined && data.name !== null) setField('name', data.name);
        if (data.age !== undefined && data.age !== null) setField('age', data.age);
        if (data.gender !== undefined && data.gender !== null) setField('gender', data.gender);
        if (data.cp !== undefined && data.cp !== null) setField('cp', mapSelectValue(data.cp, cpMapping));
        if (data.trestbps !== undefined && data.trestbps !== null) setField('trestbps', data.trestbps);
        if (data.chol !== undefined && data.chol !== null) setField('chol', data.chol);
        if (data.fbs !== undefined && data.fbs !== null) setField('fbs', data.fbs);
        if (data.restecg !== undefined && data.restecg !== null) setField('restecg', data.restecg);
        if (data.thalach !== undefined && data.thalach !== null) setField('thalach', data.thalach);
        if (data.exang !== undefined && data.exang !== null) setField('exang', data.exang);
        if (data.oldpeak !== undefined && data.oldpeak !== null) setField('oldpeak', data.oldpeak);
        if (data.slope !== undefined && data.slope !== null) setField('slope', mapSelectValue(data.slope, slopeMapping));
        if (data.ca !== undefined && data.ca !== null) setField('ca', data.ca);
        if (data.thal !== undefined && data.thal !== null) setField('thal', mapSelectValue(data.thal, thalMapping));
    }
});
