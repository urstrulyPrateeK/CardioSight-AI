document.addEventListener('DOMContentLoaded', () => {

    // --- NAVBAR SCROLL EFFECT ---
    const navbar = document.getElementById('main-navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            navbar.classList.toggle('scrolled', window.scrollY > 10);
        });
    }

    // --- FIREBASE INIT ---
    if (typeof firebaseConfig !== 'undefined' && firebaseConfig.apiKey) {
        firebase.initializeApp(firebaseConfig);
        const auth = firebase.auth();
        const googleProvider = new firebase.auth.GoogleAuthProvider();

        auth.onAuthStateChanged(async (user) => {
            const navLinks = document.getElementById('nav-links');
            if (user) {
                const path = window.location.pathname;
                if (path === '/' || path === '/login' || path === '/register') {
                    window.location.href = '/home';
                }

                const loginLink = document.getElementById('login-link');
                if (loginLink) loginLink.style.display = 'none';

                if (!document.getElementById('logout-btn')) {
                    const logoutBtn = document.createElement('a');
                    logoutBtn.id = 'logout-btn';
                    logoutBtn.href = '#';
                    logoutBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> ' + (user.displayName ? user.displayName.split(' ')[0] : 'Logout');
                    logoutBtn.style.cssText = 'background:var(--bg-card-alt);color:var(--text-primary);padding:8px 18px;border-radius:8px;font-size:0.85rem;border:1px solid var(--border);';
                    logoutBtn.addEventListener('click', (e) => {
                        e.preventDefault();
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

                    const nameField = document.querySelector('input[name="name"]');
                    const emailField = document.querySelector('input[name="email"]');
                    if (nameField && !nameField.value) nameField.value = user.displayName || '';
                    if (emailField && !emailField.value) emailField.value = user.email || '';
                }
            } else {
                const logoutBtn = document.getElementById('logout-btn');
                if (logoutBtn) logoutBtn.remove();
                const loginLink = document.getElementById('login-link');
                if (loginLink) loginLink.style.display = 'inline-flex';
            }
        });

        // Google Login
        const googleBtn = document.getElementById('google-login-btn');
        if (googleBtn) {
            googleBtn.addEventListener('click', () => {
                auth.signInWithPopup(googleProvider)
                    .then(() => window.location.href = '/home')
                    .catch((error) => showAuthError(error.message));
            });
        }

        // Email Login
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                const btn = loginForm.querySelector('button[type="submit"]');
                btn.innerHTML = '<span class="spinner"></span> Signing in...';
                btn.disabled = true;

                auth.signInWithEmailAndPassword(email, password)
                    .then(() => window.location.href = '/home')
                    .catch((error) => {
                        showAuthError(error.message);
                        btn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In';
                        btn.disabled = false;
                    });
            });
        }

        // Registration
        const regForm = document.getElementById('register-form');
        if (regForm) {
            regForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('reg-email').value;
                const password = document.getElementById('reg-password').value;
                const name = document.getElementById('reg-name').value;
                const phone = document.getElementById('reg-phone').value;
                const dob = document.getElementById('reg-dob').value;
                const btn = regForm.querySelector('button[type="submit"]');
                btn.innerHTML = '<span class="spinner"></span> Creating...';
                btn.disabled = true;

                auth.createUserWithEmailAndPassword(email, password)
                    .then((userCredential) => {
                        return userCredential.user.updateProfile({ displayName: name })
                            .then(() => fetch('/save_profile', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ uid: userCredential.user.uid, email, fullName: name, phone, dob })
                            }));
                    })
                    .then(() => window.location.href = '/home')
                    .catch((error) => {
                        showAuthError(error.message);
                        btn.innerHTML = '<i class="fas fa-user-plus"></i> Create Account';
                        btn.disabled = false;
                    });
            });
        }
    }

    function showAuthError(msg) {
        const el = document.getElementById('auth-message');
        if (el) {
            el.textContent = msg;
            el.style.cssText = 'color:#FF8A80;background:rgba(220,20,60,0.1);padding:12px;border-radius:8px;border:1px solid rgba(220,20,60,0.2);';
        }
    }

    // --- OCR LOGIC ---
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        const fileInput = document.getElementById('file-input');
        const extractBtn = document.getElementById('extract-btn');

        dropZone.addEventListener('click', (e) => {
            if (e.target !== extractBtn && !extractBtn.contains(e.target)) fileInput.click();
        });
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFile(); }
        });
        fileInput.addEventListener('change', handleFile);

        function handleFile() {
            if (fileInput.files.length > 0) {
                document.getElementById('file-name').textContent = fileInput.files[0].name;
                extractBtn.style.display = 'inline-flex';
            }
        }

        extractBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const ocrStatus = document.getElementById('ocr-status');
            extractBtn.innerHTML = '<span class="spinner"></span> Scanning...';
            extractBtn.disabled = true;
            ocrStatus.textContent = '';
            const fd = new FormData();
            fd.append('file', fileInput.files[0]);
            try {
                const res = await fetch('/extract_from_report', { method: 'POST', body: fd });
                const data = await res.json();
                if (data.success) {
                    ocrStatus.innerHTML = '<span style="color:var(--success);font-weight:600;"><i class="fas fa-check-circle"></i> Data extracted successfully!</span>';
                    populateForm(data.data);
                } else {
                    ocrStatus.innerHTML = '<span style="color:var(--danger);"><i class="fas fa-times-circle"></i> Extraction failed</span>';
                }
            } catch (err) {
                ocrStatus.innerHTML = '<span style="color:var(--danger);"><i class="fas fa-times-circle"></i> Error processing file</span>';
            }
            extractBtn.innerHTML = '<i class="fas fa-magic"></i> Extract Data';
            extractBtn.disabled = false;
        });
    }

    function populateForm(data) {
        const setField = (name, val) => {
            const el = document.querySelector(`[name="${name}"]`);
            if (el && val !== undefined && val !== null) el.value = val;
        };
        Object.keys(data).forEach(key => setField(key, data[key]));
    }

    // --- FORM SUBMIT LOADING STATE ---
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', () => {
            const btn = document.getElementById('submit-btn');
            if (btn) {
                btn.innerHTML = '<span class="spinner"></span> Analyzing...';
                btn.disabled = true;
            }
        });
    }
});
