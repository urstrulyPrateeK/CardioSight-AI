// Import Firebase (handled via CDN in base.html)

document.addEventListener('DOMContentLoaded', () => {
    // --- FIREBASE INIT ---
    // The config is injected by the server into a global variable in base.html
    if (typeof firebaseConfig !== 'undefined') {
        firebase.initializeApp(firebaseConfig);
        const auth = firebase.auth();
        const googleProvider = new firebase.auth.GoogleAuthProvider();

        // --- AUTH STATE LISTENER ---
        auth.onAuthStateChanged(async (user) => {
            const navLinks = document.querySelector('.nav-links');
            if (user) {
                // User is signed in.
                console.log("Logged in as:", user.email);

                // Redirect if on login/register/root page
                const path = window.location.pathname;
                if (path === '/' || path === '/login' || path === '/register') {
                    window.location.href = '/home';
                }

                // Update Navbar
                const loginLink = document.querySelector('a[href="/login"]');
                if (loginLink) loginLink.style.display = 'none';

                // Check if logout button exists, if not add it
                if (!document.getElementById('logout-btn')) {
                    const logoutBtn = document.createElement('a');
                    logoutBtn.id = 'logout-btn';
                    logoutBtn.href = "#";
                    logoutBtn.textContent = "Logout (" + user.displayName?.split(' ')[0] + ")";
                    logoutBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        auth.signOut().then(() => window.location.href = '/');
                    });
                    navLinks.appendChild(logoutBtn);
                }

                // Inject Token into Forms
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

                    // Auto-fill name/email
                    if (document.querySelector('input[name="name"]'))
                        document.querySelector('input[name="name"]').value = user.displayName;
                    if (document.querySelector('input[name="email"]'))
                        document.querySelector('input[name="email"]').value = user.email;
                }

            } else {
                // User is signed out.
                console.log("User logged out");
                const logoutBtn = document.getElementById('logout-btn');
                if (logoutBtn) logoutBtn.remove();

                const loginLink = document.querySelector('a[href="/login"]');
                if (loginLink) loginLink.style.display = 'inline-block';
            }
        });

        // --- GOOGLE LOGIN ---
        const googleBtn = document.getElementById('google-login-btn');
        if (googleBtn) {
            googleBtn.addEventListener('click', () => {
                auth.signInWithPopup(googleProvider).then((result) => {
                    // Success
                    window.location.href = '/home';
                }).catch((error) => {
                    alert(error.message);
                });
            });
        }

        // --- EMAIL LOGIN ---
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;

                auth.signInWithEmailAndPassword(email, password)
                    .then(() => {
                        window.location.href = '/home';
                    })
                    .catch((error) => {
                        document.getElementById('auth-message').textContent = error.message;
                        document.getElementById('auth-message').style.color = 'red';
                    });
            });
        }

        // --- REGISTRATION ---
        const regForm = document.getElementById('register-form');
        if (regForm) {
            regForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('reg-email').value;
                const password = document.getElementById('reg-password').value;
                const name = document.getElementById('reg-name').value;
                const phone = document.getElementById('reg-phone').value;
                const dob = document.getElementById('reg-dob').value;

                auth.createUserWithEmailAndPassword(email, password)
                    .then((userCredential) => {
                        const user = userCredential.user;
                        return user.updateProfile({ displayName: name })
                            .then(() => {
                                // Save extra details to backend
                                return fetch('/save_profile', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        uid: user.uid,
                                        email: email,
                                        fullName: name,
                                        phone: phone,
                                        dob: dob
                                    })
                                });
                            });
                    })
                    .then(() => {
                        alert("Account created successfully!");
                        window.location.href = '/home';
                    })
                    .catch((error) => {
                        document.getElementById('auth-message').textContent = error.message;
                        document.getElementById('auth-message').style.color = 'red';
                    });
            });
        }
    }

    // --- OCR LOGIC (Existing) ---
    // (Kept separate to avoid breaking existing functionality, assuming the previous main.js logic is merged or this replaces specific parts.
    // For safety, I included the previous dropzone logic below)

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
                    ocrStatus.textContent = "Success!";
                    populateForm(data.data);
                } else {
                    ocrStatus.textContent = "Failed.";
                }
            } catch (e) { console.error(e); }
        });
    }

    function populateForm(data) {
        console.log("OCR extracted data:", data);
        const setField = (name, val) => {
            const el = document.querySelector(`[name="${name}"]`);
            if (el && val !== undefined && val !== null) {
                el.value = val;
            }
        };
        if (data.name) setField('name', data.name);
        if (data.age) setField('age', data.age);
        if (data.gender) setField('gender', data.gender);
        if (data.cp) setField('cp', data.cp);
        if (data.trestbps) setField('trestbps', data.trestbps);
        if (data.chol) setField('chol', data.chol);
        if (data.fbs) setField('fbs', data.fbs);
        if (data.restecg) setField('restecg', data.restecg);
        if (data.thalach) setField('thalach', data.thalach);
        if (data.exang) setField('exang', data.exang);
        if (data.oldpeak) setField('oldpeak', data.oldpeak);
        if (data.slope) setField('slope', data.slope);
        if (data.ca) setField('ca', data.ca);
        if (data.thal) setField('thal', data.thal);
    }
});
