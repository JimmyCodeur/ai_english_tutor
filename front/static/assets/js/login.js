async function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    try {
        // ✅ URL RELATIVE au lieu de 127.0.0.1
        const response = await fetch('/login/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                email: email,
                password: password
            })
        });

        if (response.ok) {
            const data = await response.json();
            
            // ✅ STOCKAGE COHÉRENT
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user_id', data.user_id);
            localStorage.setItem('userEmail', data.email);
            localStorage.setItem('nom', data.nom);

            console.log('✅ Connexion réussie:', data);
            
            // Redirection
            window.location.href = '/home.html';
        } else {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || 'Erreur de connexion';
            } catch {
                errorMessage = await response.text();
            }
            
            alert(`Email ou mot de passe incorrect : ${errorMessage}`);
        }
    } catch (error) {
        console.error('❌ Erreur réseau:', error);
        alert('Erreur réseau lors de la connexion.');
    }
}