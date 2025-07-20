async function logout() {
    const token = localStorage.getItem('access_token');
    
    if (token) {
        try {
            // Informer le serveur de la déconnexion
            await fetch('/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ token: token })
            });
        } catch (error) {
            console.error('Erreur lors de la déconnexion:', error);
        }
    }
    
    // Nettoyer le localStorage
    localStorage.clear();
    
    // Rediriger vers la page de connexion
    window.location.href = '/form-login.html';
}

// Ajouter un écouteur pour le bouton de déconnexion
document.addEventListener('DOMContentLoaded', function() {
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }
});