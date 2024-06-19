async function logout() {
    try {
        const accessToken = localStorage.getItem('accessToken'); // Récupérer le token depuis le localStorage
        const response = await fetch('http://127.0.0.1:8000/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ token: accessToken }) // Envoyer le token dans le corps de la requête JSON
        });

        if (response.ok) {
            // Déconnexion réussie
            localStorage.removeItem('accessToken'); // Supprimez le jeton d'accès du stockage local
            alert('Vous êtes maintenant déconnecté.');
            window.location.href = 'form-login.html'; // Redirigez vers la page de connexion après la déconnexion
        } else {
            const errorMessage = await response.text();
            alert(`Erreur lors de la déconnexion : ${errorMessage}`);
        }
    } catch (error) {
        console.error('Erreur lors de la requête de déconnexion :', error);
        alert('Erreur réseau lors de la déconnexion. Veuillez réessayer.');
    }
}

// Ajout d'un gestionnaire d'événements pour le lien de déconnexion
const logoutLink = document.getElementById('logout-link');
if (logoutLink) {
    logoutLink.addEventListener('click', function(event) {
        event.preventDefault(); // Empêche le comportement par défaut du lien
        logout(); // Appel de la fonction de déconnexion
    });
}
