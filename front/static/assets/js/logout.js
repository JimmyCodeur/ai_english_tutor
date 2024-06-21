async function logout() {
    try {
        const accessToken = localStorage.getItem('accessToken');
        const response = await fetch('http://127.0.0.1:8000/logout', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ token: accessToken }) 
        });

        if (response.ok) {
            localStorage.removeItem('accessToken');
            alert('Vous êtes maintenant déconnecté.');
            window.location.href = 'form-login.html';
        } else {
            const errorMessage = await response.text();
            alert(`Erreur lors de la déconnexion : ${errorMessage}`);
        }
    } catch (error) {
        console.error('Erreur lors de la requête de déconnexion :', error);
        alert('Erreur réseau lors de la déconnexion. Veuillez réessayer.');
    }
}
