document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('register-form');
    if (form) {
        form.addEventListener('submit', register);
    } else {
        console.error("Form not found!");
    }
});

async function register(event) {
    event.preventDefault();  // Empêche le rechargement de la page

    const usernameElement = document.getElementById('username');
    const emailElement = document.getElementById('email');
    const passwordElement = document.getElementById('password');
    const dateNaissanceElement = document.getElementById('date_naissance');
    const termsAccepted = document.getElementById('consent').checked;

    // Vérification de l'acceptation des termes
    if (!termsAccepted) {
        alert('Veuillez accepter les termes et conditions.');
        return;
    }

    // Vérification que tous les éléments sont trouvés avant d'accéder à leurs valeurs
    if (!usernameElement || !emailElement || !passwordElement || !dateNaissanceElement) {
        console.error('Certains éléments du formulaire ne sont pas trouvés.');
        alert('Erreur lors de la soumission du formulaire. Veuillez réessayer.');
        return;
    }

    // Récupération des valeurs des champs
    const username = usernameElement.value;
    const email = emailElement.value;
    const password = passwordElement.value;
    const date_naissance = dateNaissanceElement.value;

    // Préparation des données pour l'envoi
    const formData = new URLSearchParams();
    formData.append('email', email);
    formData.append('nom', username);
    formData.append('date_naissance', date_naissance);
    formData.append('password', password);
    formData.append('consent', termsAccepted);

    try {
        const response = await fetch('http://127.0.0.1:8000/users/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });

        if (response.ok) {
            alert('Enregistrement réussi ! Vous pouvez maintenant vous connecter.');
            window.location.href = 'form-login.html';
        } else {
            const errorMessage = await response.text();
            alert(`Erreur lors de l'enregistrement : ${errorMessage}`);
        }
    } catch (error) {
        console.error('Erreur lors de la requête :', error);
        alert('Erreur réseau lors de l\'enregistrement. Veuillez réessayer.');
    }
}