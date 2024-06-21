document.addEventListener('DOMContentLoaded', async () => {
    const token = localStorage.getItem('accessToken');
    
    if (token) {
        try {
            const response = await fetch('http://127.0.0.1:8000/users/me', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const userData = await response.json();
                
                // Mettre à jour tous les éléments avec la classe info-name
                const nameElements = document.querySelectorAll('.info-name');
                nameElements.forEach(element => {
                    element.value = userData.nom || 'Nom';
                    element.innerText = userData.nom || 'Nom';
                });

                // Mettre à jour tous les éléments avec la classe info-email
                const emailElements = document.querySelectorAll('.info-email');
                emailElements.forEach(element => {
                    element.value = userData.email || 'Email';
                    element.innerText = userData.email || 'Email';
                });

                // Mettre à jour tous les éléments avec la classe info-date-naissance
                const dateNaissanceElements = document.querySelectorAll('.info-date-naissance');
                dateNaissanceElements.forEach(element => {
                    element.value = userData.date_naissance || '';
                    element.innerText = userData.date_naissance || '';
                });

                // Mettre à jour tous les éléments avec la classe info-location
                const locationElements = document.querySelectorAll('.info-location');
                locationElements.forEach(element => {
                    element.value = userData.location || '';
                    element.innerText = userData.location || '';
                });

                // Mettre à jour tous les éléments avec la classe info-about
                const aboutElements = document.querySelectorAll('.info-about');
                aboutElements.forEach(element => {
                    element.value = userData.about || '';
                    element.innerText = userData.about || '';
                });
            } else {
                console.error('Failed to fetch user data:', response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    } else {
        console.warn('No access token found in localStorage.');
    }
});
