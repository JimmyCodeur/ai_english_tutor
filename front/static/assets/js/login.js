async function login() {
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  try {
      const response = await fetch('http://127.0.0.1:8000/login/', {
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
          const accessToken = data.access_token;
  
          // Stockage du token dans LocalStorage
          localStorage.setItem('accessToken', accessToken);
          localStorage.setItem('userEmail', data.email);
          localStorage.setItem('nom', data.nom);
  
          // Redirection vers la page des profils utilisateur après la connexion réussie
          window.location.href = 'home.html';
      } else {
          const errorMessage = await response.text();
          alert(`Email ou mot de passe incorrect : ${errorMessage}`);
      }
  } catch (error) {
      console.error('Erreur lors de la requête de connexion :', error);
      alert('Erreur réseau lors de la connexion. Veuillez réessayer.');
  }
}