async function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
  
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
      window.location.href = 'users-profile.html';
    } else {
      alert('Email ou mot de passe incorrect');
    }
  }
  