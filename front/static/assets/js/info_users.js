// Fonction pour obtenir le token
function getAuthToken() {
    return localStorage.getItem('access_token');
}

// Fonction pour faire des requêtes authentifiées
async function authenticatedFetch(url, options = {}) {
    const token = getAuthToken();
    
    if (!token) {
        window.location.href = '/form-login.html';
        return;
    }
    
    const defaultHeaders = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    };
    
    const requestOptions = {
        ...options,
        headers: {
            ...defaultHeaders,
            ...options.headers
        }
    };
    
    try {
        const response = await fetch(url, requestOptions);
        
        if (response.status === 401) {
            // Token expiré ou invalide
            localStorage.clear();
            window.location.href = '/form-login.html';
            return;
        }
        
        return response;
    } catch (error) {
        console.error('Erreur lors de la requête:', error);
        throw error;
    }
}

// Charger les informations utilisateur
async function loadUserInfo() {
    try {
        const response = await authenticatedFetch('/users/me');
        
        if (response && response.ok) {
            const user = await response.json();
            
            // Mettre à jour l'interface avec les infos utilisateur
            const nameElement = document.getElementById('user-name');
            const emailElement = document.getElementById('user-email');
            const avatarElement = document.getElementById('user-avatar');
            
            if (nameElement) nameElement.textContent = user.nom || 'Utilisateur';
            if (emailElement) emailElement.textContent = user.email;
            if (avatarElement && user.avatar_url) {
                avatarElement.src = user.avatar_url;
            }
            
            console.log('✅ Informations utilisateur chargées:', user);
        }
    } catch (error) {
        console.error('❌ Erreur chargement infos utilisateur:', error);
    }
}

// Charger les conversations
async function loadUserConversations() {
    try {
        const response = await authenticatedFetch('/user/conversations');
        
        if (response && response.ok) {
            const conversations = await response.json();
            console.log('✅ Conversations chargées:', conversations);
            
            // Afficher les conversations dans l'interface
            displayConversations(conversations);
        }
    } catch (error) {
        console.error('❌ Erreur chargement conversations:', error);
    }
}

// Afficher les conversations (exemple)
function displayConversations(conversations) {
    const conversationsContainer = document.getElementById('conversations-list');
    
    if (!conversationsContainer) return;
    
    if (conversations.length === 0) {
        conversationsContainer.innerHTML = '<p class="text-gray-500">Aucune conversation trouvée</p>';
        return;
    }
    
    conversationsContainer.innerHTML = conversations.map(conv => `
        <div class="bg-white rounded-lg shadow p-4 mb-4">
            <h3 class="font-semibold">${conv.category}</h3>
            <p class="text-sm text-gray-600">Démarré le ${new Date(conv.start_time).toLocaleDateString()}</p>
            <button onclick="viewConversation(${conv.id})" class="mt-2 bg-blue-500 text-white px-3 py-1 rounded text-sm">
                Voir détails
            </button>
        </div>
    `).join('');
}

// Fonction à appeler au chargement de la page
function initializePage() {
    // Vérifier si l'utilisateur est connecté
    const token = getAuthToken();
    if (!token) {
        window.location.href = '/form-login.html';
        return;
    }
    
    // Charger les données
    loadUserInfo();
    loadUserConversations();
}

// Initialiser quand la page est chargée
document.addEventListener('DOMContentLoaded', initializePage);