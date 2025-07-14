// Avatar Manager - Gestion des avatars utilisateur
class AvatarManager {
    constructor() {
        this.defaultAvatars = [
            '/static/assets/images/avatars/default-1.png',
            '/static/assets/images/avatars/default-2.png',
            '/static/assets/images/avatars/default-3.png',
            '/static/assets/images/avatars/default-4.png',
            '/static/assets/images/avatars/default-5.png'
        ];
        this.fallbackAvatar = '/static/assets/images/avatars/placeholder.png';
    }

    // Générer un avatar par défaut basé sur l'ID utilisateur
    getDefaultAvatar(userId) {
        if (!userId) return this.fallbackAvatar;
        const index = userId % this.defaultAvatars.length;
        return this.defaultAvatars[index];
    }

    // Générer un avatar avec initiales
    generateInitialsAvatar(name, userId) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const size = 200;
        
        canvas.width = size;
        canvas.height = size;
        
        // Couleurs basées sur l'ID utilisateur
        const colors = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c',
            '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
            '#fa709a', '#fee140', '#a8edea', '#fed6e3'
        ];
        
        const colorIndex = userId % colors.length;
        const bgColor = colors[colorIndex];
        
        // Fond circulaire
        ctx.fillStyle = bgColor;
        ctx.beginPath();
        ctx.arc(size/2, size/2, size/2, 0, Math.PI * 2);
        ctx.fill();
        
        // Texte (initiales)
        const initials = this.getInitials(name);
        ctx.fillStyle = 'white';
        ctx.font = `bold ${size * 0.4}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(initials, size/2, size/2);
        
        return canvas.toDataURL();
    }

    // Extraire les initiales d'un nom
    getInitials(name) {
        if (!name) return 'U';
        const parts = name.trim().split(' ');
        if (parts.length === 1) {
            return parts[0].charAt(0).toUpperCase();
        }
        return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
    }

    // Mettre à jour tous les avatars dans la page
    updateAllAvatars(avatarUrl, userName, userId) {
        const avatarSelectors = [
            '#header-avatar',
            '#menu-avatar', 
            '#profile-main-avatar',
            '.user-avatar'
        ];

        let finalAvatarUrl = avatarUrl;
        
        // Si pas d'avatar personnalisé, utiliser les initiales
        if (!avatarUrl) {
            finalAvatarUrl = this.generateInitialsAvatar(userName, userId);
        }

        avatarSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (element) {
                    element.src = finalAvatarUrl;
                    element.onerror = () => {
                        element.src = this.generateInitialsAvatar(userName, userId);
                    };
                }
            });
        });
    }

    // Upload d'avatar
    async uploadAvatar(file) {
        try {
            const formData = new FormData();
            formData.append('avatar', file);
            
            const accessToken = localStorage.getItem('accessToken');
            const response = await fetch('/user/upload-avatar', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`
                },
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                return result;
            } else {
                throw new Error('Erreur lors de l\'upload');
            }
        } catch (error) {
            console.error('Erreur upload avatar:', error);
            throw error;
        }
    }
}

// Instance globale
window.avatarManager = new AvatarManager();