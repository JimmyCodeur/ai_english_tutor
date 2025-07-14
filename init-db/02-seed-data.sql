-- init-db/02-seed-data.sql
-- Données de test pour TalkAI

-- Insérer un utilisateur de test (mot de passe: "password")
INSERT INTO users (email, hashed_password, nom, date_naissance, consent, role, bio, country) 
VALUES (
    'test@talkai.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewflfQT/c0R5c5rK',
    'Marc Testeur', 
    '1990-05-15', 
    true, 
    'utilisateur',
    'Utilisateur de test pour TalkAI - J''apprends l''anglais avec passion !',
    'France'
) ON CONFLICT (email) DO NOTHING;

-- Insérer un admin de test (mot de passe: "admin123")
INSERT INTO users (email, hashed_password, nom, date_naissance, consent, role, bio, country) 
VALUES (
    'admin@talkai.com', 
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi',
    'Admin TalkAI', 
    '1985-03-20', 
    true, 
    'admin',
    'Administrateur principal de la plateforme TalkAI',
    'France'
) ON CONFLICT (email) DO NOTHING;

-- Insérer votre compte existant avec le bon hash
INSERT INTO users (email, hashed_password, nom, date_naissance, consent, role, bio, country) 
VALUES (
    'marc@gmail.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewflfQT/c0R5c5rK',
    'Marc', 
    '1990-01-01', 
    true, 
    'utilisateur',
    'Créateur et utilisateur principal de TalkAI',
    'France'
) ON CONFLICT (email) DO UPDATE SET
    hashed_password = EXCLUDED.hashed_password,
    nom = EXCLUDED.nom,
    bio = EXCLUDED.bio,
    country = EXCLUDED.country,
    role = EXCLUDED.role;

-- Afficher les utilisateurs créés
DO $$
DECLARE
    user_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM users;
    RAISE NOTICE 'Total users in database: %', user_count;
END $$;