-- init-db/01-init-schema.sql
-- Script d'initialisation de la base de données TalkAI

-- Activer les extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Supprimer l'enum existant s'il existe et le recréer
DROP TYPE IF EXISTS role_enum CASCADE;
CREATE TYPE role_enum AS ENUM ('utilisateur', 'admin');

-- Créer la table users avec TOUTES les colonnes nécessaires
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    nom VARCHAR(255),
    date_naissance DATE,
    consent BOOLEAN NOT NULL DEFAULT FALSE,
    role role_enum DEFAULT 'utilisateur' NOT NULL,
    date_creation TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    avatar_url VARCHAR(500),
    bio TEXT,
    country VARCHAR(100)
);

-- Créer la table conversations
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    user1_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user2_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category VARCHAR(255) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE
);

-- Créer la table messages
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    content TEXT NOT NULL,
    user_input TEXT,
    user_audio_base64 TEXT,
    ia_audio_base64 TEXT,
    response TEXT,
    marker VARCHAR(255),
    suggestion TEXT,
    ia_audio_duration FLOAT
);

-- Créer la table conversation_logs (pour la compatibilité)
CREATE TABLE IF NOT EXISTS conversation_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    user_audio_base64 TEXT,
    user_input TEXT
);

-- Créer les index pour les performances
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_avatar_url ON users(avatar_url);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_conversations_user1 ON conversations(user1_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user2 ON conversations(user2_id);
CREATE INDEX IF NOT EXISTS idx_conversations_category ON conversations(category);
CREATE INDEX IF NOT EXISTS idx_conversations_active ON conversations(active);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_marker ON messages(marker);

-- Ajouter des contraintes de validation
ALTER TABLE users ADD CONSTRAINT IF NOT EXISTS check_email_format 
    CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$');

ALTER TABLE messages ADD CONSTRAINT IF NOT EXISTS check_ia_audio_duration_positive 
    CHECK (ia_audio_duration IS NULL OR ia_audio_duration >= 0);

-- Afficher un message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'TalkAI Database Schema initialized successfully!';
END $$;