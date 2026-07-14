-- ============================================================
-- Genkit AI
-- MySQL Database Schema
-- Version : 2.0
-- Engine  : MySQL 8+
-- ============================================================

CREATE DATABASE IF NOT EXISTS genkit_ai
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE genkit_ai;

-- ============================================================
-- CHATS
-- ============================================================

CREATE TABLE IF NOT EXISTS chats (

    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    session_id VARCHAR(120) NOT NULL,

    question LONGTEXT NOT NULL,

    answer LONGTEXT NOT NULL,

    intent VARCHAR(100) DEFAULT 'general',

    source VARCHAR(100) DEFAULT 'rag',

    confidence FLOAT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session(session_id),

    INDEX idx_created(created_at)

) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- LEADS
-- ============================================================

CREATE TABLE IF NOT EXISTS leads (

    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    name VARCHAR(150) NOT NULL,

    email VARCHAR(255) NOT NULL,

    phone VARCHAR(30),

    company VARCHAR(200),

    message TEXT,

    status VARCHAR(50) DEFAULT 'New',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_email(email),

    INDEX idx_status(status)

) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- FEEDBACK
-- ============================================================

CREATE TABLE IF NOT EXISTS feedback (

    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    session_id VARCHAR(120) NOT NULL,

    question LONGTEXT,

    answer LONGTEXT,

    rating INT NOT NULL,

    comments TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_rating
        CHECK (rating BETWEEN 1 AND 5),

    INDEX idx_feedback_session(session_id),

    INDEX idx_feedback_rating(rating)

) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- USER PROFILES
-- ============================================================

CREATE TABLE IF NOT EXISTS user_profiles (

    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    session_id VARCHAR(120) NOT NULL UNIQUE,

    name VARCHAR(150),

    email VARCHAR(255),

    phone VARCHAR(30),

    company VARCHAR(200),

    interest TEXT,

    last_query TEXT,

    total_chats INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_user_email(email),

    INDEX idx_user_name(name)

) ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- OPTIONAL SAMPLE ADMIN USER
-- ============================================================

-- INSERT INTO user_profiles
-- (
--     session_id,
--     name,
--     email
-- )
-- VALUES
-- (
--     'admin',
--     'Administrator',
--     'admin@genkit.in'
-- );


-- ============================================================
-- VIEWS
-- ============================================================

CREATE OR REPLACE VIEW chat_statistics AS

SELECT

    COUNT(*) AS total_chats,

    COUNT(DISTINCT session_id) AS total_sessions,

    MAX(created_at) AS latest_chat

FROM chats;


CREATE OR REPLACE VIEW lead_statistics AS

SELECT

    COUNT(*) AS total_leads,

    SUM(status='New') AS new_leads,

    SUM(status='Contacted') AS contacted,

    SUM(status='Closed') AS closed

FROM leads;


CREATE OR REPLACE VIEW feedback_statistics AS

SELECT

    COUNT(*) AS total_feedback,

    ROUND(AVG(rating),2) AS average_rating,

    MAX(created_at) AS latest_feedback

FROM feedback;


-- ============================================================
-- STORED PROCEDURE
-- ============================================================

DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS ClearOldChats(IN days_old INT)

BEGIN

    DELETE FROM chats

    WHERE created_at <
    DATE_SUB(NOW(), INTERVAL days_old DAY);

END $$

DELIMITER ;


-- ============================================================
-- END OF SCHEMA
-- ============================================================ 