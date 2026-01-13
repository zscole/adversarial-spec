# Technical Specification: User Authentication API

## Overview
A simple REST API for user authentication with JWT tokens.

## Goals
- Allow users to register and login
- Issue JWT tokens for authenticated sessions
- Support password reset via email

## API Endpoints

### POST /auth/register
Request: `{ "email": string, "password": string }`
Response: `{ "userId": string, "token": string }`

### POST /auth/login
Request: `{ "email": string, "password": string }`
Response: `{ "token": string, "expiresIn": number }`

### POST /auth/reset-password
Request: `{ "email": string }`
Response: `{ "message": string }`

## Data Model

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
```

## Security
- Passwords hashed with bcrypt
- JWT tokens expire in 24 hours
- Rate limiting: 10 requests per minute per IP

## Open Questions
- Should we support OAuth providers?
- What's the token refresh strategy?
