# DeepGuard — Design & Security Notes

This document captures architectural decisions, known limitations, and security
guidance for operators deploying DeepGuard.

---

## Security

### Upload Path Traversal (resolved)

All three detection endpoints (`/detect/image`, `/detect/video`, `/detect/audio`)
previously extracted the file extension directly from `file.filename`, which is
user-controlled.  A filename like `../../etc/passwd` could in theory produce an
unexpected suffix, and a double-extension payload like `shell.php.jpg` relied on
the `suffix` call returning only `.jpg`.

**Fix applied (Oct 2026):** The extracted suffix is now validated against a
strict per-endpoint whitelist before use.  If the suffix is absent or not in the
whitelist, a safe default is substituted.  The stored filename is always
`uuid4().hex + whitelisted_ext`, so no user-controlled bytes reach the filesystem
path.

Whitelists:
- Image: `.jpg .jpeg .png .webp .bmp`
- Video: `.mp4 .webm .avi .mov .mkv`
- Audio: `.mp3 .wav .flac .ogg .m4a`

---

### Video Memory Exhaustion (resolved)

`detect_video.py` previously read the entire upload body into memory before
checking the file size.  A malicious client could send a multi-gigabyte payload
and exhaust server RAM before the size check fired.

**Fix applied (Oct 2026):** The endpoint now inspects the `Content-Length` request
header before reading the body.  If the header value already exceeds the
configured `MAX_VIDEO_SIZE_MB` limit the request is rejected immediately with
HTTP 413, without reading the body.  The post-read size check is retained as a
hard backstop for clients that lie about `Content-Length`.

> **Note:** `Content-Length` is client-supplied and therefore unreliable.
> For a fully hardened deployment, configure your Nginx `client_max_body_size`
> directive to match (or be slightly larger than) `MAX_VIDEO_SIZE_MB`.  Nginx
> will then reject oversized uploads at the proxy layer, before the payload
> reaches the FastAPI process.

---

### CORS Configuration

`main.py` uses `allow_headers=["*"]` and derives `allow_origins` from the
`ALLOWED_ORIGINS` environment variable.

**Development:** `ALLOWED_ORIGINS=http://localhost:3000` — permissive headers
are acceptable because there is no cross-origin attack surface on a local
machine.

**Production:** Set `ALLOWED_ORIGINS` to the exact frontend domain(s), e.g.:

```
ALLOWED_ORIGINS=https://app.deepguard.example.com
```

Replace `allow_headers=["*"]` with the explicit list of headers the frontend
actually sends, for example:

```python
allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
```

This prevents a compromised or malicious third-party origin from making
credentialed requests with unexpected headers.

---

### Brute-Force Protection — Login Endpoint (open issue)

The `POST /api/v1/auth/login` endpoint has **no rate limiting**.  An attacker
can make unlimited login attempts, enabling:

- Credential stuffing (replaying leaked username/password lists)
- Password spraying (trying common passwords against many accounts)

**Recommended mitigations (choose one or combine):**

1. **`slowapi` middleware** — add per-IP rate limiting in Python:
   ```python
   # pip install slowapi
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded

   limiter = Limiter(key_func=get_remote_address, storage_uri=settings.REDIS_URL)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

   @router.post("/login")
   @limiter.limit("10/minute")
   async def login(request: Request, ...):
       ...
   ```

2. **Nginx `limit_req_zone`** — enforce limits at the proxy layer (zero Python
   overhead, effective even during high load):
   ```nginx
   limit_req_zone $binary_remote_addr zone=login:10m rate=10r/m;

   location /api/v1/auth/login {
       limit_req zone=login burst=5 nodelay;
       proxy_pass http://backend;
   }
   ```

3. **Account lockout** — add a `failed_login_attempts` counter and
   `locked_until` timestamp column to the `User` model.  Lock the account for
   N minutes after M consecutive failures.  Reset on successful login.

**Current status:** Not implemented.  Medium-risk in isolation; becomes
higher-risk if the application is exposed to the internet without a WAF or
Cloudflare in front.

---

### Password & Token Handling

- Passwords are hashed with bcrypt (via `passlib`) before storage.  Plaintext
  passwords are never persisted or logged.
- JWT tokens are signed with `JWT_SECRET_KEY` (HS256).  Ensure this key is at
  least 32 random bytes and never committed to source control.
- Refresh tokens are stored as SHA-256 hashes in the database.  The raw token
  is only ever returned to the client once (at issue time).
- The `log_requests` middleware in `main.py` logs only the HTTP method and URL
  path.  Request bodies (including passwords and tokens) are **never** logged.
- The `UserResponse` schema explicitly enumerates returned fields (`id`, `email`,
  `username`, `full_name`, `is_active`, `is_admin`, `created_at`).  The
  `hashed_password` field is not included and will never be serialised into an
  API response.

---

### Sensitive Configuration

All secrets (`SECRET_KEY`, `JWT_SECRET_KEY`, `DATABASE_URL`,
`POSTGRES_PASSWORD`) must be provided via environment variables or the `.env`
file.  The `.env` file is listed in `.gitignore` and must never be committed.
Use `.env.example` as the template — it contains no real secrets.
