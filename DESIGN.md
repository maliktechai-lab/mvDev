# DeepGuard — Design & Security Notes

> **This file is a summary.  The full design document is at [`docs/DESIGN.md`](docs/DESIGN.md).**

---

## Security posture (current)

| Area | Status |
|---|---|
| Login brute-force protection | ✅ Implemented — slowapi (IP rate limit, 5/min) + per-account lockout (15 min after 5 failures) |
| Upload path traversal | ✅ Resolved — uuid4 filenames + strict extension whitelist |
| Video memory exhaustion | ✅ Resolved — Content-Length pre-check before body read |
| DB commit on failure | ✅ Resolved — explicit commit before re-raise in image endpoint |
| Random-weight inference in production | ✅ Blocked — `require_model_ready()` gate + production startup validation |
| Docker internal service exposure | ✅ Resolved — Postgres, Redis, backend, Flower no longer mapped to host |
| Weak production secrets | ✅ Resolved — startup validation refuses to run with placeholder secrets |
| Security headers | ✅ Added — X-Content-Type-Options, X-Frame-Options, CSP, Referrer-Policy via Nginx |
| CORS over-permissive headers | ⚠️ Acceptable in development; restrict `allow_headers` for production |
| Refresh token reuse detection | ⚠️ Token rotation present; proactive reuse detection not yet implemented |
| Email verification / password reset | ⚠️ Not yet implemented |
| Rate limiting on other auth endpoints | ✅ Nginx `limit_req` zones cover `/auth/*` and all API traffic |

---

For architecture decisions, confirmed bugs, root-cause analysis, and phased
requirements, see [`docs/DESIGN.md`](docs/DESIGN.md).
