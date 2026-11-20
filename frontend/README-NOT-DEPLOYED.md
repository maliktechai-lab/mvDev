# ⚠️ This directory is NOT the deployed frontend

The **deployed** frontend is `../frontend-blink/`.

This `frontend/` directory is an earlier iteration kept for reference.
Docker Compose builds and serves `frontend-blink` only.

## Why two directories?

`frontend-blink` was rebuilt on a newer stack (React 19, Vite 8, Tailwind v4,
TypeScript 6, Vitest) with performance and design improvements.  `frontend/`
used the older React 18 / Vite 5 / Tailwind v3 stack.

## What to do

- **Making UI changes?** Edit files in `frontend-blink/`.
- **Running locally?**  `cd frontend-blink && npm run dev`
- **Building for production?** `docker compose up --build` (builds `frontend-blink` only)

The `frontend/` directory will be removed in a future cleanup commit once all
features have been confirmed present in `frontend-blink`.
