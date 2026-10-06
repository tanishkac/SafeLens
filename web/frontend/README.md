This is the Next.js 15 frontend for the SafeLens demo. It talks to the FastAPI backend in `web/` and shows upload, analysis status, and results.

## Prerequisites

- Node.js 20 LTS (recommended) or ≥ 18.18
- pnpm 8+
- A running backend at `http://localhost:8000` (see `web/README.md`)

## 1) Configure environment (Optional)

The frontend works out-of-the-box with default settings. If your backend is running on a different port, you can create a `.env` file:

```bash
cd web/frontend
cp .env.example .env
```

Local `.env`:

```env
# Backend API
BACKEND_URL=http://localhost:8000
```


## 2) Install deps and run

```bash
pnpm install
pnpm dev
```

Open http://localhost:3000 in your browser.

Sanity checks
- Visiting http://localhost:3000 renders the UI.
- “Sign in” redirects to the auth service and returns to the app.
- The backend health endpoint is reachable (the UI calls through `BACKEND_URL`).

## 3) Backend integration

- The frontend calls the backend through its own Next.js route handlers using `BACKEND_URL` — you do not need to enable CORS on the backend in this topology.
- Ensure the backend is up (uvicorn on port 8000) and migrations are applied before using the UI.

## 4) Production build

```bash
pnpm build
pnpm start
```

Behind a proxy, set `PORT` and `HOSTNAME` as needed, and make sure `AUTH_URL/NEXTAUTH_URL/NEXT_PUBLIC_APP_URL` point to the public site URL.

Local vs Production mapping
- Local auth: `AUTH_ISSUER_URL=http://localhost:3001/api/auth` → Prod: `https://auth.example.com/api/auth`
- Local app URLs: `http://localhost:3000` → Prod: `https://app.example.com`
- Local backend: `BACKEND_URL=http://localhost:8000` → Prod: your backend host URL

Port assumptions
- Frontend: 3000
- Auth service: 3001
- Backend API: 8000

## Project Structure

- `src/app` — Application routes and layouts
- `src/components` — Reusable UI components
- `src/lib` — Auth.js config and helpers

## Troubleshooting

- Auth callback errors: verify `AUTH_SECRET` and your OIDC client settings (redirect URLs must include `/api/auth/callback/...`).
- CallbackRouteError with "unexpected JWT iss": ensure the provider's issuer matches. Our config derives the issuer from the origin of `AUTH_ISSUER_URL` (e.g., `http://localhost:3001`), while discovery/endpoints are under `/api/auth`. Clear cookies for `localhost:3000` and retry after fixing env.
- Cannot reach backend: check `BACKEND_URL` and backend logs at `web/`.
- Video analysis pending forever: confirm vLLM endpoints are running and backend `.env` is configured (see `web/README.md` vLLM section).
