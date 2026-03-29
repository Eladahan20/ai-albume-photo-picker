# AI Album Selector

React app for ranking and selecting the best photos for an album with AI scoring and a diversity pass.

Supports common formats plus `HEIC` / `HEIF` (auto-converted to JPEG in-browser on upload).

## Run

```bash
npm install
npm run dev
```

## Providers

### 1) Local (default, free, no API)
- Works fully in-browser.
- Uses image metrics (contrast, sharpness, saturation, scene heuristics) to generate analysis JSON.

### 2) Cloudflare endpoint (proxy Worker)
Set:

```bash
VITE_AI_PROVIDER=cloudflare
VITE_CLOUDFLARE_VISION_ENDPOINT=/api/cloudflare/vision
```

You can deploy the included worker in `cloudflare/worker.js`:

```bash
cd cloudflare
wrangler secret put CF_API_TOKEN
wrangler secret put CF_ACCOUNT_ID
wrangler deploy
```

Then set:

```bash
VITE_AI_PROVIDER=cloudflare
VITE_CLOUDFLARE_VISION_ENDPOINT=https://<your-worker>.workers.dev
```

In the app UI, when `Cloudflare endpoint (proxy)` is selected, you can now set:
- Proxy endpoint URL
- Optional proxy bearer token
- Model hint (sent as `model` in request body)

After changing Worker code, redeploy:

```bash
cd cloudflare
wrangler deploy
```

Optional Worker hardening:

```bash
cd cloudflare
wrangler secret put PROXY_BEARER_TOKEN
```

If you set `PROXY_BEARER_TOKEN`, the app must send the same value in the `Proxy Bearer Token` field.

### 3) OpenAI endpoint (proxy Worker)
The same Worker can proxy OpenAI server-side so coworkers do not need browser-side OpenAI keys.

Configure the Worker:

```bash
cd cloudflare
wrangler secret put OPENAI_API_KEY
```

Optional Worker vars:

```toml
[vars]
OPENAI_MODEL = "gpt-5-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1/responses"
```

Then in the app UI choose `OpenAI endpoint (proxy)` and set:
- Proxy endpoint URL
- Optional proxy bearer token
- OpenAI model

### 4) Cloudflare Direct (API key in UI)
- In the app, choose `Cloudflare Direct (API key)`.
- Fill `Cloudflare Account ID`, `Cloudflare API Key / Token`, and `Cloudflare Vision Model`.
- Values are stored in browser `localStorage` for convenience.
- Note: direct browser calls to `api.cloudflare.com` may be blocked by CORS. If that happens, use `Cloudflare endpoint (proxy)` mode with the included Worker.

### 5) OpenAI Direct (API key in UI)
- In the app, choose `OpenAI Direct (API key)`.
- Fill `OpenAI API Key`, `OpenAI Model`, and `OpenAI Base URL`.
- Values are stored in browser `localStorage`.
- If browser CORS blocks direct calls, use a backend/proxy endpoint instead.

Expected request body sent by the app:

```json
{
  "provider": "cloudflare",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "...prompt..." },
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "..."
          }
        }
      ]
    }
  ]
}
```

Your backend/proxy should return text containing a JSON array of:

```json
{
  "index": 0,
  "quality": 0,
  "composition": 0,
  "emotion": 0,
  "uniqueness": 0,
  "overall": 0,
  "scene": "...",
  "tags": ["..."],
  "reason": "...",
  "albumWorthy": true
}
```

The client extracts the first `[ ... ]` block from response text to handle markdown wrapping.

## Cloudflare deployment

### Option A: frontend only
Use this if coworkers only need the local analyzer.

1. Create a Cloudflare Pages project.
2. Build command: `npm run build`
3. Output directory: `dist`
4. Optional Pages env vars:

```bash
VITE_AI_PROVIDER=local
```

### Option B: frontend + Worker proxy
Use this if coworkers should test Cloudflare AI or OpenAI through the proxy.

1. Deploy the Worker:

```bash
cd cloudflare
wrangler login
wrangler secret put CF_API_TOKEN
wrangler secret put CF_ACCOUNT_ID
wrangler secret put OPENAI_API_KEY
wrangler secret put PROXY_BEARER_TOKEN
wrangler deploy
```

2. Copy the deployed Worker URL, for example:

```text
https://ai-album-selector-vision.<subdomain>.workers.dev
```

3. Create a Cloudflare Pages project for the Vite app:
   - Build command: `npm run build`
   - Output directory: `dist`

4. Set Pages environment variables:

```bash
VITE_AI_PROVIDER=openai
VITE_CLOUDFLARE_VISION_ENDPOINT=https://ai-album-selector-vision.<subdomain>.workers.dev
VITE_OPENAI_MODEL=gpt-5-mini
```

If you want Cloudflare AI as the default instead:

```bash
VITE_AI_PROVIDER=cloudflare
VITE_CLOUDFLARE_MODEL=@cf/meta/llama-3.2-11b-vision-instruct
VITE_CLOUDFLARE_VISION_ENDPOINT=https://ai-album-selector-vision.<subdomain>.workers.dev
```

5. In the deployed app, choose either:
   - `OpenAI endpoint (proxy)`
   - `Cloudflare endpoint (proxy)`

6. If you configured `PROXY_BEARER_TOKEN`, paste that value into the app’s `Proxy Bearer Token` field.
