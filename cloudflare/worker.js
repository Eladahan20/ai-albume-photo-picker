function createCorsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, Origin",
    "Access-Control-Max-Age": "86400"
  };
}

function extractFirstJsonArray(text) {
  const match = String(text || "").match(/\[[\s\S]*\]/);
  if (!match) return null;
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, Number(value || 0)));
}

function isResultObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const numericFields = ["quality", "composition", "emotion", "uniqueness", "overall"];
  return numericFields.every((field) => Number.isFinite(Number(value[field])));
}

function normalizeResultObject(value, indexHint = 0) {
  return {
    index: Number.isFinite(Number(value?.index)) ? Number(value.index) : indexHint,
    quality: clampNumber(value?.quality, 0, 100),
    composition: clampNumber(value?.composition, 0, 100),
    emotion: clampNumber(value?.emotion, 0, 100),
    uniqueness: clampNumber(value?.uniqueness, 0, 100),
    overall: clampNumber(value?.overall, 0, 100),
    scene: String(value?.scene || "unknown"),
    tags: Array.isArray(value?.tags) ? value.tags.map((tag) => String(tag)).filter(Boolean).slice(0, 6) : [],
    descriptor: value?.descriptor && typeof value.descriptor === "object" ? value.descriptor : undefined,
    reason: String(value?.reason || "").trim().slice(0, 240),
    albumWorthy: Boolean(value?.albumWorthy)
  };
}

function parseLooseScoredObject(text, indexHint = 0) {
  const source = String(text || "");
  if (!source.trim()) return null;

  const readNumber = (field) => {
    const match = source.match(new RegExp(`"${field}"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)`, "i"));
    return match ? clampNumber(match[1], 0, 100) : null;
  };
  const readString = (field) => {
    const match = source.match(new RegExp(`"${field}"\\s*:\\s*"([^"]*)`, "i"));
    return match ? match[1].trim() : "";
  };
  const readBoolean = (field) => {
    const match = source.match(new RegExp(`"${field}"\\s*:\\s*(true|false)`, "i"));
    return match ? match[1].toLowerCase() === "true" : null;
  };

  const quality = readNumber("quality");
  const composition = readNumber("composition");
  const emotion = readNumber("emotion");
  const uniqueness = readNumber("uniqueness");
  const overall = readNumber("overall");

  if ([quality, composition, emotion, uniqueness, overall].some((value) => value === null)) {
    return null;
  }

  const tagsBlockMatch = source.match(/"tags"\s*:\s*\[([^\]]*)/i);
  const tags = tagsBlockMatch
    ? Array.from(tagsBlockMatch[1].matchAll(/"([^"]+)"/g))
        .map((match) => match[1].trim())
        .filter(Boolean)
        .slice(0, 6)
    : [];

  return normalizeResultObject(
    {
      index: readNumber("index") ?? indexHint,
      quality,
      composition,
      emotion,
      uniqueness,
      overall,
      scene: readString("scene") || "unknown",
      tags,
      reason: readString("reason") || "Recovered from truncated model output.",
      albumWorthy: readBoolean("albumWorthy") ?? overall >= 70
    },
    indexHint
  );
}

function extractJsonResults(text) {
  const array = extractFirstJsonArray(text);
  if (Array.isArray(array)) {
    const normalized = array
      .map((item, index) => (isResultObject(item) ? normalizeResultObject(item, index) : null))
      .filter(Boolean);
    if (normalized.length) return normalized;
  }

  const source = String(text || "");
  const objectMatch = source.match(/\{[\s\S]*\}/);
  if (!objectMatch) return null;
  try {
    const parsed = JSON.parse(objectMatch[0]);
    return isResultObject(parsed) ? [normalizeResultObject(parsed, 0)] : null;
  } catch {
    const loose = parseLooseScoredObject(source, 0);
    return loose ? [loose] : null;
  }
}

function parseLooseBoolean(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (["true", "yes", "1"].includes(normalized)) return true;
  if (["false", "no", "0"].includes(normalized)) return false;
  return null;
}

function parseMarkdownResult(text, indexHint) {
  const source = String(text || "");
  if (!source.trim()) return null;

  const readNumber = (label) => {
    const match = source.match(new RegExp(`\\*\\*${label}:\\*\\*\\s*([0-9]{1,3})`, "i"));
    return match ? Number(match[1]) : 0;
  };
  const readText = (label) => {
    const match = source.match(new RegExp(`\\*\\*${label}:\\*\\*\\s*([^\\n]+)`, "i"));
    return match ? match[1].trim() : "";
  };

  const tagsMatch = source.match(/\*\*Tags:\*\*\s*\[([^\]]*)\]/i);
  const tags = tagsMatch
    ? tagsMatch[1]
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean)
    : [];

  const parsedBoolean = parseLooseBoolean(readText("AlbumWorthy"));
  const looksStructured =
    /\*\*Quality:\*\*/i.test(source) ||
    /\*\*Composition:\*\*/i.test(source) ||
    /\*\*Overall:\*\*/i.test(source);

  if (!looksStructured) return null;

  return [
    {
      index: readNumber("Index") || indexHint,
      quality: readNumber("Quality"),
      composition: readNumber("Composition"),
      emotion: readNumber("Emotion"),
      uniqueness: readNumber("Uniqueness"),
      overall: readNumber("Overall"),
      scene: readText("Scene") || "unknown",
      tags,
      reason: readText("Reason") || "Model returned a non-JSON formatted review.",
      albumWorthy: parsedBoolean ?? false
    }
  ];
}

function makeFallbackResult(index, reason) {
  return {
    index,
    quality: 0,
    composition: 0,
    emotion: 0,
    uniqueness: 0,
    overall: 0,
    scene: "unknown",
    tags: [],
    reason,
    albumWorthy: false
  };
}

function responseTextFromPayload(raw) {
  try {
    const parsed = JSON.parse(raw);
    const contentOut = parsed?.choices?.[0]?.message?.content;
    if (typeof contentOut === "string") return contentOut;
    if (Array.isArray(contentOut)) return contentOut.map((part) => part?.text || "").join("\n");
    const candidate =
      parsed?.result?.response ??
      parsed?.result?.output_text ??
      parsed?.response ??
      parsed?.text ??
      parsed?.result ??
      parsed;
    if (typeof candidate === "string") return candidate;
    return JSON.stringify(candidate);
  } catch {
    return raw;
  }
}

function jsonResponse(body, status, corsHeaders, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders,
      ...extraHeaders,
      "Content-Type": "application/json; charset=utf-8"
    }
  });
}

function textResponse(body, status, corsHeaders, extraHeaders = {}) {
  return new Response(body, {
    status,
    headers: {
      ...corsHeaders,
      ...extraHeaders,
      "Content-Type": "text/plain; charset=utf-8"
    }
  });
}

function requireProxyToken(request, env) {
  const expected = String(env.PROXY_BEARER_TOKEN || "").trim();
  if (!expected) return null;

  const authHeader = request.headers.get("Authorization") || "";
  const token = authHeader.replace(/^Bearer\s+/i, "").trim();
  if (token === expected) return null;

  return textResponse("Unauthorized", 401, createCorsHeaders(), {
    "WWW-Authenticate": 'Bearer realm="proxy"'
  });
}

function asCfModel(value) {
  const v = String(value || "").trim();
  if (!v) return "";
  if (v.startsWith("@cf/")) return v;
  if (v.startsWith("meta/")) return `@cf/${v}`;
  if (v.startsWith("models/")) return `@cf/${v.replace(/^models\//, "")}`;
  return `@cf/meta/${v}`;
}

function normalizeOpenAIBaseUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return "https://api.openai.com/v1/chat/completions";
  if (/^https?:\/\//i.test(raw)) return raw;
  return `https://${raw}`;
}

async function handleCloudflareProxy(body, env, corsHeaders) {
  const userMsg = body?.messages?.[0];
  const requestedModel = asCfModel(body?.model || "@cf/meta/llama-3.2-11b-vision-instruct");
  const content = Array.isArray(userMsg?.content) ? userMsg.content : [];

  const textBlock = content.find((c) => c?.type === "text");
  const imageBlocks = content.filter((c) => c?.type === "image" && c?.source?.data);

  if (!textBlock || !imageBlocks.length) {
    return textResponse("WORKER_VALIDATION_ERROR: expected text + image blocks.", 400, corsHeaders);
  }

  const perImageResults = [];

  for (let i = 0; i < imageBlocks.length; i += 1) {
    const imageBlock = imageBlocks[i];
    const imageUrl = `data:${imageBlock.source.media_type || "image/jpeg"};base64,${imageBlock.source.data}`;
    const requestMessages = [
      {
        role: "system",
        content: "You are a vision photo reviewer. Return only the JSON array requested by the user."
      },
      {
        role: "user",
        content: `${textBlock.text}\nThis request contains exactly one image. Return an array with exactly one object. Use index ${i}.`
      }
    ];

    console.log(
      JSON.stringify({
        stage: "cloudflare-run-attempt",
        model: requestedModel,
        imageIndex: i,
        imageCount: imageBlocks.length
      })
    );

    let response = null;
    let raw = "";
    let attempt = 0;
    while (attempt < 2) {
      attempt += 1;
      response = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/ai/run/${requestedModel}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.CF_API_TOKEN}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: requestMessages,
            image: imageUrl,
            max_tokens: 1400,
            temperature: 0.2
          })
        }
      );

      raw = await response.text();
      if (response.ok || response.status !== 408 || attempt >= 2) {
        break;
      }
      console.warn(
        JSON.stringify({
          stage: "cloudflare-run-timeout-retry",
          model: requestedModel,
          imageIndex: i,
          attempt,
          status: response.status
        })
      );
    }

    if (!response.ok) {
      console.error(
        JSON.stringify({
          stage: "cloudflare-run-error",
          model: requestedModel,
          imageIndex: i,
          status: response.status,
          body: String(raw).slice(0, 1200)
        })
      );
      perImageResults.push(
        makeFallbackResult(i, `Cloudflare AI request failed for this image (${response.status}).`)
      );
      continue;
    }

    const textOut = responseTextFromPayload(raw);
    const arr = extractJsonResults(textOut) || parseMarkdownResult(textOut, i);
    if (!Array.isArray(arr)) {
      console.error(
        JSON.stringify({
          stage: "cloudflare-run-parse-error",
          model: requestedModel,
          imageIndex: i,
          raw: String(raw).slice(0, 1200),
          body: String(textOut).slice(0, 1200)
        })
      );
      perImageResults.push(
        makeFallbackResult(i, "Model returned an unparseable response for this image.")
      );
      continue;
    }

    if (!arr.length) {
      console.warn(
        JSON.stringify({
          stage: "cloudflare-run-empty-result",
          model: requestedModel,
          imageIndex: i
        })
      );
      perImageResults.push(makeFallbackResult(i, "Model returned an empty result for this image."));
      continue;
    }

    perImageResults.push(arr[0]);
  }

  return textResponse(JSON.stringify(perImageResults), 200, corsHeaders, {
    "X-Upstream-Route": "ai/run",
    "X-Upstream-Model": requestedModel
  });
}

async function handleOpenAIProxy(body, env, corsHeaders) {
  const apiKey = String(env.OPENAI_API_KEY || "").trim();
  if (!apiKey) {
    return textResponse("OPENAI_API_KEY is not configured on the Worker.", 500, corsHeaders);
  }

  const baseUrl = normalizeOpenAIBaseUrl(env.OPENAI_BASE_URL);
  const model = String(body?.model || env.OPENAI_MODEL || "gpt-4.1-mini").trim();
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  const userMsg = messages[0];
  const content = Array.isArray(userMsg?.content) ? userMsg.content : [];
  const textBlock = content.find((c) => c?.type === "text");
  const imageBlocks = content.filter((c) => c?.type === "image_url" && c?.image_url?.url);

  if (!textBlock || !imageBlocks.length) {
    return textResponse("WORKER_VALIDATION_ERROR: expected text + image_url blocks.", 400, corsHeaders);
  }

  const response = await fetch(baseUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.2
    })
  });

  const raw = await response.text();
  if (!response.ok) {
    console.error(
      JSON.stringify({
        stage: "openai-proxy-error",
        model,
        status: response.status,
        body: String(raw).slice(0, 1200)
      })
    );
    return textResponse(raw, response.status, corsHeaders, {
      "X-Error-Source": "upstream-openai",
      "X-Upstream-Model": model
    });
  }

  const outputText = responseTextFromPayload(raw);
  return textResponse(outputText, 200, corsHeaders, {
    "X-Upstream-Model": model
  });
}

export default {
  async fetch(request, env) {
    const corsHeaders = createCorsHeaders();

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    if (request.method === "GET") {
      return jsonResponse(
        {
          ok: true,
          service: "ai-album-selector-proxy",
          providerSupport: ["cloudflare", "openai"],
          hasCloudflareConfig: Boolean(env.CF_ACCOUNT_ID && env.CF_API_TOKEN),
          hasOpenAIConfig: Boolean(env.OPENAI_API_KEY)
        },
        200,
        corsHeaders
      );
    }

    if (request.method !== "POST") {
      return textResponse("Method not allowed", 405, corsHeaders);
    }

    const authError = requireProxyToken(request, env);
    if (authError) {
      return authError;
    }

    try {
      const body = await request.json();
      const provider = String(body?.provider || "cloudflare").trim().toLowerCase();

      if (provider === "openai") {
        return await handleOpenAIProxy(body, env, corsHeaders);
      }

      if (provider === "cloudflare") {
        if (!env.CF_ACCOUNT_ID || !env.CF_API_TOKEN) {
          return textResponse("CF_ACCOUNT_ID and CF_API_TOKEN must be configured on the Worker.", 500, corsHeaders);
        }
        return await handleCloudflareProxy(body, env, corsHeaders);
      }

      return textResponse(`Unsupported provider: ${provider}`, 400, corsHeaders);
    } catch (error) {
      return textResponse(String(error?.message || error), 500, corsHeaders);
    }
  }
};
