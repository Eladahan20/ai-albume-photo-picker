function createCorsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
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

function responseTextFromPayload(raw) {
  try {
    const parsed = JSON.parse(raw);
    const contentOut = parsed?.choices?.[0]?.message?.content;
    if (typeof contentOut === "string") return contentOut;
    if (Array.isArray(contentOut)) return contentOut.map((part) => part?.text || "").join("\n");
    return parsed?.result?.response || parsed?.result?.output_text || parsed?.response || parsed?.text || raw;
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
  const requestedModel = String(body?.model || "@cf/meta/llama-3.2-11b-vision-instruct").trim();
  const modelCandidates = Array.from(
    new Set(
      [
        asCfModel(requestedModel),
        requestedModel,
        asCfModel(requestedModel.replace(/^@cf\//, "")),
        "@cf/meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-11b-vision-instruct",
        "llama-3.2-11b-vision-instruct"
      ].filter(Boolean)
    )
  );
  const content = Array.isArray(userMsg?.content) ? userMsg.content : [];

  const textBlock = content.find((c) => c?.type === "text");
  const imageBlocks = content.filter((c) => c?.type === "image" && c?.source?.data);

  if (!textBlock || !imageBlocks.length) {
    return textResponse("WORKER_VALIDATION_ERROR: expected text + image blocks.", 400, corsHeaders);
  }

  const cfMessagesFormatA = [
    {
      role: "system",
      content: "You are a vision photo reviewer. Return only the JSON array requested by the user."
    },
    {
      role: "user",
      content: textBlock.text
    },
    ...imageBlocks.map((img, idx) => ({
      role: "user",
      content: [
        { type: "text", text: `Image ${idx + 1}` },
        {
          type: "image_url",
          image_url: {
            url: `data:${img.source.media_type || "image/jpeg"};base64,${img.source.data}`
          }
        }
      ]
    }))
  ];

  const cfMessagesFormatB = [
    {
      role: "system",
      content: "You are a vision photo reviewer. Return only the JSON array requested by the user."
    },
    {
      role: "user",
      content: [
        { type: "text", text: textBlock.text },
        ...imageBlocks.map((img) => ({
          type: "image_url",
          image_url: {
            url: `data:${img.source.media_type || "image/jpeg"};base64,${img.source.data}`
          }
        }))
      ]
    }
  ];

  async function callCloudflareChat(messages) {
    let last = null;
    for (const candidateModel of modelCandidates) {
      const response = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/ai/v1/chat/completions`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.CF_API_TOKEN}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: candidateModel,
            messages,
            temperature: 0.2,
            max_tokens: 1400
          })
        }
      );
      const raw = await response.text();
      last = { response, raw, candidateModel, route: "ai/v1/chat/completions" };
      if (response.ok) return last;
    }
    return last;
  }

  async function callCloudflareRun() {
    let last = null;
    for (const candidateModel of modelCandidates) {
      const response = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/ai/run/${candidateModel}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.CF_API_TOKEN}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: body?.messages,
            max_tokens: 1400
          })
        }
      );
      const raw = await response.text();
      last = { response, raw, candidateModel, route: "ai/run" };
      if (response.ok) return last;
    }
    return last;
  }

  let firstAttempt = await callCloudflareChat(cfMessagesFormatA);
  let { response: cfResponse, raw } = firstAttempt;
  if (!cfResponse.ok && /Unable to add image/i.test(raw)) {
    const secondAttempt = await callCloudflareChat(cfMessagesFormatB);
    cfResponse = secondAttempt.response;
    raw = secondAttempt.raw;
    firstAttempt = secondAttempt;
  }

  if (!cfResponse.ok && imageBlocks.length > 1) {
    const perImageResults = [];
    const perImageErrors = [];
    for (let i = 0; i < imageBlocks.length; i += 1) {
      const singleImageMessages = [
        {
          role: "system",
          content: "You are a vision photo reviewer. Return only the JSON array requested by the user."
        },
        {
          role: "user",
          content: [
            { type: "text", text: `${textBlock.text}\nThis request contains exactly one image. Return array with one object.` },
            {
              type: "image_url",
              image_url: {
                url: `data:${imageBlocks[i].source.media_type || "image/jpeg"};base64,${imageBlocks[i].source.data}`
              }
            }
          ]
        }
      ];

      const singleCall = await callCloudflareChat(singleImageMessages);
      if (!singleCall.response.ok) {
        perImageErrors.push({
          image: i,
          status: singleCall.response.status,
          model: singleCall.candidateModel,
          route: singleCall.route,
          body: String(singleCall.raw).slice(0, 700)
        });
        continue;
      }

      const textOut = responseTextFromPayload(singleCall.raw);
      const arr = extractFirstJsonArray(textOut);
      if (!Array.isArray(arr) || !arr.length) {
        perImageErrors.push({
          image: i,
          status: 200,
          body: `NO_JSON_ARRAY_IN_RESPONSE: ${String(textOut).slice(0, 700)}`
        });
        continue;
      }
      perImageResults.push(arr[0]);
    }

    if (perImageResults.length === imageBlocks.length) {
      return textResponse(JSON.stringify(perImageResults), 200, corsHeaders);
    }

    const runAttemptAfterPerImage = await callCloudflareRun();
    if (runAttemptAfterPerImage?.response?.ok) {
      const outputText = responseTextFromPayload(runAttemptAfterPerImage.raw);
      return textResponse(outputText, 200, corsHeaders);
    }

    return jsonResponse(
      {
        error: "UPSTREAM_MULTI_IMAGE_FAILED_AND_PER_IMAGE_FALLBACK_FAILED",
        initial_status: cfResponse.status,
        initial_route: firstAttempt?.route,
        initial_model: firstAttempt?.candidateModel,
        initial_body: String(raw).slice(0, 1000),
        per_image_errors: perImageErrors
      },
      400,
      corsHeaders,
      { "X-Error-Source": "upstream-cloudflare-ai" }
    );
  }

  if (!cfResponse.ok) {
    const runAttempt = await callCloudflareRun();
    if (runAttempt?.response?.ok) {
      const outputText = responseTextFromPayload(runAttempt.raw);
      return textResponse(outputText, 200, corsHeaders);
    }
  }

  if (!cfResponse.ok) {
    return textResponse(raw, cfResponse.status, corsHeaders, {
      "X-Error-Source": "upstream-cloudflare-ai",
      "X-Upstream-Route": firstAttempt?.route || "unknown",
      "X-Upstream-Model": firstAttempt?.candidateModel || "unknown"
    });
  }

  let outputText = raw;
  try {
    outputText = responseTextFromPayload(raw);
  } catch {
    outputText = raw;
  }

  return textResponse(outputText, 200, corsHeaders);
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
