import React, { useEffect, useMemo, useRef, useState } from "react";

const MAX_UPLOADS = 1000;
const DEFAULT_BATCH_SIZE = 3;
const MIN_ALBUM = 10;
const MAX_ALBUM = 100;
const CLOUDFLARE_ENDPOINT = import.meta.env.VITE_CLOUDFLARE_VISION_ENDPOINT || "/api/cloudflare/vision";
const DEFAULT_PROVIDER = (import.meta.env.VITE_AI_PROVIDER || "local").toLowerCase();
const DEFAULT_CF_MODEL = import.meta.env.VITE_CLOUDFLARE_MODEL || "@cf/meta/llama-3.2-11b-vision-instruct";
const DEFAULT_OPENAI_MODEL = import.meta.env.VITE_OPENAI_MODEL || "gpt-4.1-mini";
const DEFAULT_OPENAI_BASE_URL = import.meta.env.VITE_OPENAI_BASE_URL || "https://api.openai.com/v1/chat/completions";

const ACCEPTED_MIME_PREFIX = "image/";
const HEIC_EXTENSIONS = new Set(["heic", "heif"]);

function uid() {
  return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function nowStamp() {
  return new Date().toLocaleTimeString();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeEndpointUrl(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  if (/^https?:\/\//i.test(value)) return value;
  if (value.startsWith("//")) return `https:${value}`;
  if (value.includes(".workers.dev") || value.includes(".pages.dev")) return `https://${value}`;
  return value;
}

function isAbsoluteHttpUrl(value) {
  return /^https?:\/\//i.test(String(value || "").trim());
}

function getBatchSizeForProvider(provider) {
  return provider === "cloudflare" ? 1 : DEFAULT_BATCH_SIZE;
}

function errorToMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const maybeMessage = error.message || error.reason || error.error || error.code;
    if (maybeMessage) return String(maybeMessage);
    try {
      const json = JSON.stringify(error);
      if (json && json !== "{}") return json;
    } catch {
      return String(error);
    }
  }
  return String(error);
}

function isAbortError(error) {
  return error?.name === "AbortError" || /aborted|abort/i.test(String(error?.message || error));
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new DOMException("Operation aborted", "AbortError");
  }
}

function fileExtension(name = "") {
  const parts = name.toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function isHeicFile(file) {
  const mime = String(file?.type || "").toLowerCase();
  return mime === "image/heic" || mime === "image/heif" || HEIC_EXTENSIONS.has(fileExtension(file?.name));
}

function isBrowserReadableImage(file) {
  const mime = String(file?.type || "").toLowerCase();
  return mime.startsWith("image/") && mime !== "image/heic" && mime !== "image/heif";
}

function needsHeicConversion(file) {
  return isHeicFile(file) && !isBrowserReadableImage(file);
}

function isSupportedImage(file) {
  const mime = String(file?.type || "").toLowerCase();
  return mime.startsWith(ACCEPTED_MIME_PREFIX) || HEIC_EXTENSIONS.has(fileExtension(file?.name));
}

async function convertHeicToJpeg(file) {
  const { default: heic2any } = await import("heic2any");
  const output = await heic2any({ blob: file, toType: "image/jpeg", quality: 0.9 });
  const jpegBlob = Array.isArray(output) ? output[0] : output;
  if (!(jpegBlob instanceof Blob)) {
    throw new Error(`HEIC converter returned invalid output: ${errorToMessage(jpegBlob)}`);
  }
  return new File([jpegBlob], file.name.replace(/\.(heic|heif)$/i, ".jpg"), {
    type: "image/jpeg",
    lastModified: file.lastModified
  });
}

function extractFirstJsonArray(text) {
  const match = text.match(/\[[\s\S]*\]/);
  if (!match) {
    throw new Error("No JSON array found in API response text.");
  }
  return JSON.parse(match[0]);
}

async function fileToImageBitmap(file) {
  if ("createImageBitmap" in window) {
    return createImageBitmap(file);
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

async function toBase64JpegThumbnail(file, base = 512, quality = 0.82) {
  const img = await fileToImageBitmap(file);
  const width = img.width;
  const height = img.height;
  const maxSide = Math.max(width, height);
  const scale = maxSide > base ? base / maxSide : 1;
  const outWidth = Math.max(1, Math.round(width * scale));
  const outHeight = Math.max(1, Math.round(height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = outWidth;
  canvas.height = outHeight;
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.drawImage(img, 0, 0, outWidth, outHeight);

  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  return dataUrl.split(",")[1];
}

function average(nums) {
  if (!nums.length) return 0;
  return nums.reduce((sum, n) => sum + n, 0) / nums.length;
}

function hashStringToUnit(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return Math.abs(hash >>> 0) / 4294967295;
}

function applyDiversitySelection(analyses, targetCount) {
  const sceneCounts = analyses.reduce((acc, item) => {
    const key = (item.scene || "unknown").trim().toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const penalized = analyses.map((item) => {
    const key = (item.scene || "unknown").trim().toLowerCase();
    const count = sceneCounts[key] || 1;
    const penalty = Math.max(0, count - 1) * 6;
    const adjustedOverall = clamp(Math.round((item.overall || 0) - penalty), 0, 100);
    return { ...item, adjustedOverall, diversityPenalty: penalty };
  });

  const ranked = [...penalized].sort((a, b) => b.adjustedOverall - a.adjustedOverall);
  return ranked.slice(0, targetCount).map((item) => item.index);
}

function extractThemeKeywords(themePrompt) {
  return String(themePrompt || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .filter((word) => word.length >= 3);
}

function buildThemeProfile(themePrompt) {
  const keywords = extractThemeKeywords(themePrompt);
  const peopleWords = new Set([
    "people",
    "person",
    "portrait",
    "portraits",
    "family",
    "friends",
    "wedding",
    "couple",
    "kids",
    "children",
    "baby",
    "bride",
    "groom",
    "human",
    "humans"
  ]);
  const peopleFocused = keywords.some((w) => peopleWords.has(w));
  return { keywords, peopleFocused };
}

function normalizeDescriptor(input, scene, tags = []) {
  const raw = input && typeof input === "object" ? input : {};
  const joinedTags = tags.map((t) => String(t).toLowerCase());
  const has = (x) => joinedTags.some((t) => t.includes(x));
  const sceneLower = String(scene || "unknown").toLowerCase();

  const peoplePresence = clamp(Number(raw.people_presence ?? (has("person") || has("people") || has("portrait") ? 80 : 10)), 0, 100);
  const textPresence = clamp(Number(raw.text_presence ?? (has("document") || has("text") || has("screenshot") ? 85 : 10)), 0, 100);
  const screenshotLikelihood = clamp(Number(raw.screenshot_likelihood ?? (has("screenshot") || has("screen") ? 85 : 10)), 0, 100);
  const documentLikelihood = clamp(Number(raw.document_likelihood ?? (has("document") || has("receipt") || has("invoice") ? 85 : 10)), 0, 100);

  const primarySubject =
    String(raw.primary_subject || "").toLowerCase() ||
    (peoplePresence > 60
      ? "people"
      : screenshotLikelihood > 70
        ? "screenshot"
        : documentLikelihood > 70
          ? "document"
          : sceneLower === "landscape" || sceneLower === "outdoor"
            ? "landscape"
            : "object");

  const setting = String(raw.setting || sceneLower || "unknown").toLowerCase();
  const shotType = String(raw.shot_type || (sceneLower === "portrait" ? "portrait" : "medium")).toLowerCase();
  const eventRole = String(raw.event_role || "context").toLowerCase();
  const colorProfile = String(raw.color_profile || (has("warm") ? "warm" : has("cool") ? "cool" : "neutral")).toLowerCase();

  return {
    primary_subject: primarySubject,
    setting,
    shot_type: shotType,
    event_role: eventRole,
    color_profile: colorProfile,
    people_presence: peoplePresence,
    text_presence: textPresence,
    screenshot_likelihood: screenshotLikelihood,
    document_likelihood: documentLikelihood
  };
}

function hammingDistance(a, b) {
  if (!a || !b || a.length !== b.length) return Number.MAX_SAFE_INTEGER;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) diff += 1;
  }
  return diff;
}

async function buildImageFingerprint(file) {
  const img = await fileToImageBitmap(file);
  const width = img.width;
  const height = img.height;
  const ratio = width / Math.max(1, height);
  const fmt = (file.type || "image/unknown").toLowerCase();

  const hashCanvas = document.createElement("canvas");
  hashCanvas.width = 16;
  hashCanvas.height = 16;
  const hashCtx = hashCanvas.getContext("2d");
  hashCtx.drawImage(img, 0, 0, 16, 16);
  const hashData = hashCtx.getImageData(0, 0, 16, 16).data;
  const lum = [];
  let lumSum = 0;
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let satSum = 0;
  for (let i = 0; i < hashData.length; i += 4) {
    const r = hashData[i];
    const g = hashData[i + 1];
    const b = hashData[i + 2];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const sat = max === 0 ? 0 : (max - min) / max;
    lum.push(l);
    lumSum += l;
    rSum += r;
    gSum += g;
    bSum += b;
    satSum += sat;
  }
  const lumMean = lumSum / lum.length;
  const aHash = lum.map((v) => (v >= lumMean ? "1" : "0")).join("");

  const dHashCanvas = document.createElement("canvas");
  dHashCanvas.width = 9;
  dHashCanvas.height = 8;
  const dHashCtx = dHashCanvas.getContext("2d");
  dHashCtx.drawImage(img, 0, 0, 9, 8);
  const dHashData = dHashCtx.getImageData(0, 0, 9, 8).data;
  const dHashLum = [];
  for (let i = 0; i < dHashData.length; i += 4) {
    dHashLum.push(0.299 * dHashData[i] + 0.587 * dHashData[i + 1] + 0.114 * dHashData[i + 2]);
  }
  let dHash = "";
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const left = dHashLum[y * 9 + x];
      const right = dHashLum[y * 9 + x + 1];
      dHash += right > left ? "1" : "0";
    }
  }

  const featureSize = 24;
  const featureCanvas = document.createElement("canvas");
  featureCanvas.width = featureSize;
  featureCanvas.height = featureSize;
  const featureCtx = featureCanvas.getContext("2d");
  featureCtx.drawImage(img, 0, 0, featureSize, featureSize);
  const featureData = featureCtx.getImageData(0, 0, featureSize, featureSize).data;
  const lumVector = [];
  const colorHist = new Array(64).fill(0);
  let edgeSum = 0;
  let lumSqSum = 0;
  const lumGrid = new Float32Array(featureSize * featureSize);

  for (let i = 0, p = 0; i < featureData.length; i += 4, p += 1) {
    const r = featureData[i];
    const g = featureData[i + 1];
    const b = featureData[i + 2];
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    lumVector.push(l / 255);
    lumGrid[p] = l;
    lumSqSum += l * l;

    const rb = Math.min(3, Math.floor(r / 64));
    const gb = Math.min(3, Math.floor(g / 64));
    const bb = Math.min(3, Math.floor(b / 64));
    colorHist[rb * 16 + gb * 4 + bb] += 1;
  }

  for (let y = 1; y < featureSize - 1; y += 1) {
    for (let x = 1; x < featureSize - 1; x += 1) {
      const p = y * featureSize + x;
      const dx = Math.abs(lumGrid[p + 1] - lumGrid[p - 1]);
      const dy = Math.abs(lumGrid[p + featureSize] - lumGrid[p - featureSize]);
      edgeSum += dx + dy;
    }
  }

  const featurePixels = featureSize * featureSize;
  const lumVariance = Math.max(0, lumSqSum / featurePixels - lumMean * lumMean);
  const contrastStd = Math.sqrt(lumVariance);
  const edgeDensity = edgeSum / ((featureSize - 2) * (featureSize - 2) * 2);
  const visibilityScore = clamp(Math.round(contrastStd * 0.7 + edgeDensity * 1.1), 0, 100);
  const histNorm = colorHist.map((v) => v / featurePixels);

  // Coarse structure vector: 8x8 average luminance blocks from 32x32 image.
  const coarseSize = 32;
  const blockCount = 8;
  const blockW = coarseSize / blockCount;
  const coarseCanvas = document.createElement("canvas");
  coarseCanvas.width = coarseSize;
  coarseCanvas.height = coarseSize;
  const coarseCtx = coarseCanvas.getContext("2d");
  coarseCtx.drawImage(img, 0, 0, coarseSize, coarseSize);
  const coarseData = coarseCtx.getImageData(0, 0, coarseSize, coarseSize).data;
  const structure = [];
  for (let by = 0; by < blockCount; by += 1) {
    for (let bx = 0; bx < blockCount; bx += 1) {
      let sum = 0;
      for (let y = 0; y < blockW; y += 1) {
        for (let x = 0; x < blockW; x += 1) {
          const px = bx * blockW + x;
          const py = by * blockW + y;
          const idx = (py * coarseSize + px) * 4;
          const l = 0.299 * coarseData[idx] + 0.587 * coarseData[idx + 1] + 0.114 * coarseData[idx + 2];
          sum += l;
        }
      }
      structure.push((sum / (blockW * blockW)) / 255);
    }
  }

  return {
    aHash,
    dHash,
    avgR: rSum / lum.length,
    avgG: gSum / lum.length,
    avgB: bSum / lum.length,
    avgSat: satSum / lum.length,
    lumMean,
    width,
    height,
    ratio,
    format: fmt,
    size: Number(file.size || 0),
    lastModified: Number(file.lastModified || 0),
    lumVector,
    colorHist: histNorm,
    structure,
    contrastStd,
    edgeDensity,
    visibilityScore
  };
}

function fingerprintSimilarity(a, b) {
  if (!a || !b) return 0;
  const aHashDist = hammingDistance(a.aHash, b.aHash) / Math.max(1, a.aHash?.length || 1);
  const dHashDist = hammingDistance(a.dHash, b.dHash) / Math.max(1, a.dHash?.length || 1);
  const hashSim = 1 - Math.min(1, (aHashDist + dHashDist) / 2);

  const colorDist =
    (Math.abs(a.avgR - b.avgR) + Math.abs(a.avgG - b.avgG) + Math.abs(a.avgB - b.avgB)) / (255 * 3);
  const colorSim = 1 - Math.min(1, colorDist);

  const satSim = 1 - Math.min(1, Math.abs(a.avgSat - b.avgSat));
  const ratioSim = 1 - Math.min(1, Math.abs(a.ratio - b.ratio) / 1.2);
  const dimScaleSim =
    1 - Math.min(1, Math.abs(Math.log((a.width * a.height + 1) / (b.width * b.height + 1))) / 2.4);
  const lumSim =
    a.lumVector && b.lumVector && a.lumVector.length === b.lumVector.length
      ? 1 -
        Math.min(
          1,
          a.lumVector.reduce((acc, v, idx) => acc + Math.abs(v - b.lumVector[idx]), 0) / a.lumVector.length
        )
      : 0;
  const histSim =
    a.colorHist && b.colorHist && a.colorHist.length === b.colorHist.length
      ? a.colorHist.reduce((acc, v, idx) => acc + Math.min(v, b.colorHist[idx]), 0)
      : 0;
  const structureSim =
    a.structure && b.structure && a.structure.length === b.structure.length
      ? 1 -
        Math.min(
          1,
          a.structure.reduce((acc, v, idx) => acc + Math.abs(v - b.structure[idx]), 0) / a.structure.length
        )
      : 0;
  const visibilitySim = 1 - Math.min(1, Math.abs((a.visibilityScore || 0) - (b.visibilityScore || 0)) / 100);

  const timeDiffMs = Math.abs((a.lastModified || 0) - (b.lastModified || 0));
  const burstBonus = timeDiffMs > 0 && timeDiffMs < 6000 ? 0.06 : 0;

  const sim =
    0.3 * hashSim +
    0.14 * colorSim +
    0.08 * satSim +
    0.08 * ratioSim +
    0.05 * dimScaleSim +
    0.19 * lumSim +
    0.13 * histSim +
    0.08 * structureSim +
    0.03 * visibilitySim +
    burstBonus;
  return clamp(sim, 0, 1);
}

function quickSkipPair(a, b) {
  if (!a || !b) return true;
  if (Math.abs(a.ratio - b.ratio) > 0.55) return true;
  const colorGap = Math.abs(a.avgR - b.avgR) + Math.abs(a.avgG - b.avgG) + Math.abs(a.avgB - b.avgB);
  const timeDiff = Math.abs((a.lastModified || 0) - (b.lastModified || 0));
  if (colorGap > 245 && timeDiff > 10 * 60 * 1000) return true;
  return false;
}

function computeThemeSignals(analysis, themeProfile, themeStrictness = 2) {
  const strictScale = clamp(Number(themeStrictness || 2), 1, 3) / 2;
  const descriptor = analysis.descriptor || {};
  const featurePairs = {
    primary_subject: String(descriptor.primary_subject || "unknown"),
    setting: String(descriptor.setting || analysis.scene || "unknown"),
    shot_type: String(descriptor.shot_type || "unknown"),
    event_role: String(descriptor.event_role || "unknown"),
    color_profile: String(descriptor.color_profile || "unknown"),
    people_presence_band: descriptor.people_presence >= 70 ? "high" : descriptor.people_presence >= 35 ? "medium" : "low",
    text_presence_band: descriptor.text_presence >= 70 ? "high" : descriptor.text_presence >= 35 ? "medium" : "low"
  };
  const parameterTokens = Object.entries(featurePairs).map(([k, v]) => `${k}:${String(v).toLowerCase()}`);
  const matchedKeywords = themeProfile.keywords.filter((kw) => parameterTokens.some((token) => token.includes(kw)));
  let themeBonus = Math.min(28 * strictScale, matchedKeywords.length * (7 * strictScale));
  let themePenalty = 0;
  let themeBlocked = false;

  if (themeProfile.peopleFocused) {
    const hasPersonSignal =
      descriptor.people_presence >= 55 || descriptor.primary_subject === "people" || descriptor.shot_type === "portrait";
    const hasOffThemeSignal =
      descriptor.screenshot_likelihood >= 55 || descriptor.document_likelihood >= 55 || descriptor.text_presence >= 75;

    if (hasPersonSignal) {
      themeBonus += 16 * strictScale;
    } else {
      themePenalty += 22 * strictScale;
    }
    if (hasOffThemeSignal) {
      themePenalty += 48 * strictScale;
      themeBlocked = strictScale >= 1;
    }
  }

  return { themeBonus, themePenalty, themeBlocked, matchedKeywords };
}

function applySmartSelection({
  analyses,
  targetCount,
  fingerprintByIndex,
  themePrompt,
  lookalikeThreshold = 0.86,
  clusterStrictness = 2,
  themeStrictness = 2,
  maxPerCluster = 2
}) {
  const strictCluster = clamp(Number(clusterStrictness || 2), 1, 3);
  const strictScale = strictCluster / 2;
  const nearLookalikeThreshold = clamp(Number(lookalikeThreshold || 0.86), 0.75, 0.95);
  const clusterEdgeThreshold = clamp(nearLookalikeThreshold - 0.04, 0.74, 0.92);
  const hardDedupeThreshold = clamp(nearLookalikeThreshold - 0.08, 0.7, 0.88);
  const maxClusterPicks = clamp(Number(maxPerCluster || 2), 1, 6);
  const clusterDiameterThreshold = clamp(clusterEdgeThreshold + 0.02, 0.78, 0.95);

  const themeProfile = buildThemeProfile(themePrompt);
  const sceneCounts = analyses.reduce((acc, item) => {
    const key = (item.scene || "unknown").trim().toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const candidates = analyses.map((item, position) => {
    const key = (item.scene || "unknown").trim().toLowerCase();
    const scenePenalty = Math.max(0, (sceneCounts[key] || 1) - 1) * 4;
    const theme = computeThemeSignals(item, themeProfile, themeStrictness);
    const baseRelevance =
      item.overall * 0.54 +
      item.quality * 0.16 +
      item.composition * 0.12 +
      item.emotion * 0.08 +
      item.uniqueness * 0.1 +
      (item.albumWorthy ? 5 : 0);

    const fp = fingerprintByIndex.get(item.index);
    const metadataPenalty = fp?.format?.includes("gif") ? 6 : 0;
    const metadataBonus = fp && fp.width >= fp.height ? 2 : 0;
    const visibilityPenalty = fp ? Math.max(0, 28 - Number(fp.visibilityScore || 0)) * 0.8 : 0;
    const visibilityBonus = fp ? Math.max(0, Number(fp.visibilityScore || 0) - 48) * 0.12 : 0;
    return {
      ...item,
      candidateIdx: position,
      sceneKey: key,
      scenePenalty,
      duplicatePenalty: 0,
      lookalikePenalty: 0,
      metadataPenalty,
      metadataBonus,
      visibilityPenalty,
      visibilityBonus,
      baseRelevance,
      adjustedBase:
        baseRelevance -
        scenePenalty -
        metadataPenalty +
        metadataBonus -
        visibilityPenalty +
        visibilityBonus +
        theme.themeBonus -
        theme.themePenalty,
      ...theme
    };
  });

  const n = candidates.length;
  const simMatrix = Array.from({ length: n }, () => new Float32Array(n));
  const lookalikeCounts = new Array(n).fill(0);
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };

  for (let i = 0; i < n; i += 1) {
    simMatrix[i][i] = 1;
    for (let j = i + 1; j < n; j += 1) {
      const aFp = fingerprintByIndex.get(candidates[i].index);
      const bFp = fingerprintByIndex.get(candidates[j].index);
      let sim = 0;
      if (!quickSkipPair(aFp, bFp)) {
        sim = fingerprintSimilarity(aFp, bFp);
      }
      simMatrix[i][j] = sim;
      simMatrix[j][i] = sim;

      if (sim >= nearLookalikeThreshold - 0.04) {
        lookalikeCounts[i] += 1;
        lookalikeCounts[j] += 1;
      }
      const timeDiff = Math.abs((aFp?.lastModified || 0) - (bFp?.lastModified || 0));
      const sameMoment = timeDiff > 0 && timeDiff <= 20000;
      const sameScene = candidates[i].sceneKey === candidates[j].sceneKey;
      const shouldMerge =
        sim >= 0.96 ||
        (sim >= 0.9 && sameMoment) ||
        (sim >= 0.93 && sameScene && timeDiff > 0 && timeDiff <= 120000) ||
        (sim >= clusterEdgeThreshold + 0.03 && sameMoment);
      if (shouldMerge) {
        union(i, j);
      }
    }
  }

  const scored = candidates.map((item) => {
    const localLookalikePenalty = Math.min(40, lookalikeCounts[item.candidateIdx] * (1.6 + strictScale * 0.7));
    const adjustedBase = item.adjustedBase - item.duplicatePenalty - localLookalikePenalty;
    return { ...item, lookalikePenalty: localLookalikePenalty, adjustedBase };
  });

  const clustersMap = new Map();
  for (const item of scored) {
    const cid = find(item.candidateIdx);
    if (!clustersMap.has(cid)) clustersMap.set(cid, []);
    clustersMap.get(cid).push(item);
  }
  const splitByDiameter = (clusterItems) => {
    const rankedItems = [...clusterItems].sort((a, b) => b.adjustedBase - a.adjustedBase);
    const groups = [];
    for (const item of rankedItems) {
      let placed = false;
      for (const group of groups) {
        const minSim = group.reduce((min, gItem) => Math.min(min, simMatrix[item.candidateIdx][gItem.candidateIdx]), 1);
        if (minSim >= clusterDiameterThreshold) {
          group.push(item);
          placed = true;
          break;
        }
      }
      if (!placed) groups.push([item]);
    }
    return groups;
  };

  const clusters = Array.from(clustersMap.values())
    .flatMap((clusterItems) => splitByDiameter(clusterItems))
    .map((clusterItems, clusterIdx) => {
      const ranked = [...clusterItems].sort((a, b) => b.adjustedBase - a.adjustedBase);
      return {
        id: clusterIdx,
        items: ranked,
        champion: ranked[0],
        size: ranked.length
      };
    });

  const byIndex = new Map();
  for (const cluster of clusters) {
    cluster.items.forEach((item, rank) => {
      byIndex.set(item.index, {
        clusterId: cluster.id,
        clusterSize: cluster.size,
        intraClusterRank: rank,
        champion: rank === 0
      });
    });
  }

  const selectable = scored.filter((c) => !c.themeBlocked || scored.length < targetCount * 1.4);
  const allowed = new Set(selectable.map((s) => s.index));
  const champions = clusters
    .map((c) => c.champion)
    .filter((c) => allowed.has(c.index))
    .sort((a, b) => b.adjustedBase - a.adjustedBase);

  const nonChampions = [];
  for (const cluster of clusters) {
    for (let i = 1; i < cluster.items.length; i += 1) {
      const candidate = cluster.items[i];
      if (allowed.has(candidate.index)) nonChampions.push(candidate);
    }
  }
  nonChampions.sort((a, b) => b.adjustedBase - a.adjustedBase);

  const selected = [];
  const selectedSet = new Set();
  const scenePicked = {};
  const peoplePicked = new Set();
  const clusterPickedCounts = {};
  const targetSceneQuota = Math.max(1, Math.round(targetCount / Math.max(1, Object.keys(sceneCounts).length)));

  const pickGreedyMMR = (pool, strictUniq) => {
    let best = null;
    let bestScore = -Infinity;
    for (const candidate of pool) {
      if (selectedSet.has(candidate.index)) continue;
      const meta = byIndex.get(candidate.index);
      const clusterAlreadyPicked = clusterPickedCounts[meta?.clusterId] || 0;
      if (clusterAlreadyPicked >= maxClusterPicks) continue;

      let maxSimToSelected = 0;
      let nearDupCount = 0;
      let burstPenalty = 0;
      for (const chosen of selected) {
        const sim = simMatrix[candidate.candidateIdx][chosen.candidateIdx];
        if (sim > maxSimToSelected) maxSimToSelected = sim;
        if (sim >= nearLookalikeThreshold) nearDupCount += 1;
        const cFp = fingerprintByIndex.get(candidate.index);
        const sFp = fingerprintByIndex.get(chosen.index);
        const tDiff = Math.abs((cFp?.lastModified || 0) - (sFp?.lastModified || 0));
        if (tDiff > 0 && tDiff < 3000) burstPenalty = Math.max(burstPenalty, 8);
      }

      if (strictUniq && (maxSimToSelected >= nearLookalikeThreshold || nearDupCount >= 1)) {
        continue;
      }

      const sceneOverQuota = Math.max(0, (scenePicked[candidate.sceneKey] || 0) - targetSceneQuota + 1);
      const scenePenaltyDynamic = sceneOverQuota * (5 + strictScale * 2);
      const clusterPenalty =
        meta?.intraClusterRank > 0
          ? meta.intraClusterRank * (8 + strictScale * 3) + clusterAlreadyPicked * (10 + strictScale * 5)
          : clusterAlreadyPicked * (8 + strictScale * 4);
      const mmrPenalty = maxSimToSelected * (52 + strictScale * 18) + nearDupCount * (14 + strictScale * 6);
      const peopleThemeBonus =
        themeProfile.peopleFocused &&
        (candidate.descriptor?.people_presence >= 55 ||
          candidate.descriptor?.primary_subject === "people" ||
          candidate.descriptor?.shot_type === "portrait")
          ? 6
          : 0;
      const noveltyBonus = themeProfile.peopleFocused ? (peoplePicked.size < 4 ? 2 : 0) : 0;

      const score = candidate.adjustedBase + peopleThemeBonus + noveltyBonus - scenePenaltyDynamic - clusterPenalty - mmrPenalty - burstPenalty;
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    if (best) {
      selected.push(best);
      selectedSet.add(best.index);
      scenePicked[best.sceneKey] = (scenePicked[best.sceneKey] || 0) + 1;
      const meta = byIndex.get(best.index);
      clusterPickedCounts[meta?.clusterId] = (clusterPickedCounts[meta?.clusterId] || 0) + 1;
      if (best.descriptor?.primary_subject === "people") {
        peoplePicked.add("people");
      }
      if (best.descriptor?.shot_type === "portrait") {
        peoplePicked.add("portrait");
      }
      return true;
    }
    return false;
  };

  // Stage 1: pick cluster champions only (one per cluster, global diversity).
  while (selected.length < targetCount && selected.length < champions.length) {
    if (!pickGreedyMMR(champions, true)) break;
  }
  // Stage 2: fill from all candidates, but still strict on lookalikes.
  while (selected.length < targetCount && selected.length < selectable.length) {
    if (!pickGreedyMMR([...champions, ...nonChampions], true)) break;
  }
  // Stage 3: relaxed fill only if needed.
  while (selected.length < targetCount && selected.length < selectable.length) {
    if (!pickGreedyMMR([...champions, ...nonChampions], false)) break;
  }

  // Final strict post-process: eliminate lookalikes and low-visibility picks.
  const postProcessed = [];
  const sortedSelected = [...selected].sort((a, b) => b.adjustedBase - a.adjustedBase);
  const postProcessSimilarityThreshold = hardDedupeThreshold;
  const hardMinVisibility = 22;
  let lowVisibilityFiltered = 0;
  for (const candidate of sortedSelected) {
    const fpCandidate = fingerprintByIndex.get(candidate.index);
    if (fpCandidate && fpCandidate.visibilityScore < hardMinVisibility) {
      lowVisibilityFiltered += 1;
      continue;
    }
    const tooSimilar = postProcessed.some(
      (kept) => simMatrix[candidate.candidateIdx][kept.candidateIdx] >= postProcessSimilarityThreshold
    );
    if (!tooSimilar) {
      postProcessed.push(candidate);
    }
  }

  if (postProcessed.length < targetCount) {
    const refillPool = [...champions, ...nonChampions].sort((a, b) => b.adjustedBase - a.adjustedBase);
    for (const candidate of refillPool) {
      if (postProcessed.length >= targetCount) break;
      if (postProcessed.some((x) => x.index === candidate.index)) continue;
      const fpCandidate = fingerprintByIndex.get(candidate.index);
      if (fpCandidate && fpCandidate.visibilityScore < hardMinVisibility) {
        lowVisibilityFiltered += 1;
        continue;
      }
      const tooSimilar = postProcessed.some(
        (kept) => simMatrix[candidate.candidateIdx][kept.candidateIdx] >= postProcessSimilarityThreshold
      );
      if (!tooSimilar) {
        postProcessed.push(candidate);
      }
    }
  }

  const selectedIndexes = postProcessed.slice(0, targetCount).map((s) => s.index);
  const selectedPostSet = new Set(selectedIndexes);
  const finalScored = scored.map((item) => {
    const isPicked = selectedPostSet.has(item.index);
    const meta = byIndex.get(item.index) || {};
    const clusterPenalty =
      meta?.intraClusterRank > 0 ? meta.intraClusterRank * (5 + strictScale * 2) + (meta.clusterSize || 1) * 1.2 : 0;
    const adjustedOverall = clamp(Math.round(item.adjustedBase - clusterPenalty - (isPicked ? 0 : 4)), 0, 100);
    return {
      ...item,
      adjustedOverall,
      mmrSelected: isPicked,
      clusterId: meta.clusterId,
      clusterSize: meta.clusterSize || 1,
      intraClusterRank: meta.intraClusterRank || 0,
      clusterChampion: Boolean(meta.champion)
    };
  });

  const selectedRows = finalScored.filter((row) => selectedPostSet.has(row.index));
  const clusterSelectedCounts = finalScored.reduce((acc, row) => {
    if (!selectedPostSet.has(row.index)) return acc;
    const key = row.clusterId;
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const clustersWithMultiSelected = Object.values(clusterSelectedCounts).filter((count) => count > 1).length;
  const selectedLowVisibility = selectedRows.filter((row) => (row.visibilityScore || 0) < 30).length;
  const selectedNonChampion = selectedRows.filter((row) => !row.clusterChampion).length;

  const clusterSummaries = clusters
    .map((cluster) => {
      const ranked = [...cluster.items].sort((a, b) => b.adjustedBase - a.adjustedBase);
      return {
        clusterId: cluster.id,
        size: cluster.size,
        selectedCount: ranked.filter((item) => selectedPostSet.has(item.index)).length,
        topIndexes: ranked.slice(0, 4).map((item) => item.index)
      };
    })
    .sort((a, b) => b.size - a.size);

  return {
    selectedIndexes,
    scored: finalScored,
    stats: {
      duplicateSuppressed: finalScored.filter((c) => c.clusterSize > 1 && c.intraClusterRank > 0).length,
      blockedByTheme: scored.filter((c) => c.themeBlocked).length,
      clusters: clusters.length,
      lookalikeThreshold: nearLookalikeThreshold,
      postProcessRemoved: Math.max(0, selected.length - selectedIndexes.length),
      postProcessThreshold: postProcessSimilarityThreshold,
      lowVisibilityFiltered,
      selectedLowVisibility,
      selectedNonChampion,
      clustersWithMultiSelected,
      maxPerCluster: maxClusterPicks
    },
    diagnostics: {
      clusters: clusterSummaries
    }
  };
}

function buildPrompt(theme) {
  const themeLine = theme?.trim()
    ? `Album theme preference: ${theme.trim()}.`
    : "No specific album theme was provided.";

  return [
    "Analyze each image and return only a raw JSON array (no markdown, no commentary).",
    "Return one object per image with fields: index, quality, composition, emotion, uniqueness, overall, scene, tags, descriptor, reason, albumWorthy.",
    "descriptor must be key:value only with keys: primary_subject, setting, shot_type, event_role, color_profile, people_presence, text_presence, screenshot_likelihood, document_likelihood.",
    "Scoring is 0-100 for quality, composition, emotion, uniqueness, and overall.",
    "Use theme relevance strongly in scoring. Off-theme images should score lower.",
    "If the theme is people/family/portraits, penalize documents, screenshots, and text-heavy images.",
    "tags must be an array of short strings.",
    "albumWorthy must be boolean.",
    themeLine
  ].join(" ");
}

async function runCloudflareBatch({ photos, themePrompt, appendLog, endpoint, proxyToken, model, signal }) {
  const resolvedEndpoint = normalizeEndpointUrl(endpoint);
  if (!isAbsoluteHttpUrl(resolvedEndpoint)) {
    throw new Error(
      `Cloudflare proxy endpoint must be a full URL (http/https). Example: https://your-worker.workers.dev`
    );
  }

  const prompt = buildPrompt(themePrompt);

  const content = [{ type: "text", text: prompt }];

  for (const photo of photos) {
    throwIfAborted(signal);
    const b64 = await toBase64JpegThumbnail(photo.file, 512);
    content.push({
      type: "image",
      source: {
        type: "base64",
        media_type: "image/jpeg",
        data: b64
      }
    });
  }

  const headers = {
    "Content-Type": "application/json"
  };
  if (proxyToken?.trim()) {
    headers.Authorization = `Bearer ${proxyToken.trim()}`;
  }

  let response;
  try {
    response = await fetch(resolvedEndpoint, {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify({
        provider: "cloudflare",
        model: model?.trim() || DEFAULT_CF_MODEL,
        messages: [
          {
            role: "user",
            content
          }
        ]
      })
    });
  } catch (error) {
    throw new Error(
      `Proxy request failed (${errorToMessage(error)}). Check endpoint URL and CORS on your Worker (Access-Control-Allow-Origin).`
    );
  }

  const rawText = await response.text();

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(
        `Proxy endpoint returned 404: ${resolvedEndpoint}. Use the full Worker URL and ensure that route is deployed.`
      );
    }
    throw new Error(`API ${response.status} from ${resolvedEndpoint}: ${rawText}`);
  }

  let parsed;
  try {
    const maybeJson = JSON.parse(rawText);
    const textFromPayload = maybeJson?.content?.map((c) => c?.text || "").join("\n") || maybeJson?.text || rawText;
    parsed = extractFirstJsonArray(textFromPayload);
  } catch {
    parsed = extractFirstJsonArray(rawText);
  }

  appendLog("success", `Cloudflare batch analyzed (${photos.length} images).`);
  return parsed;
}

async function runOpenAIProxyBatch({ photos, themePrompt, appendLog, endpoint, proxyToken, model, signal }) {
  const resolvedEndpoint = normalizeEndpointUrl(endpoint);
  if (!isAbsoluteHttpUrl(resolvedEndpoint)) {
    throw new Error(
      `OpenAI proxy endpoint must be a full URL (http/https). Example: https://your-worker.workers.dev`
    );
  }

  const prompt = buildPrompt(themePrompt);
  const content = [{ type: "text", text: prompt }];

  for (const photo of photos) {
    throwIfAborted(signal);
    const b64 = await toBase64JpegThumbnail(photo.file, 512);
    content.push({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${b64}`
      }
    });
  }

  const headers = {
    "Content-Type": "application/json"
  };
  if (proxyToken?.trim()) {
    headers.Authorization = `Bearer ${proxyToken.trim()}`;
  }

  let response;
  try {
    response = await fetch(resolvedEndpoint, {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify({
        provider: "openai",
        model: model?.trim() || DEFAULT_OPENAI_MODEL,
        messages: [
          {
            role: "user",
            content
          }
        ]
      })
    });
  } catch (error) {
    throw new Error(
      `OpenAI proxy request failed (${errorToMessage(error)}). Check endpoint URL and CORS on your Worker (Access-Control-Allow-Origin).`
    );
  }

  const rawText = await response.text();

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(
        `Proxy endpoint returned 404: ${resolvedEndpoint}. Use the full Worker URL and ensure that route is deployed.`
      );
    }
    throw new Error(`API ${response.status} from ${resolvedEndpoint}: ${rawText}`);
  }

  let parsed;
  try {
    const maybeJson = JSON.parse(rawText);
    const contentOut = maybeJson?.choices?.[0]?.message?.content;
    const textFromPayload =
      typeof contentOut === "string"
        ? contentOut
        : Array.isArray(contentOut)
          ? contentOut.map((part) => part?.text || "").join("\n")
          : maybeJson?.text || rawText;
    parsed = extractFirstJsonArray(textFromPayload);
  } catch {
    parsed = extractFirstJsonArray(rawText);
  }

  appendLog("success", `OpenAI proxy batch analyzed (${photos.length} images).`);
  return parsed;
}

async function runCloudflareDirectBatch({ photos, themePrompt, appendLog, accountId, apiKey, model, signal }) {
  if (!accountId?.trim()) {
    throw new Error("Cloudflare account ID is required for direct mode.");
  }
  if (!apiKey?.trim()) {
    throw new Error("Cloudflare API key/token is required for direct mode.");
  }

  const prompt = buildPrompt(themePrompt);
  const content = [{ type: "text", text: prompt }];
  for (const photo of photos) {
    throwIfAborted(signal);
    const b64 = await toBase64JpegThumbnail(photo.file, 512);
    content.push({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${b64}`
      }
    });
  }

  const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${accountId.trim()}/ai/run/${(model || DEFAULT_CF_MODEL).trim()}`;
  let response;
  try {
    response = await fetch(cfUrl, {
      method: "POST",
      signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey.trim()}`
      },
      body: JSON.stringify({
        messages: [
          {
            role: "user",
            content
          }
        ],
        max_tokens: 1600
      })
    });
  } catch (error) {
    const message = errorToMessage(error);
    throw new Error(
      `Cloudflare Direct request failed (${message}). This is usually browser CORS blocking direct calls to api.cloudflare.com. Use Cloudflare endpoint (proxy/worker) mode instead.`
    );
  }

  const rawText = await response.text();
  if (!response.ok) {
    throw new Error(`Cloudflare Direct API ${response.status}: ${rawText}`);
  }

  let parsed;
  try {
    const maybeJson = JSON.parse(rawText);
    const textFromPayload = maybeJson?.result?.response || maybeJson?.response || maybeJson?.result?.output_text || maybeJson?.text || rawText;
    parsed = extractFirstJsonArray(textFromPayload);
  } catch {
    parsed = extractFirstJsonArray(rawText);
  }

  appendLog("success", `Cloudflare Direct batch analyzed (${photos.length} images).`);
  return parsed;
}

async function runOpenAIDirectBatch({ photos, themePrompt, appendLog, apiKey, model, baseUrl, signal }) {
  if (!apiKey?.trim()) {
    throw new Error("OpenAI API key is required for OpenAI direct mode.");
  }

  const prompt = buildPrompt(themePrompt);
  const content = [{ type: "text", text: prompt }];
  for (const photo of photos) {
    throwIfAborted(signal);
    const b64 = await toBase64JpegThumbnail(photo.file, 512);
    content.push({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${b64}`
      }
    });
  }

  const endpoint = normalizeEndpointUrl(baseUrl || DEFAULT_OPENAI_BASE_URL);
  if (!isAbsoluteHttpUrl(endpoint)) {
    throw new Error(`OpenAI base URL must be a full URL (http/https).`);
  }

  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey.trim()}`
      },
      body: JSON.stringify({
        model: (model || DEFAULT_OPENAI_MODEL).trim(),
        messages: [
          {
            role: "user",
            content
          }
        ],
        temperature: 0.2
      })
    });
  } catch (error) {
    const message = errorToMessage(error);
    throw new Error(
      `OpenAI direct request failed (${message}). This can be caused by CORS restrictions when calling from browser.`
    );
  }

  const rawText = await response.text();
  if (!response.ok) {
    throw new Error(`OpenAI API ${response.status}: ${rawText}`);
  }

  let parsed;
  try {
    const maybeJson = JSON.parse(rawText);
    const contentOut = maybeJson?.choices?.[0]?.message?.content;
    const textFromPayload =
      typeof contentOut === "string"
        ? contentOut
        : Array.isArray(contentOut)
          ? contentOut.map((part) => part?.text || "").join("\n")
          : maybeJson?.text || rawText;
    parsed = extractFirstJsonArray(textFromPayload);
  } catch {
    parsed = extractFirstJsonArray(rawText);
  }

  appendLog("success", `OpenAI Direct batch analyzed (${photos.length} images).`);
  return parsed;
}

async function buildImageMetrics(file) {
  const img = await fileToImageBitmap(file);
  const sample = 64;
  const canvas = document.createElement("canvas");
  canvas.width = sample;
  canvas.height = sample;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, sample, sample);
  const { data } = ctx.getImageData(0, 0, sample, sample);

  let lumSum = 0;
  let lumSqSum = 0;
  let satSum = 0;
  let warmSum = 0;
  let edgeSum = 0;
  const lums = new Float32Array(sample * sample);

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const sat = max === 0 ? 0 : ((max - min) / max) * 100;
    lumSum += lum;
    lumSqSum += lum * lum;
    satSum += sat;
    warmSum += (r - b) / 255;
    lums[p] = lum;
  }

  for (let y = 1; y < sample - 1; y += 1) {
    for (let x = 1; x < sample - 1; x += 1) {
      const p = y * sample + x;
      const dx = Math.abs(lums[p + 1] - lums[p - 1]);
      const dy = Math.abs(lums[p + sample] - lums[p - sample]);
      edgeSum += dx + dy;
    }
  }

  const pixels = sample * sample;
  const avgLum = lumSum / pixels;
  const variance = Math.max(0, lumSqSum / pixels - avgLum * avgLum);
  const contrast = Math.sqrt(variance);
  const saturation = satSum / pixels;
  const warmth = warmSum / pixels;
  const sharpness = edgeSum / ((sample - 2) * (sample - 2) * 2);

  return { avgLum, contrast, saturation, warmth, sharpness, width: img.width, height: img.height };
}

function pickSceneFromMetrics(metrics) {
  const { width, height, avgLum } = metrics;
  if (height > width * 1.1) return "portrait";
  if (width > height * 1.25 && avgLum > 120) return "landscape";
  if (avgLum < 75) return "night";
  if (avgLum > 150) return "outdoor";
  return "indoor";
}

function makeTags(metrics, scene, themePrompt) {
  const tags = [scene];
  if (metrics.saturation > 45) tags.push("vivid");
  if (metrics.saturation < 20) tags.push("muted");
  if (metrics.sharpness > 20) tags.push("crisp");
  if (metrics.sharpness < 9) tags.push("soft");
  if (metrics.warmth > 0.08) tags.push("warm");
  if (metrics.warmth < -0.08) tags.push("cool");
  if (metrics.avgLum > 155) tags.push("bright");
  if (metrics.avgLum < 85) tags.push("low-light");
  if (metrics.avgLum > 170 && metrics.saturation < 12 && metrics.contrast < 45) tags.push("document-like");
  if (metrics.saturation < 10 && metrics.sharpness > 14 && metrics.contrast < 38) tags.push("screenshot-like");
  if (scene === "portrait" && metrics.sharpness > 10) tags.push("person-like");
  if (themePrompt?.trim()) {
    const keywords = themePrompt
      .toLowerCase()
      .split(/[^a-z0-9]+/g)
      .filter((word) => word.length >= 4)
      .slice(0, 2);
    tags.push(...keywords);
  }
  return Array.from(new Set(tags)).slice(0, 6);
}

async function runLocalBatch({ photos, themePrompt, appendLog, signal }) {
  const out = [];

  for (let i = 0; i < photos.length; i += 1) {
    throwIfAborted(signal);
    const photo = photos[i];
    const metrics = await buildImageMetrics(photo.file);
    const scene = pickSceneFromMetrics(metrics);
    const noise = hashStringToUnit(`${photo.name}-${photo.file.size}`);

    const quality = clamp(Math.round(40 + metrics.contrast * 0.9 + metrics.sharpness * 1.3 + noise * 18), 0, 100);
    const composition = clamp(Math.round(50 + (metrics.width > metrics.height ? 8 : 4) + metrics.contrast * 0.5 + noise * 14), 0, 100);
    const emotion = clamp(Math.round(45 + metrics.saturation * 0.6 + metrics.warmth * 28 + noise * 12), 0, 100);
    const uniqueness = clamp(Math.round(38 + (metrics.saturation > 42 ? 16 : 6) + noise * 36), 0, 100);
    const overall = clamp(Math.round(quality * 0.33 + composition * 0.27 + emotion * 0.2 + uniqueness * 0.2), 0, 100);
    const tags = makeTags(metrics, scene, themePrompt);
    const descriptor = normalizeDescriptor(
      {
        primary_subject:
          tags.includes("person-like") ? "people" : tags.includes("document-like") ? "document" : tags.includes("screenshot-like") ? "screenshot" : "object",
        setting: scene,
        shot_type: scene === "portrait" ? "portrait" : metrics.width > metrics.height * 1.25 ? "wide" : "medium",
        event_role: overall >= 82 ? "hero" : overall >= 68 ? "context" : "detail",
        color_profile: tags.includes("warm") ? "warm" : tags.includes("cool") ? "cool" : "neutral",
        people_presence: tags.includes("person-like") ? 80 : 15,
        text_presence: tags.includes("document-like") || tags.includes("screenshot-like") ? 85 : 10,
        screenshot_likelihood: tags.includes("screenshot-like") ? 88 : 12,
        document_likelihood: tags.includes("document-like") ? 88 : 12
      },
      scene,
      tags
    );
    const reason = `Local estimate: ${scene} scene, contrast ${metrics.contrast.toFixed(1)}, sharpness ${metrics.sharpness.toFixed(1)}, saturation ${metrics.saturation.toFixed(1)}.`;

    out.push({
      index: i,
      quality,
      composition,
      emotion,
      uniqueness,
      overall,
      scene,
      tags,
      descriptor,
      reason,
      albumWorthy: overall >= 70
    });
  }

  appendLog("success", `Local batch analyzed (${photos.length} images).`);
  return out;
}

function ScoreBar({ label, value }) {
  return (
    <div className="score-row">
      <span>{label}</span>
      <div className="bar-wrap">
        <div className="bar-fill" style={{ width: `${clamp(value || 0, 0, 100)}%` }} />
      </div>
      <strong>{Math.round(value || 0)}</strong>
    </div>
  );
}

export default function App() {
  const [photos, setPhotos] = useState([]);
  const [albumSize, setAlbumSize] = useState(MIN_ALBUM);
  const [themePrompt, setThemePrompt] = useState("");
  const [isPreparingFiles, setIsPreparingFiles] = useState(false);
  const [filePrepProgress, setFilePrepProgress] = useState(0);
  const [filePrepLabel, setFilePrepLabel] = useState("");
  const [currentStep, setCurrentStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisProgressLabel, setAnalysisProgressLabel] = useState("");
  const [logs, setLogs] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [selectedIds, setSelectedIds] = useState([]);
  const [clusterReport, setClusterReport] = useState([]);
  const [openClusterIds, setOpenClusterIds] = useState([]);
  const [diagnosticInsights, setDiagnosticInsights] = useState([]);
  const [activePhotoId, setActivePhotoId] = useState(null);
  const [lookalikeThreshold, setLookalikeThreshold] = useState(() => Number(localStorage.getItem("picker_lookalike_threshold") || 0.86));
  const [clusterStrictness, setClusterStrictness] = useState(() => Number(localStorage.getItem("picker_cluster_strictness") || 2));
  const [themeStrictness, setThemeStrictness] = useState(() => Number(localStorage.getItem("picker_theme_strictness") || 2));
  const [maxPerCluster, setMaxPerCluster] = useState(() => Number(localStorage.getItem("picker_max_per_cluster") || 2));
  const [provider, setProvider] = useState(
    DEFAULT_PROVIDER === "cloudflare" ||
      DEFAULT_PROVIDER === "openai" ||
      DEFAULT_PROVIDER === "cloudflare-direct" ||
      DEFAULT_PROVIDER === "openai-direct"
      ? DEFAULT_PROVIDER
      : "local"
  );
  const [cfProxyEndpoint, setCfProxyEndpoint] = useState(() =>
    localStorage.getItem("cf_proxy_endpoint") || normalizeEndpointUrl(CLOUDFLARE_ENDPOINT)
  );
  const [cfProxyToken, setCfProxyToken] = useState(() => localStorage.getItem("cf_proxy_token") || "");
  const [cfProxyModel, setCfProxyModel] = useState(() => localStorage.getItem("cf_proxy_model") || DEFAULT_CF_MODEL);
  const [cfAccountId, setCfAccountId] = useState(() => localStorage.getItem("cf_account_id") || "");
  const [cfApiKey, setCfApiKey] = useState(() => localStorage.getItem("cf_api_key") || "");
  const [cfModel, setCfModel] = useState(() => localStorage.getItem("cf_model") || DEFAULT_CF_MODEL);
  const [openaiApiKey, setOpenaiApiKey] = useState(() => localStorage.getItem("openai_api_key") || "");
  const [openaiModel, setOpenaiModel] = useState(() => localStorage.getItem("openai_model") || DEFAULT_OPENAI_MODEL);
  const [openaiBaseUrl, setOpenaiBaseUrl] = useState(
    () => localStorage.getItem("openai_base_url") || DEFAULT_OPENAI_BASE_URL
  );
  const inputRef = useRef(null);
  const analysisAbortRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("cf_proxy_endpoint", cfProxyEndpoint);
  }, [cfProxyEndpoint]);
  useEffect(() => {
    localStorage.setItem("cf_proxy_token", cfProxyToken);
  }, [cfProxyToken]);
  useEffect(() => {
    localStorage.setItem("cf_proxy_model", cfProxyModel);
  }, [cfProxyModel]);
  useEffect(() => {
    localStorage.setItem("cf_account_id", cfAccountId);
  }, [cfAccountId]);
  useEffect(() => {
    localStorage.setItem("cf_api_key", cfApiKey);
  }, [cfApiKey]);
  useEffect(() => {
    localStorage.setItem("cf_model", cfModel);
  }, [cfModel]);
  useEffect(() => {
    localStorage.setItem("openai_api_key", openaiApiKey);
  }, [openaiApiKey]);
  useEffect(() => {
    localStorage.setItem("openai_model", openaiModel);
  }, [openaiModel]);
  useEffect(() => {
    localStorage.setItem("openai_base_url", openaiBaseUrl);
  }, [openaiBaseUrl]);
  useEffect(() => {
    localStorage.setItem("picker_lookalike_threshold", String(lookalikeThreshold));
  }, [lookalikeThreshold]);
  useEffect(() => {
    localStorage.setItem("picker_cluster_strictness", String(clusterStrictness));
  }, [clusterStrictness]);
  useEffect(() => {
    localStorage.setItem("picker_theme_strictness", String(themeStrictness));
  }, [themeStrictness]);
  useEffect(() => {
    localStorage.setItem("picker_max_per_cluster", String(maxPerCluster));
  }, [maxPerCluster]);

  const activePhotos = useMemo(() => photos.filter((p) => !p.removed), [photos]);

  const canAdvanceToConfigure = activePhotos.length > 0 && !isPreparingFiles;
  const canShowResults = selectedIds.length > 0 || isAnalyzing;

  const stats = useMemo(() => {
    const selected = activePhotos.filter((p) => selectedIds.includes(p.id));
    const avg = average(selected.map((p) => p.analysis?.adjustedOverall || 0));
    return {
      selected: selected.length,
      uploaded: activePhotos.length,
      avg,
      rate: activePhotos.length ? (selected.length / activePhotos.length) * 100 : 0
    };
  }, [activePhotos, selectedIds]);

  const activePhoto = photos.find((p) => p.id === activePhotoId) || null;
  const lookalikeHint =
    lookalikeThreshold <= 0.8
      ? "Loose: keeps more similar shots. Example: burst photos with slight pose changes can both stay."
      : lookalikeThreshold <= 0.89
        ? "Balanced: removes most near-duplicates. Example: very similar frames usually collapse to one."
        : "Strict: aggressively keeps only distinct frames. Example: only one image from a burst is likely kept.";
  const clusterHint =
    clusterStrictness === 1
      ? "Loose: allows more from the same moment. Example: two angles from one scene may both be selected."
      : clusterStrictness === 2
        ? "Balanced: usually one champion per moment, plus occasional extras."
        : "Strict: favors one best per cluster. Example: near-identical scene variants are strongly filtered.";
  const themeHint =
    themeStrictness === 1
      ? "Soft: theme guides ranking lightly. Example: off-theme but strong photos can still pass."
      : themeStrictness === 2
        ? "Balanced: theme has clear impact while keeping flexibility."
        : "Hard: theme is enforced strongly. Example: people theme heavily removes screenshots/documents.";
  const maxPerClusterHint =
    maxPerCluster <= 1
      ? "Very strict: usually only one photo from each cluster/moment."
      : maxPerCluster === 2
        ? "Strict: up to two photos per cluster if they are meaningfully different."
        : "Flexible: allows more from the same cluster when album size is large.";

  useEffect(() => {
    setOpenClusterIds([]);
  }, [clusterReport]);

  useEffect(() => {
    if (!activePhotos.length && currentStep !== 1) {
      setCurrentStep(1);
    } else if (currentStep === 3 && !canShowResults) {
      setCurrentStep(canAdvanceToConfigure ? 2 : 1);
    }
  }, [activePhotos.length, canAdvanceToConfigure, canShowResults, currentStep]);

  function appendLog(level, message) {
    setLogs((prev) => [...prev, `[${nowStamp()}] ${level.toUpperCase()}: ${message}`]);
  }

  async function addFiles(fileList) {
    setErrorMessage("");
    const incoming = Array.from(fileList || []);
    const imageFiles = incoming.filter(isSupportedImage);

    if (imageFiles.length !== incoming.length) {
      appendLog("error", "Some files were skipped because they are not recognized image formats.");
    }

    if (!imageFiles.length) {
      setErrorMessage("No supported image files found in your selection.");
      return;
    }

    setIsPreparingFiles(true);
    setFilePrepProgress(0);
    setFilePrepLabel(`Preparing 0/${imageFiles.length} images...`);

    try {
      const preparedFiles = [];
      for (let i = 0; i < imageFiles.length; i += 1) {
        const file = imageFiles[i];
        setFilePrepLabel(`Preparing ${i + 1}/${imageFiles.length}: ${file.name}`);

        if (needsHeicConversion(file)) {
          try {
            const converted = await convertHeicToJpeg(file);
            preparedFiles.push(converted);
            appendLog("info", `Converted HEIC/HEIF to JPEG: ${file.name}`);
          } catch (error) {
            const message = errorToMessage(error);
            if (message.includes("Image is already browser readable")) {
              preparedFiles.push(file);
              appendLog("info", `Skipped HEIC conversion for ${file.name} because browser can already decode it.`);
            } else {
              appendLog("error", `HEIC conversion failed for ${file.name}: ${message}`);
              setErrorMessage(`HEIC conversion failed for ${file.name}: ${message}`);
            }
          }
        } else {
          preparedFiles.push(file);
        }

        setFilePrepProgress(Math.round(((i + 1) / imageFiles.length) * 100));
      }

      if (!preparedFiles.length) {
        setErrorMessage("No supported image files could be prepared for upload.");
        return;
      }

      setFilePrepLabel("Finalizing image import...");
      setPhotos((prev) => {
        const cap = MAX_UPLOADS - prev.length;
        if (cap <= 0) {
          setErrorMessage(`Upload limit reached (${MAX_UPLOADS} photos).`);
          return prev;
        }

        const nextFiles = preparedFiles.slice(0, cap).map((file) => ({
          id: uid(),
          file,
          name: file.name,
          previewUrl: URL.createObjectURL(file),
          analysis: null,
          removed: false
        }));

        if (preparedFiles.length > cap) {
          setErrorMessage(`Only ${cap} additional photos were added (max ${MAX_UPLOADS}).`);
        }

        appendLog("info", `Added ${nextFiles.length} image(s).`);
        return [...prev, ...nextFiles];
      });
    } finally {
      setTimeout(() => {
        setIsPreparingFiles(false);
        setFilePrepProgress(0);
        setFilePrepLabel("");
      }, 500);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    void addFiles(e.dataTransfer.files);
  }

  function onPickFiles(e) {
    void addFiles(e.target.files);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function removePhoto(id) {
    setPhotos((prev) => prev.map((p) => (p.id === id ? { ...p, removed: true } : p)));
    setSelectedIds((prev) => prev.filter((item) => item !== id));
  }

  function stopAnalysis() {
    if (analysisAbortRef.current) {
      analysisAbortRef.current.abort();
      appendLog("info", "Stop requested. Cancelling in-flight analysis calls...");
    }
  }

  function toggleCluster(clusterId) {
    setOpenClusterIds((prev) => (prev.includes(clusterId) ? prev.filter((id) => id !== clusterId) : [...prev, clusterId]));
  }

  async function startAnalysis() {
    if (!activePhotos.length) {
      setErrorMessage("Upload at least one image before analyzing.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage("");
    setLogs([]);
    setClusterReport([]);
    setDiagnosticInsights([]);
    setAnalysisProgress(0);
    setAnalysisProgressLabel("Preparing analysis...");
    appendLog("info", `Starting ${provider} analysis for ${activePhotos.length} photos.`);
    const abortController = new AbortController();
    analysisAbortRef.current = abortController;

    const batchSize = getBatchSizeForProvider(provider);
    const batches = [];
    for (let i = 0; i < activePhotos.length; i += batchSize) {
      batches.push(activePhotos.slice(i, i + batchSize));
    }

    const byId = new Map(activePhotos.map((p, idx) => [p.id, idx]));

    try {
      const merged = [];
      const fingerprintByIndex = new Map();
      const totalUnits = Math.max(1, activePhotos.length * 2);
      let completedUnits = 0;
      const updateProgress = (label, increment = 0) => {
        completedUnits += increment;
        const pct = Math.round((completedUnits / totalUnits) * 100);
        setAnalysisProgress(clamp(pct, 0, 100));
        setAnalysisProgressLabel(label);
      };

      appendLog("info", "Computing image fingerprints (hash/color/metadata) for uniqueness model...");
      setAnalysisProgressLabel("Computing fingerprints...");
      for (let i = 0; i < activePhotos.length; i += 1) {
        throwIfAborted(abortController.signal);
        try {
          const fingerprint = await buildImageFingerprint(activePhotos[i].file);
          fingerprintByIndex.set(i, fingerprint);
        } catch (error) {
          if (isAbortError(error)) throw error;
          appendLog("error", `Fingerprint compute failed for ${activePhotos[i].name}: ${errorToMessage(error)}`);
        }
        updateProgress(`Computing fingerprints (${i + 1}/${activePhotos.length})`, 1);
      }
      appendLog("info", `Computed fingerprints: ${fingerprintByIndex.size}/${activePhotos.length}.`);

      for (let i = 0; i < batches.length; i += 1) {
        throwIfAborted(abortController.signal);
        const batch = batches[i];
        appendLog("info", `Analyzing batch ${i + 1}/${batches.length}...`);

        const results =
          provider === "cloudflare"
            ? await runCloudflareBatch({
                photos: batch,
                themePrompt,
                appendLog,
                endpoint: cfProxyEndpoint?.trim() || CLOUDFLARE_ENDPOINT,
                proxyToken: cfProxyToken,
                model: cfProxyModel,
                signal: abortController.signal
              })
            : provider === "openai"
              ? await runOpenAIProxyBatch({
                  photos: batch,
                  themePrompt,
                  appendLog,
                  endpoint: cfProxyEndpoint?.trim() || CLOUDFLARE_ENDPOINT,
                  proxyToken: cfProxyToken,
                  model: openaiModel,
                  signal: abortController.signal
                })
            : provider === "cloudflare-direct"
              ? await runCloudflareDirectBatch({
                  photos: batch,
                  themePrompt,
                  appendLog,
                  accountId: cfAccountId,
                  apiKey: cfApiKey,
                  model: cfModel,
                  signal: abortController.signal
                })
              : provider === "openai-direct"
                ? await runOpenAIDirectBatch({
                    photos: batch,
                    themePrompt,
                    appendLog,
                    apiKey: openaiApiKey,
                    model: openaiModel,
                    baseUrl: openaiBaseUrl,
                    signal: abortController.signal
                  })
            : await runLocalBatch({
                photos: batch,
                themePrompt,
                appendLog,
                signal: abortController.signal
              });

        if (!Array.isArray(results)) {
          throw new Error(`Batch ${i + 1} returned invalid payload: expected JSON array.`);
        }

        const normalized = results.map((item, idx) => {
          const batchPhoto = batch[idx];
          if (!batchPhoto) {
            throw new Error(`Batch ${i + 1} returned more results than images sent.`);
          }
          const mappedIndex = byId.get(batchPhoto.id);

          return {
            index: mappedIndex,
            modelIndex: typeof item.index === "number" ? item.index : idx,
            quality: clamp(Number(item.quality || 0), 0, 100),
            composition: clamp(Number(item.composition || 0), 0, 100),
            emotion: clamp(Number(item.emotion || 0), 0, 100),
            uniqueness: clamp(Number(item.uniqueness || 0), 0, 100),
            overall: clamp(Number(item.overall || 0), 0, 100),
            scene: String(item.scene || "unknown"),
            tags: Array.isArray(item.tags) ? item.tags.map((t) => String(t)) : [],
            descriptor: normalizeDescriptor(item.descriptor, item.scene, item.tags),
            reason: String(item.reason || "No reason provided."),
            albumWorthy: Boolean(item.albumWorthy)
          };
        });

        merged.push(...normalized);
        appendLog("info", `Batch ${i + 1} raw parsed result: ${JSON.stringify(normalized)}`);
        appendLog("info", `Batch ${i + 1} completed.`);
        updateProgress(`Analyzing images (${merged.length}/${activePhotos.length})`, batch.length);
      }

      setAnalysisProgressLabel("Optimizing final album selection...");
      const desired = clamp(albumSize, MIN_ALBUM, MAX_ALBUM);
      const pickedCount = Math.min(desired, merged.length);
      const { selectedIndexes: topIndexes, scored, stats: selectionStats, diagnostics } = applySmartSelection({
        analyses: merged,
        targetCount: pickedCount,
        fingerprintByIndex,
        themePrompt,
        lookalikeThreshold,
        clusterStrictness,
        themeStrictness,
        maxPerCluster
      });
      const selected = activePhotos
        .filter((p, index) => topIndexes.includes(index))
        .map((p) => p.id);

      setPhotos((prev) =>
        prev.map((p) => {
          const index = byId.get(p.id);
          const analysis = scored.find((item) => item.index === index);
          if (!analysis) return p;
          const diversityPenalty =
            analysis.scenePenalty +
            analysis.duplicatePenalty +
            analysis.lookalikePenalty +
            analysis.themePenalty +
            (analysis.metadataPenalty || 0);
          return {
            ...p,
            analysis: {
              ...analysis,
              diversityPenalty,
              adjustedOverall: analysis.adjustedOverall
            }
          };
        })
      );

      setSelectedIds(selected);
      const scoredByIndex = new Map(scored.map((s) => [s.index, s]));
      const clusterRows = (diagnostics?.clusters || []).map((cluster) => ({
        ...cluster,
        items: (cluster.topIndexes || []).map((idx) => {
          const photo = activePhotos[idx];
          const analysis = scoredByIndex.get(idx);
          return {
            index: idx,
            id: photo?.id,
            name: photo?.name || `Photo ${idx + 1}`,
            previewUrl: photo?.previewUrl,
            selected: selected.includes(photo?.id),
            score: Math.round(analysis?.adjustedOverall || 0),
            visibilityScore: Math.round(analysis?.visibilityScore || 0),
            champion: Boolean(analysis?.clusterChampion)
          };
        })
      }));
      setClusterReport(clusterRows);

      const insights = [];
      if ((selectionStats?.clustersWithMultiSelected || 0) > 0) {
        insights.push(
          `${selectionStats.clustersWithMultiSelected} cluster(s) still have multiple selected photos (likely remaining lookalikes).`
        );
      }
      if ((selectionStats?.selectedLowVisibility || 0) > 0) {
        insights.push(`${selectionStats.selectedLowVisibility} selected photo(s) have low visibility and may be weak picks.`);
      }
      if ((selectionStats?.selectedNonChampion || 0) > 0) {
        insights.push(`${selectionStats.selectedNonChampion} selected photo(s) are non-champions from their clusters.`);
      }
      if (!insights.length) {
        insights.push("No major cluster conflicts detected in this run.");
      }
      setDiagnosticInsights(insights);

      appendLog(
        "info",
        `Uniqueness stats: clusters=${selectionStats?.clusters || 0}, duplicate-suppressed=${selectionStats?.duplicateSuppressed || 0}, post-process-removed=${selectionStats?.postProcessRemoved || 0}, low-visibility-filtered=${selectionStats?.lowVisibilityFiltered || 0}, theme-blocked=${selectionStats?.blockedByTheme || 0}, max-per-cluster=${selectionStats?.maxPerCluster || maxPerCluster}, lookalike-threshold=${(selectionStats?.lookalikeThreshold || lookalikeThreshold).toFixed(2)}, post-threshold=${(selectionStats?.postProcessThreshold || lookalikeThreshold).toFixed(2)}.`
      );
      appendLog("success", `Selected ${selected.length} photos after relevance + metadata + uniqueness MMR pass.`);
      setAnalysisProgress(100);
      setAnalysisProgressLabel("Done");
      setCurrentStep(3);
    } catch (error) {
      if (isAbortError(error)) {
        appendLog("info", "Analysis stopped by user.");
        setAnalysisProgressLabel("Stopped");
      } else {
        const message = error instanceof Error ? error.message : String(error);
        setErrorMessage(message);
        appendLog("error", message);
      }
    } finally {
      analysisAbortRef.current = null;
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="app-shell" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <header className="hero">
        <h1>AI Album Selector</h1>
        <p>Pick your strongest album photos with AI scoring and diversity-aware ranking.</p>
      </header>

      <section className="progress">
        {["Upload", "Configure", "Results"].map((label, idx) => {
          const stepNumber = idx + 1;
          const isCompleted =
            stepNumber === 1
              ? canAdvanceToConfigure
              : stepNumber === 2
                ? canShowResults
                : selectedIds.length > 0;
          return (
            <div key={label} className={`step ${currentStep === stepNumber ? "active" : ""} ${isCompleted ? "done" : ""}`}>
              <span>{stepNumber}</span>
              <p>{label}</p>
            </div>
          );
        })}
      </section>

      {errorMessage && <div className="error-banner">API/Error: {errorMessage}</div>}

      {currentStep === 1 ? (
        <section className="step-screen card">
          <div className="step-screen-head">
            <div>
              <small>Step 1</small>
              <h2>Upload Photos</h2>
            </div>
            <p>Add the full trip folder first. Once the import is done, move to configuration.</p>
          </div>

          {isPreparingFiles ? (
            <div className="analysis-progress-wrap">
              <div className="analysis-progress-head">
                <strong>Preparing Images</strong>
                <span>{filePrepProgress}%</span>
              </div>
              <div className="analysis-progress-track">
                <div className="analysis-progress-fill" style={{ width: `${filePrepProgress}%` }} />
              </div>
              <small>{filePrepLabel}</small>
            </div>
          ) : null}

          <div className="upload-zone" onClick={() => inputRef.current?.click()}>
            <strong>Drag & drop images here</strong>
            <p>Supports PNG, JPEG, WebP, GIF, BMP, TIFF, HEIC, HEIF, and browser-supported image types. Max 1,000.</p>
            <button type="button">Choose Files</button>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={onPickFiles}
              hidden
            />
          </div>

          <section className="stats card">
            <div>
              <small>Imported</small>
              <strong>{stats.uploaded}</strong>
            </div>
            <div>
              <small>Ready For Next Step</small>
              <strong>{canAdvanceToConfigure ? "Yes" : "No"}</strong>
            </div>
            <div>
              <small>Preparing</small>
              <strong>{isPreparingFiles ? `${filePrepProgress}%` : "Done"}</strong>
            </div>
            <div>
              <small>Album Goal</small>
              <strong>{albumSize}</strong>
            </div>
          </section>

          <div className="step-actions">
            <button type="button" className="analyze-btn" onClick={() => setCurrentStep(2)} disabled={!canAdvanceToConfigure}>
              Continue To Configure
            </button>
          </div>
        </section>
      ) : null}

      {currentStep === 2 ? (
        <section className="step-screen card">
          <div className="step-screen-head">
            <div>
              <small>Step 2</small>
              <h2>Configure Curation</h2>
            </div>
            <p>Choose the album size, provider, and selection rules before running the curation pass.</p>
          </div>

          {isAnalyzing ? (
            <div className="analysis-progress-wrap">
              <div className="analysis-progress-head">
                <strong>Processing Progress</strong>
                <span>{analysisProgress}%</span>
              </div>
              <div className="analysis-progress-track">
                <div className="analysis-progress-fill" style={{ width: `${analysisProgress}%` }} />
              </div>
              <small>{analysisProgressLabel}</small>
            </div>
          ) : null}

          <section className="stats card">
            <div>
              <small>Imported</small>
              <strong>{stats.uploaded}</strong>
            </div>
            <div>
              <small>Selected</small>
              <strong>{stats.selected}</strong>
            </div>
            <div>
              <small>Average Score</small>
              <strong>{stats.avg.toFixed(1)}</strong>
            </div>
            <div>
              <small>Selection Rate</small>
              <strong>{stats.rate.toFixed(1)}%</strong>
            </div>
          </section>

          <div className="config-grid">
          <label>
            Album Size ({albumSize} photos)
            <input
              type="range"
              min={MIN_ALBUM}
              max={MAX_ALBUM}
              step="1"
              value={albumSize}
              onChange={(e) => setAlbumSize(clamp(Number(e.target.value || MIN_ALBUM), MIN_ALBUM, MAX_ALBUM))}
            />
            <small className="control-hint">Example: `10` for a tight highlight reel, `80` for a fuller story album.</small>
          </label>

          <label>
            Analyzer Provider
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="local">Local (Free, no API key)</option>
              <option value="cloudflare">Cloudflare endpoint (proxy)</option>
              <option value="openai">OpenAI endpoint (proxy)</option>
              <option value="cloudflare-direct">Cloudflare Direct (API key)</option>
              <option value="openai-direct">OpenAI Direct (API key)</option>
            </select>
          </label>

          {provider === "cloudflare" || provider === "openai" ? (
            <>
              <label>
                Proxy Endpoint
                <input
                  type="text"
                  placeholder="https://<worker>.workers.dev"
                  value={cfProxyEndpoint}
                  onChange={(e) => setCfProxyEndpoint(normalizeEndpointUrl(e.target.value))}
                />
              </label>
              <label>
                Proxy Bearer Token (Optional)
                <input
                  type="password"
                  placeholder="If your proxy requires auth"
                  value={cfProxyToken}
                  onChange={(e) => setCfProxyToken(e.target.value)}
                />
              </label>
              <label>
                {provider === "cloudflare" ? "Proxy Model Hint" : "OpenAI Model"}
                <input
                  type="text"
                  placeholder={provider === "cloudflare" ? "@cf/meta/llama-3.2-11b-vision-instruct" : "gpt-4.1-mini"}
                  value={provider === "cloudflare" ? cfProxyModel : openaiModel}
                  onChange={(e) => (provider === "cloudflare" ? setCfProxyModel(e.target.value) : setOpenaiModel(e.target.value))}
                />
              </label>
            </>
          ) : null}

          {provider === "cloudflare-direct" ? (
            <>
              <label>
                Cloudflare Account ID
                <input
                  type="text"
                  placeholder="e.g. 123abc..."
                  value={cfAccountId}
                  onChange={(e) => setCfAccountId(e.target.value)}
                />
              </label>
              <label>
                Cloudflare API Key / Token
                <input
                  type="password"
                  placeholder="Paste token"
                  value={cfApiKey}
                  onChange={(e) => setCfApiKey(e.target.value)}
                />
              </label>
              <label>
                Cloudflare Vision Model
                <input
                  type="text"
                  placeholder="@cf/meta/llama-3.2-11b-vision-instruct"
                  value={cfModel}
                  onChange={(e) => setCfModel(e.target.value)}
                />
              </label>
            </>
          ) : null}

          {provider === "openai-direct" ? (
            <>
              <label>
                OpenAI API Key
                <input
                  type="password"
                  placeholder="sk-..."
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                />
              </label>
              <label>
                OpenAI Model
                <input
                  type="text"
                  placeholder="gpt-4.1-mini"
                  value={openaiModel}
                  onChange={(e) => setOpenaiModel(e.target.value)}
                />
              </label>
              <label>
                OpenAI Base URL
                <input
                  type="text"
                  placeholder="https://api.openai.com/v1/chat/completions"
                  value={openaiBaseUrl}
                  onChange={(e) => setOpenaiBaseUrl(normalizeEndpointUrl(e.target.value))}
                />
              </label>
            </>
          ) : null}

          <label>
            Theme Prompt (Optional)
            <textarea
              rows={3}
              placeholder="e.g. warm candid family moments, outdoor golden hour"
              value={themePrompt}
              onChange={(e) => setThemePrompt(e.target.value)}
            />
          </label>

          <label>
            Lookalike Threshold ({lookalikeThreshold.toFixed(2)})
            <input
              type="range"
              min="0.75"
              max="0.95"
              step="0.01"
              value={lookalikeThreshold}
              onChange={(e) => setLookalikeThreshold(Number(e.target.value))}
            />
            <small className="control-hint">{lookalikeHint}</small>
          </label>

          <label>
            Cluster Strictness
            <select value={clusterStrictness} onChange={(e) => setClusterStrictness(Number(e.target.value))}>
              <option value={1}>Loose</option>
              <option value={2}>Balanced</option>
              <option value={3}>Strict</option>
            </select>
            <small className="control-hint">{clusterHint}</small>
          </label>

          <label>
            Theme Strictness
            <select value={themeStrictness} onChange={(e) => setThemeStrictness(Number(e.target.value))}>
              <option value={1}>Soft</option>
              <option value={2}>Balanced</option>
              <option value={3}>Hard</option>
            </select>
            <small className="control-hint">{themeHint}</small>
          </label>

          <label>
            Max Per Cluster
            <select value={maxPerCluster} onChange={(e) => setMaxPerCluster(Number(e.target.value))}>
              <option value={1}>1 (Very Strict)</option>
              <option value={2}>2 (Recommended)</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
            </select>
            <small className="control-hint">{maxPerClusterHint}</small>
          </label>

          <button type="button" className="analyze-btn" onClick={startAnalysis} disabled={isAnalyzing || !activePhotos.length}>
            {isAnalyzing
              ? "Analyzing..."
              : `Run ${
                    provider === "cloudflare"
                      ? "Cloudflare Proxy"
                    : provider === "cloudflare-direct"
                      ? "Cloudflare Direct"
                      : provider === "openai-direct"
                        ? "OpenAI Direct"
                      : "Local"
                } Analysis`}
          </button>
          {isAnalyzing ? (
            <button type="button" className="stop-btn" onClick={stopAnalysis}>
              Stop Analysis
            </button>
          ) : null}
          <button type="button" className="secondary-btn" onClick={() => setCurrentStep(1)} disabled={isAnalyzing}>
            Back To Upload
          </button>
        </div>
          <section className="log-console card">
            <h2>Live Analysis Log</h2>
            <div className="log-body">
              {logs.length ? logs.map((line, i) => <pre key={`${line}-${i}`}>{line}</pre>) : <pre>No activity yet.</pre>}
            </div>
          </section>
        </section>
      ) : null}

      {currentStep === 3 ? (
        <section className="step-screen results-screen">
          <div className="step-screen-head card">
            <div>
              <small>Step 3</small>
              <h2>Final Selection</h2>
            </div>
            <p>Review the chosen album photos first, then inspect the collapsed clusters underneath if you want to compare near-duplicates.</p>
          </div>

          <section className="stats card">
            <div>
              <small>Selected</small>
              <strong>{stats.selected}</strong>
            </div>
            <div>
              <small>Total Uploaded</small>
              <strong>{stats.uploaded}</strong>
            </div>
            <div>
              <small>Average Score</small>
              <strong>{stats.avg.toFixed(1)}</strong>
            </div>
            <div>
              <small>Selection Rate</small>
              <strong>{stats.rate.toFixed(1)}%</strong>
            </div>
          </section>

          <section className="preview card">
            <h2>Preview Grid</h2>
            <div className="grid">
              {activePhotos
                .filter((p) => selectedIds.includes(p.id))
                .map((photo) => (
                  <article key={photo.id} className="photo-card" onClick={() => setActivePhotoId(photo.id)}>
                    <img src={photo.previewUrl} alt={photo.name} />
                    <button
                      type="button"
                      className="remove-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        removePhoto(photo.id);
                      }}
                    >
                      Remove
                    </button>
                    <div className="card-meta">
                      <span className="badge">{Math.round(photo.analysis?.adjustedOverall || 0)}</span>
                      <div className="tags">
                        {(photo.analysis?.tags || []).slice(0, 4).map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                    </div>
                  </article>
                ))}
            </div>
            {!selectedIds.length && <p>No selected photos yet. Run analysis to build a preview set.</p>}
          </section>

          <section className="diagnostics card">
            <h2>Selection Diagnostics</h2>
            <div className="insights">
              {diagnosticInsights.map((line, i) => (
                <p key={`${line}-${i}`}>{line}</p>
              ))}
            </div>
            <div className="cluster-list">
              {clusterReport.map((cluster) => {
                const isOpen = openClusterIds.includes(cluster.clusterId);
                return (
                  <article key={cluster.clusterId} className="cluster-item">
                    <button type="button" className="cluster-toggle" onClick={() => toggleCluster(cluster.clusterId)}>
                      <div className="cluster-head">
                        <strong>Cluster {cluster.clusterId + 1}</strong>
                        <span>
                          size: {cluster.size} | selected: {cluster.selectedCount} | {isOpen ? "Hide" : "Show"}
                        </span>
                      </div>
                    </button>
                    {isOpen ? (
                      <div className="cluster-thumbs">
                        {cluster.items.map((item) => (
                          <div key={`${cluster.clusterId}-${item.index}`} className={`cluster-thumb ${item.selected ? "selected" : ""}`}>
                            {item.previewUrl ? <img src={item.previewUrl} alt={item.name} /> : null}
                            <small>
                              {item.score} | vis {item.visibilityScore}
                              {item.champion ? " | champ" : ""}
                            </small>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </section>

          <div className="step-actions">
            <button type="button" className="secondary-btn" onClick={() => setCurrentStep(2)}>
              Back To Configure
            </button>
          </div>
        </section>
      ) : null}

      {activePhoto?.analysis && (
        <div className="modal-backdrop" onClick={() => setActivePhotoId(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="close" onClick={() => setActivePhotoId(null)}>
              Close
            </button>
            <img src={activePhoto.previewUrl} alt={activePhoto.name} />
            <h3>{activePhoto.name}</h3>
            <p>{activePhoto.analysis.reason}</p>
            <div className="score-bars">
              <ScoreBar label="Quality" value={activePhoto.analysis.quality} />
              <ScoreBar label="Composition" value={activePhoto.analysis.composition} />
              <ScoreBar label="Emotion" value={activePhoto.analysis.emotion} />
              <ScoreBar label="Uniqueness" value={activePhoto.analysis.uniqueness} />
              <ScoreBar label="Overall" value={activePhoto.analysis.overall} />
              <ScoreBar label="Adjusted" value={activePhoto.analysis.adjustedOverall} />
            </div>
          </div>
        </div>
      )}

      <footer>
        Provider: <code>{provider}</code>
        {provider === "cloudflare" ? (
          <>
            {" "}| Endpoint: <code>{cfProxyEndpoint || CLOUDFLARE_ENDPOINT}</code> | Model: <code>{cfProxyModel || DEFAULT_CF_MODEL}</code>
          </>
        ) : null}
        {provider === "openai" ? (
          <>
            {" "}| Endpoint: <code>{cfProxyEndpoint || CLOUDFLARE_ENDPOINT}</code> | OpenAI Model: <code>{openaiModel || DEFAULT_OPENAI_MODEL}</code>
          </>
        ) : null}
        {provider === "cloudflare-direct" ? (
          <>
            {" "}| Account: <code>{cfAccountId || "(not set)"}</code> | Model: <code>{cfModel || DEFAULT_CF_MODEL}</code>
          </>
        ) : null}
        {provider === "openai-direct" ? (
          <>
            {" "}| OpenAI Model: <code>{openaiModel || DEFAULT_OPENAI_MODEL}</code> | URL:{" "}
            <code>{openaiBaseUrl || DEFAULT_OPENAI_BASE_URL}</code>
          </>
        ) : null}
      </footer>
    </div>
  );
}
