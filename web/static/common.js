// Article text comes from the web, so every page builds DOM nodes and never uses innerHTML.
const ORIGIN_NAMES = { arxiv: "arXiv", hn: "Hacker News", devto: "Dev.to" };

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function safeUrl(url) {
  try {
    return ["http:", "https:"].includes(new URL(url).protocol) ? url : null;
  } catch {
    return null;
  }
}

function externalLink(url, props, ...children) {
  const href = safeUrl(url);
  if (!href) return el("span", props, ...children);
  return el("a", { ...props, href, target: "_blank", rel: "noopener noreferrer" }, ...children);
}

function timeAgo(iso) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso)) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function articleKey(article) {
  return `${article.source}:${article.source_id}`;
}

function articleSummary(article) {
  if (article.summary) return el("p", { className: "summary", textContent: article.summary });
  // Dev.to articles are not summarized by our model (it was trained on paper abstracts);
  // the author's own description is shown instead, labelled as theirs.
  if (article.source === "devto" && article.description) {
    return el("p", { className: "description" },
      el("span", { className: "description-label", textContent: "Author's description: " }), article.description);
  }
  const reason = article.source !== "arxiv"
    ? "Summaries are written for arXiv papers only."
    : "Summary not generated yet.";
  return el("p", { className: "no-summary", textContent: reason });
}

function shortAuthors(author) {
  const names = author.split(", ");
  return names.length > 2 ? `${names[0]} et al.` : author;
}

function articleNode(article) {
  const points = article.score === 1 ? "1 point" : `${article.score} points`;
  const meta = el("p", { className: "meta" },
    el("span", { className: `origin pill pill-${article.source}`, textContent: ORIGIN_NAMES[article.source] }),
    el("time", { dateTime: article.published_at, textContent: timeAgo(article.published_at) }),
    article.score ? el("span", { textContent: points }) : null,
    article.author ? el("span", { textContent: shortAuthors(article.author) }) : null,
  );
  return el("article", { className: `item item-${article.source}` },
    el("h2", {}, externalLink(article.url, { textContent: article.title })),
    articleSummary(article),
    meta,
  );
}

// Each page is its own HTML file, so what a page shows is kept in localStorage to survive
// moving between pages. Private browsing can block storage or make it throw, and stored
// data can be stale or edited, so every access is guarded and callers validate what they read.
const STORAGE_PREFIX = "techpulse.";

function loadSaved(key, fallback) {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function save(key, value) {
  try {
    localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
  } catch {
    // Storage is blocked or full: the page still works, it just won't be restored.
  }
}

function forget(key) {
  try {
    localStorage.removeItem(STORAGE_PREFIX + key);
  } catch {
    // Nothing to do when storage is unavailable.
  }
}

// Grey placeholder cards shown while articles load.
function skeletonCards(count) {
  return Array.from({ length: count }, () =>
    el("div", { className: "skeleton", ariaHidden: "true" }, el("span"), el("span"), el("span"), el("span")));
}

// Light/dark switch. The neon dark look is the default; the <head> of each page
// applies a saved choice of light before the page paints.
const THEME_KEY = "theme";

function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function icon(paths) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("aria-hidden", "true");
  for (const d of paths) {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

const SUN = ["M12 4V2", "M12 22v-2", "M4 12H2", "M22 12h-2", "M5.6 5.6 4.2 4.2", "M19.8 19.8l-1.4-1.4",
  "M5.6 18.4l-1.4 1.4", "M19.8 4.2l-1.4 1.4", "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"];
const MOON = ["M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"];

function setUpThemeToggle() {
  const header = document.querySelector(".top");
  if (!header) return;
  const button = el("button", { type: "button", className: "theme-toggle" });
  const show = () => {
    const dark = currentTheme() === "dark";
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    button.replaceChildren(icon(dark ? SUN : MOON));
    button.setAttribute("aria-label", label);
    button.title = label;
  };
  button.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    save(THEME_KEY, next);
    show();
    document.dispatchEvent(new CustomEvent("themechange"));
  });
  show();
  header.append(button);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : `The server returned ${response.status}.`);
  }
  return body;
}

setUpThemeToggle();
