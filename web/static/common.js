// Article text comes from the web, so every page builds DOM nodes and never uses innerHTML.
const ORIGIN_NAMES = { arxiv: "arXiv", hn: "Hacker News" };

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
  const reason = article.source === "hn"
    ? "Hacker News stories have no abstract to summarize."
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
    el("span", { className: "origin", textContent: ORIGIN_NAMES[article.source] }),
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

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : `The server returned ${response.status}.`);
  }
  return body;
}
