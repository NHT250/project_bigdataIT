const MAX_RESULTS = 20;
const form = document.getElementById("search");
const input = document.getElementById("q");
const results = document.getElementById("results");
const statusLine = document.getElementById("status");
const searchButton = form.querySelector("button");

const STORAGE_KEY = "search";

let latestSearch = 0;

function describe(count, query) {
  if (count === 0) return `No articles match "${query}". Try a broader word.`;
  if (count === MAX_RESULTS) return `Showing the ${MAX_RESULTS} best matches for "${query}".`;
  return `${count} article${count === 1 ? "" : "s"} match "${query}".`;
}

function isArticle(article) {
  return article && Object.hasOwn(ORIGIN_NAMES, article.source) && typeof article.source_id === "string"
    && typeof article.title === "string" && typeof article.published_at === "string"
    && (article.url === null || typeof article.url === "string")
    && (article.author === null || typeof article.author === "string")
    && (article.summary === null || typeof article.summary === "string")
    && (article.score === null || Number.isInteger(article.score));
}

function showResults(query, articles) {
  results.replaceChildren(...articles.map(articleNode));
  statusLine.textContent = describe(articles.length, query);
}

function restore() {
  const saved = loadSaved(STORAGE_KEY, null);
  if (!saved || typeof saved.query !== "string" || !saved.query.trim()
      || !Array.isArray(saved.results) || !saved.results.every(isArticle)) return;
  input.value = saved.query;
  showResults(saved.query, saved.results.slice(0, MAX_RESULTS));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  const search = ++latestSearch;
  statusLine.classList.remove("is-error");
  statusLine.textContent = "";
  results.setAttribute("aria-busy", "true");
  results.replaceChildren(...skeletonCards(3));
  searchButton.disabled = true;
  try {
    const body = await fetchJson(`/api/search?limit=${MAX_RESULTS}&q=${encodeURIComponent(query)}`);
    if (search !== latestSearch) return;
    showResults(query, body.results);
    save(STORAGE_KEY, { query, results: body.results });
  } catch (error) {
    if (search !== latestSearch) return;
    results.replaceChildren();
    statusLine.classList.add("is-error");
    statusLine.textContent = error.message;
  } finally {
    if (search === latestSearch) {
      results.removeAttribute("aria-busy");
      searchButton.disabled = false;
    }
  }
});

restore();
