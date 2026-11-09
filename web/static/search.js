const MAX_RESULTS = 20;
const form = document.getElementById("search");
const input = document.getElementById("q");
const results = document.getElementById("results");
const statusLine = document.getElementById("status");

let latestSearch = 0;

function describe(count, query) {
  if (count === 0) return `No articles match "${query}". Try a broader word.`;
  if (count === MAX_RESULTS) return `Showing the ${MAX_RESULTS} best matches for "${query}".`;
  return `${count} article${count === 1 ? "" : "s"} match "${query}".`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  const search = ++latestSearch;
  statusLine.classList.remove("is-error");
  statusLine.textContent = "Searching";
  try {
    const body = await fetchJson(`/api/search?limit=${MAX_RESULTS}&q=${encodeURIComponent(query)}`);
    if (search !== latestSearch) return;
    results.replaceChildren(...body.results.map(articleNode));
    statusLine.textContent = describe(body.results.length, query);
  } catch (error) {
    if (search !== latestSearch) return;
    results.replaceChildren();
    statusLine.classList.add("is-error");
    statusLine.textContent = error.message;
  }
});
