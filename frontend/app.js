const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");

function setStatus(message) {
  statusEl.textContent = message || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function recipeCard(recipe) {
  const missing = (recipe.missing_ingredients || [])
    .map((item) => `<span class="miss">eksik: ${escapeHtml(item)}</span>`)
    .join("");
  const overlap =
    recipe.overlap_percentage == null
      ? ""
      : `<div class="badge">%${escapeHtml(recipe.overlap_percentage)} uyum</div>`;
  const preview = escapeHtml((recipe.ingredients || []).slice(0, 5).join(", "));
  return `
    <article class="card">
      ${overlap}
      <h3>${escapeHtml(recipe.title)}</h3>
      <div class="meta">${escapeHtml(recipe.diet)} · ${preview}</div>
      <div>${missing}</div>
    </article>
  `;
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((el) => el.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
    setStatus("");
  });
});

document.getElementById("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const q = document.getElementById("q").value.trim();
  setStatus("Aranıyor...");
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error("Arama başarısız");
    const data = await res.json();
    answerEl.hidden = false;
    answerEl.textContent = data.answer;
    document.getElementById("search-results").innerHTML = data.recipes
      .map(recipeCard)
      .join("");
    setStatus(data.recipes.length ? "" : "Sonuç yok.");
  } catch (err) {
    setStatus(err.message || "Hata");
  }
});

document.getElementById("match-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const ingredients = document.getElementById("bag").value.trim();
  setStatus("Eşleştiriliyor...");
  try {
    const res = await fetch("/api/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ingredients }),
    });
    if (!res.ok) throw new Error("Eşleştirme başarısız");
    const data = await res.json();
    document.getElementById("match-results").innerHTML = data.map(recipeCard).join("");
    setStatus(data.length ? "" : "Eşleşen tarif yok.");
  } catch (err) {
    setStatus(err.message || "Hata");
  }
});

async function loadGallery() {
  try {
    const res = await fetch("/api/recipes");
    if (!res.ok) throw new Error("Galeri yüklenemedi");
    const data = await res.json();
    document.getElementById("gallery").innerHTML = data.map(recipeCard).join("");
  } catch (err) {
    setStatus(err.message || "Hata");
  }
}

loadGallery();
