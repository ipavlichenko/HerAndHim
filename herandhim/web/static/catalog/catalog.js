"use strict";
const tg = window.Telegram?.WebApp;
tg?.ready();
tg?.expand();
const inTelegram = Boolean(tg && tg.platform && tg.platform !== "unknown");
const grid = document.querySelector("#grid");
const modal = document.querySelector("#profile");
const choose = document.querySelector("#choose");
let catalogue = [], selected = null, style = "all", botUrl = "https://t.me/sara_herandhim_bot";
function text(id, value) { document.querySelector(id).textContent = value; }
function closeProfile() { modal.close(); tg?.BackButton?.hide(); }
function profile(item) {
  selected = item;
  const image = document.querySelector("#profile-image");
  image.src = item.portrait; image.alt = `${item.name}, ${item.age}`;
  text("#profile-style", `${item.style === "real" ? "РЕАЛИЗМ" : "АНИМЕ"} · ВЫМЫШЛЕННЫЙ AI-ПЕРСОНАЖ`);
  text("#profile-name", `${item.name}, ${item.age}`);
  text("#profile-tagline", item.tagline);
  text("#profile-description", item.description);
  const tags = document.querySelector("#profile-interests"); tags.replaceChildren();
  for (const interest of item.interests) { const tag = document.createElement("span"); tag.textContent = interest; tags.append(tag); }
  choose.disabled = false;
  choose.textContent = inTelegram ? `Выбрать: ${item.name}` : "Открыть бота в Telegram";
  text("#choose-status", inTelegram ? "" : "Выбор героини доступен в каталоге внутри бота.");
  modal.showModal(); modal.scrollTop = 0; tg?.BackButton?.show();
}
function render() {
  const age = document.querySelector("#age").value;
  let items = catalogue.filter(c => (style === "all" || c.style === style) &&
    (age === "all" || Math.floor(c.age / 10) * 10 === Number(age)));
  if (style === "all") {
    const real = items.filter(c => c.style === "real");
    const anime = items.filter(c => c.style === "anime");
    items = [];
    for (let i = 0; i < Math.max(real.length, anime.length); i++) {
      if (real[i]) items.push(real[i]);
      if (anime[i]) items.push(anime[i]);
    }
  }
  grid.replaceChildren();
  text("#result-count", `Героинь: ${items.length}`);
  document.querySelector("#empty").hidden = items.length > 0;
  items.forEach((item, index) => {
    const card = document.createElement("button"); card.className = "card"; card.type = "button";
    card.setAttribute("aria-label", `${item.name}, ${item.age}. ${item.tagline}. Открыть карточку`);
    const photo = document.createElement("div"); photo.className = "photo";
    const image = document.createElement("img"); image.src = item.portrait; image.alt = "";
    image.loading = index < 4 ? "eager" : "lazy"; image.decoding = "async";
    const badge = document.createElement("span"); badge.className = "style-badge"; badge.textContent = item.style === "real" ? "Реализм" : "Аниме";
    photo.append(image, badge);
    const title = document.createElement("h2"); title.textContent = item.name;
    const ageNode = document.createElement("span"); ageNode.textContent = ` ${item.age}`; title.append(ageNode);
    const description = document.createElement("p"); description.textContent = item.tagline;
    card.append(photo, title, description); card.addEventListener("click", () => profile(item)); grid.append(card);
  });
}
async function load() {
  document.querySelector("#error").hidden = true;
  try {
    const response = await fetch("catalog.json");
    if (!response.ok) throw new Error("catalog_unavailable");
    const data = await response.json(); catalogue = data.characters; botUrl = data.bot_url;
    document.querySelectorAll("[data-bot-link]").forEach(link => { link.href = botUrl; });
    render();
  } catch (_) { text("#result-count", "Каталог временно недоступен"); document.querySelector("#error").hidden = false; }
}
document.querySelectorAll("[data-style]").forEach(button => button.addEventListener("click", () => {
  style = button.dataset.style;
  document.querySelectorAll("[data-style]").forEach(b => { b.classList.toggle("selected", b === button); b.setAttribute("aria-pressed", String(b === button)); });
  render();
}));
document.querySelector("#age").addEventListener("change", render);
document.querySelector("#close").addEventListener("click", closeProfile);
document.querySelector("#reload").addEventListener("click", load);
modal.addEventListener("cancel", () => tg?.BackButton?.hide());
tg?.BackButton?.onClick(closeProfile);
choose.addEventListener("click", () => {
  if (!selected) return;
  if (!inTelegram) { window.location.href = botUrl; return; }
  choose.disabled = true;
  try { tg.sendData(JSON.stringify({v: 1, character_id: selected.id})); }
  catch (_) { choose.disabled = false; text("#choose-status", "Не удалось передать выбор. Открой каталог кнопкой в чате бота и попробуй снова."); }
});
load();
