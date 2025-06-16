console.log("🟡 content.js is loaded on", window.location.hostname);

chrome.runtime.onMessage.addListener((msg) => {
  console.log("📥 content.js received message:", msg);
  if (msg.action === "scrape_cart") {
    console.log("🛠 scrape_cart triggered");
    let items = [];

    if (window.location.hostname.includes("amazon")) {
      document.querySelectorAll(".sc-list-item-content").forEach(el => {
        let name = el.querySelector("span.a-truncate-full")?.innerText;
        let link = el.querySelector("a")?.href;
        if (name) items.push({ name, link });
      });
    } else if (window.location.hostname.includes("flipkart")) {
      document.querySelectorAll("._2nQDXZ").forEach(el => {
        let name = el.querySelector("a")?.innerText;
        let link = el.querySelector("a")?.href;
        if (name) items.push({ name, link });
      });
    } else if (window.location.hostname.includes("myntra")) {
      document.querySelectorAll(".itemContainer-base-item").forEach(el => {
        let name = el.querySelector(".itemContainer-base-brand")?.innerText;
        let link = el.querySelector("a")?.href;
        if (name) items.push({ name, link });
      });
    }

    console.log("🛍 Found items:", items);

    if (items.length === 0) {
      alert("❌ No cart items found on this page.");
      return;
    }

    fetch("http://localhost:5000/track", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: msg.user.email,
        items,
        email: msg.user.email,
        phone: msg.user.phone,
        name: msg.user.name
      })
    })
    .then(() => console.log("📤 Sent cart data to Flask"))
    .catch(err => console.error("❌ Fetch failed:", err));
  }
});
